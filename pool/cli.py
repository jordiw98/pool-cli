"""Pool CLI — entry point and command routing."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm

from pool import cache
from pool.models import PipelineMode, CostEstimate

app = typer.Typer(
    name="pool",
    help="Intelligent screenshot organizer — intention archaeology for your camera roll.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _check_api_keys(mode: PipelineMode) -> None:
    """Verify required API keys are set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Missing ANTHROPIC_API_KEY environment variable.[/red]")
        console.print("Export it: export ANTHROPIC_API_KEY=sk-...")
        raise typer.Exit(1)
    if mode == PipelineMode.FULL and not os.environ.get("GOOGLE_API_KEY"):
        console.print("[red]Missing GOOGLE_API_KEY environment variable (needed for full mode).[/red]")
        console.print("Export it: export GOOGLE_API_KEY=...")
        console.print("Or use --quick to skip Gemini and run with Claude only.")
        raise typer.Exit(1)


def _estimate_cost(unique_count: int, mode: PipelineMode) -> CostEstimate:
    """Rough cost estimate based on image count and mode."""
    if mode == PipelineMode.QUICK:
        return CostEstimate(anthropic_usd=1.0, google_usd=0.0)
    elif mode == PipelineMode.SMALL:
        # All images to Claude for discovery, SigLIP + OCR for classification
        anthropic = max(0.50, unique_count * 0.01)  # ~$0.01 per image for discovery
        return CostEstimate(anthropic_usd=round(anthropic, 2), google_usd=0.0)
    else:
        # Full mode
        gemini_images = int(unique_count * 0.12)  # ~12% go to tier 3
        anthropic = 5.0  # discovery + actions/narratives
        google = max(0.50, gemini_images * 0.001)  # ~$0.001 per Gemini call
        return CostEstimate(anthropic_usd=round(anthropic, 2), google_usd=round(google, 2))


def _determine_mode(unique_count: int, quick: bool) -> PipelineMode:
    if quick:
        return PipelineMode.QUICK
    elif unique_count < 200:
        return PipelineMode.SMALL
    else:
        return PipelineMode.FULL


@app.command()
def analyze(
    path: str = typer.Argument(..., help="Path to folder of screenshots"),
    quick: bool = typer.Option(False, "--quick", help="Quick mode: ~3 min, samples 30-50 images"),
    json_output: bool = typer.Option(False, "--json", help="Output full results as JSON"),
    csv_pool: Optional[str] = typer.Option(None, "--csv", help="Export a specific pool as CSV"),
    explain: Optional[str] = typer.Option(None, "--explain", help="Explain why an image is in its pool"),
    validate: bool = typer.Option(False, "--validate", help="Validate 30 random classifications interactively"),
    top_k: int = typer.Option(10, "--top-k", help="Top N matches per pool"),
    min_confidence: float = typer.Option(0.3, "--min-confidence", help="Confidence threshold"),
    cost_only: bool = typer.Option(False, "--cost", help="Estimate cost without running"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force full re-analysis"),
    no_api: bool = typer.Option(False, "--no-api", help="Fully local mode, no API calls"),
) -> None:
    """Analyze screenshots and organize them into intelligent Pools."""
    from pool.phases import scanner, embeddings, discovery, classifier, analyzer, output

    source_dir = Path(path).expanduser().resolve()
    if not source_dir.is_dir():
        console.print(f"[red]Not a directory: {source_dir}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print("[bold]pool[/bold]")
    console.print()

    # --- Connect to cache ---
    conn = cache.connect(str(source_dir))
    if no_cache:
        cache.clear_all(conn)

    # --- Phase 1: Scan ---
    t0 = time.time()
    console.print("Scanning screenshots…", end=" ")
    scan_result = scanner.scan(conn, str(source_dir))
    t_scan = time.time() - t0
    console.print(f"[dim]{t_scan:.1f}s[/dim]")

    total = scan_result["total"]
    unique = scan_result["unique"]
    dupes = scan_result["duplicates"]
    console.print(f"Found [bold]{total:,}[/bold] screenshots ({unique:,} unique, {dupes:,} duplicates)")

    # Store duplicates count for output module
    cache.set_state(conn, "duplicates_skipped", str(dupes))

    # --- Determine mode ---
    mode = _determine_mode(unique, quick)

    console.print(f"Mode: [bold]{mode.value}[/bold]")
    console.print()

    # --- Cost estimate + confirmation ---
    if not no_api:
        _check_api_keys(mode)
        estimate = _estimate_cost(unique, mode)

        if cost_only:
            console.print(f"Estimated cost: {estimate.display()}")
            raise typer.Exit(0)

        console.print(f"Estimated cost: [bold]{estimate.display()}[/bold]")
        if not Confirm.ask("Proceed?", default=True):
            console.print("Cancelled.")
            raise typer.Exit(0)
        console.print()
    elif cost_only:
        console.print("No API calls in --no-api mode. Cost: $0.00")
        raise typer.Exit(0)

    # --- Handle --explain separately ---
    if explain:
        result = classifier.explain_image(conn, explain, no_api=no_api)
        output.print_explanation(console, result)
        conn.commit()
        conn.close()
        raise typer.Exit(0)

    # --- Run pipeline based on mode ---
    if mode == PipelineMode.QUICK:
        _run_quick(conn, str(source_dir), top_k, no_api, console)
    elif mode == PipelineMode.SMALL:
        _run_small(conn, str(source_dir), top_k, min_confidence, no_api, console)
    else:
        _run_full(conn, str(source_dir), top_k, min_confidence, no_api, console)

    conn.commit()

    # --- Handle output flags ---
    if validate:
        output.run_validation(conn, console)
    elif json_output:
        output.print_json(conn, console)
    elif csv_pool:
        output.export_csv(conn, csv_pool, console)
    else:
        output.print_pools(conn, console, top_k=top_k)

    conn.close()


def _run_quick(conn, source_dir: str, top_k: int, no_api: bool, console: Console) -> None:
    """Quick mode: sample → Claude → output."""
    from pool.phases import discovery, output as out_mod

    t0 = time.time()
    console.print("Sampling and generating pools…", end=" ")
    discovery.quick_discover(conn, source_dir, no_api=no_api)
    t_disc = time.time() - t0
    console.print(f"[dim]{t_disc:.1f}s[/dim]")


def _run_small(conn, source_dir: str, top_k: int, min_confidence: float, no_api: bool, console: Console) -> None:
    """Small dataset mode: embeddings → Claude discovery (all images) → SigLIP + OCR classification → analysis."""
    from pool.phases import embeddings, discovery, classifier, analyzer

    t0 = time.time()
    console.print("Computing embeddings…", end=" ")
    embeddings.compute_embeddings(conn)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    t0 = time.time()
    console.print("Discovering pools…", end=" ")
    discovery.discover_pools(conn, source_dir, mode="small", no_api=no_api)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    t0 = time.time()
    console.print("Classifying…", end=" ")
    classifier.classify(conn, min_confidence=min_confidence, skip_gemini=True, no_api=no_api)
    cache.update_pool_counts(conn, top_k=top_k)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    t0 = time.time()
    console.print("Analyzing loops and generating actions…", end=" ")
    analyzer.analyze(conn, source_dir, no_api=no_api)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    console.print()


def _run_full(conn, source_dir: str, top_k: int, min_confidence: float, no_api: bool, console: Console) -> None:
    """Full pipeline: embeddings → clustering → discovery → cascade → analysis."""
    from pool.phases import embeddings, discovery, classifier, analyzer

    t0 = time.time()
    console.print("Computing embeddings…", end=" ")
    embeddings.compute_embeddings(conn)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    t0 = time.time()
    console.print("Clustering…", end=" ")
    embeddings.cluster(conn)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    t0 = time.time()
    console.print("Discovering pools…", end=" ")
    discovery.discover_pools(conn, source_dir, mode="full", no_api=no_api)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    t0 = time.time()
    console.print("Classifying (3-tier cascade)…", end=" ")
    classifier.classify(conn, min_confidence=min_confidence, skip_gemini=False, no_api=no_api)
    cache.update_pool_counts(conn, top_k=top_k)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    t0 = time.time()
    console.print("Analyzing loops and generating actions…", end=" ")
    analyzer.analyze(conn, source_dir, no_api=no_api)
    console.print(f"[dim]{time.time() - t0:.1f}s[/dim]")

    console.print()
