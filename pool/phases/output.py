"""Phase 6: Output — rich terminal display, JSON export, CSV export, validation, explanation."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from pool import cache
from pool.models import ClassificationBreakdown, LoopStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_ORDER: list[LoopStatus] = [
    LoopStatus.ACTIVE,
    LoopStatus.CYCLICAL,
    LoopStatus.ABANDONED,
    LoopStatus.ONE_SHOT,
    LoopStatus.UNKNOWN,
]

_STATUS_HEADERS: dict[LoopStatus, str] = {
    LoopStatus.ACTIVE: "Active",
    LoopStatus.CYCLICAL: "Comes & Goes",
    LoopStatus.ABANDONED: "Moved On",
    LoopStatus.ONE_SHOT: "One-Time",
    LoopStatus.UNKNOWN: "\u00af\\_(\u30c4)_/\u00af",
}


def _build_breakdown(conn) -> ClassificationBreakdown:
    """Build a ClassificationBreakdown from the raw cache data."""
    raw = cache.get_classification_breakdown(conn)  # {method: count}

    bd = ClassificationBreakdown()
    bd.siglip_high = raw.get("siglip", 0)
    bd.ocr_confirmed = raw.get("ocr", 0)
    bd.gemini_resolved = raw.get("gemini", 0)
    bd.claude_direct = raw.get("claude", 0)
    bd.unresolved = raw.get("noise", 0)

    # Duplicates come from pipeline_state if stored, or from images table
    dup_val = cache.get_state(conn, "duplicates_skipped")
    bd.duplicates_skipped = int(dup_val) if dup_val else 0

    bd.total_unique = bd.siglip_high + bd.ocr_confirmed + bd.gemini_resolved + bd.claude_direct + bd.unresolved
    return bd


def _basename(filepath: str) -> str:
    return Path(filepath).name


def _sort_pools_in_group(pools: list[dict]) -> list[dict]:
    """Within a status group, sort noise pool last, then by match_count desc."""
    noise = [p for p in pools if p.get("is_noise")]
    normal = [p for p in pools if not p.get("is_noise")]
    normal.sort(key=lambda p: p.get("match_count", 0), reverse=True)
    return normal + noise


def _pool_name_matches(pool_name: str, query: str) -> bool:
    """Case-insensitive partial match."""
    return query.lower() in pool_name.lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_pools(conn, console: Console, top_k: int = 10) -> None:
    """Main display: classification breakdown, insight, grouped pools, top matches."""

    pools = cache.get_pools(conn)
    temporals = cache.get_temporals(conn)
    actions = cache.get_actions(conn)

    temporal_by_pool: dict[str, dict] = {t["pool_id"]: t for t in temporals}
    action_by_pool: dict[str, dict] = {a["pool_id"]: a for a in actions}

    # ── Classification breakdown ──────────────────────────────────────────

    bd = _build_breakdown(conn)
    summary = bd.summary_lines()

    if summary:
        console.print()
        console.print("[bold]Classification[/bold]")

        # Determine widths for alignment
        max_label = max(len(label) for label, _, _ in summary)
        max_count = max(len(f"{count:,}") for _, count, _ in summary)

        for label, count, pct in summary:
            count_str = f"{count:,}".rjust(max_count)
            if pct > 0:
                pct_str = f"[dim]({pct:3.0%})[/dim]"
            else:
                pct_str = ""
            console.print(f"  {label:<{max_label}}  {count_str}  {pct_str}")

        console.print()

    # ── Opening insight ───────────────────────────────────────────────────

    insight = cache.get_state(conn, "opening_insight")
    if insight:
        console.print(f"[italic]\"{insight}\"[/italic]")
        console.print()

    # ── Group pools by loop_status ────────────────────────────────────────

    grouped: dict[LoopStatus, list[dict]] = {s: [] for s in _STATUS_ORDER}
    for pool in pools:
        temporal = temporal_by_pool.get(pool["id"])
        if temporal:
            try:
                status = LoopStatus(temporal.get("loop_status", "unknown"))
            except ValueError:
                status = LoopStatus.UNKNOWN
        else:
            status = LoopStatus.UNKNOWN
        grouped[status].append(pool)

    # ── Print each group ──────────────────────────────────────────────────

    for status in _STATUS_ORDER:
        group = grouped[status]
        if not group:
            continue

        group = _sort_pools_in_group(group)
        header = _STATUS_HEADERS[status]
        rule = "\u2500" * 3
        console.print(f"[bold dim]{rule} {header} {rule}[/bold dim]")
        console.print()

        for pool in group:
            pool_id = pool["id"]
            name = pool["name"]
            count = pool.get("match_count", 0)
            action_data = action_by_pool.get(pool_id)

            # Pool header
            console.print(f"[bold]{name}[/bold] [dim]\u2014 {count:,} matches[/dim]")

            # Action
            if action_data and action_data.get("action"):
                console.print(f"  Action: [green]\"{action_data['action']}\"[/green]")
            else:
                console.print(f"  Action: [dim](none)[/dim]")

            # Notes
            if action_data and action_data.get("notes"):
                console.print(f"  Notes: [italic]{action_data['notes']}[/italic]")

            console.print()

    # ── Top matches per pool ──────────────────────────────────────────────

    has_top = any(pool.get("top_matches") for pool in pools)
    if has_top:
        console.print("[bold dim]\u2500\u2500\u2500 Top Matches \u2500\u2500\u2500[/bold dim]")
        console.print()
        for pool in pools:
            matches = pool.get("top_matches", [])
            if not matches:
                continue
            display = matches[:top_k]
            names = [_basename(fp) for fp in display]
            console.print(f"  [bold]{pool['name']}[/bold]: {', '.join(names)}")
        console.print()


def print_json(conn, console: Console) -> None:
    """Output the complete result set as JSON."""

    pools = cache.get_pools(conn)
    temporals = cache.get_temporals(conn)
    actions = cache.get_actions(conn)
    bd = _build_breakdown(conn)
    insight = cache.get_state(conn, "opening_insight")

    temporal_by_pool = {t["pool_id"]: t for t in temporals}
    action_by_pool = {a["pool_id"]: a for a in actions}

    pool_records = []
    for pool in pools:
        pool_id = pool["id"]
        record: dict[str, Any] = {
            "id": pool_id,
            "name": pool["name"],
            "description": pool.get("description", ""),
            "intent": pool.get("intent", "unknown"),
            "match_count": pool.get("match_count", 0),
            "top_matches": pool.get("top_matches", []),
            "is_noise": pool.get("is_noise", False),
        }
        if pool_id in temporal_by_pool:
            record["temporal"] = temporal_by_pool[pool_id]
        if pool_id in action_by_pool:
            record["action"] = action_by_pool[pool_id]
        pool_records.append(record)

    output = {
        "pools": pool_records,
        "breakdown": {
            "siglip_high": bd.siglip_high,
            "ocr_confirmed": bd.ocr_confirmed,
            "gemini_resolved": bd.gemini_resolved,
            "claude_direct": bd.claude_direct,
            "unresolved": bd.unresolved,
            "duplicates_skipped": bd.duplicates_skipped,
            "total_unique": bd.total_unique,
        },
        "opening_insight": insight or "",
    }

    console.print_json(json.dumps(output))


def export_csv(conn, pool_name: str, console: Console) -> None:
    """Export classifications for a specific pool as CSV to stdout."""

    pools = cache.get_pools(conn)

    # Find matching pool (case-insensitive partial match)
    matched = None
    for pool in pools:
        if _pool_name_matches(pool["name"], pool_name):
            matched = pool
            break

    if matched is None:
        console.print(f"[red]No pool matching \"{pool_name}\" found.[/red]", stderr=True)
        available = [p["name"] for p in pools]
        if available:
            console.print(f"[dim]Available pools: {', '.join(available)}[/dim]", stderr=True)
        return

    pool_id = matched["id"]
    name_lower = matched["name"].lower()

    classifications = cache.get_classifications_by_pool(conn, pool_id)
    if not classifications:
        print(f"No classifications found for pool \"{matched['name']}\".", file=sys.stderr)
        return

    # Build image lookup for metadata
    all_images = cache.get_all_images(conn, include_duplicates=True)
    image_by_path: dict[str, dict] = {img["filepath"]: img for img in all_images}

    # Determine if extra OCR column is needed
    include_ocr = any(
        kw in name_lower
        for kw in ("music", "song", "playlist", "product", "shop", "store", "buy")
    )

    # Column definitions
    columns = ["filename", "filepath", "confidence", "method", "explanation", "created_at"]
    if include_ocr:
        columns.append("ocr_text")

    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()

    for cls in classifications:
        filepath = cls["filepath"]
        img = image_by_path.get(filepath, {})

        row = {
            "filename": _basename(filepath),
            "filepath": filepath,
            "confidence": cls.get("confidence", 0.0),
            "method": cls.get("method", ""),
            "explanation": cls.get("explanation", ""),
            "created_at": img.get("created_at", ""),
        }
        if include_ocr:
            row["ocr_text"] = cls.get("ocr_text", "")

        writer.writerow(row)

    print(f"{len(classifications)} rows exported for \"{matched['name']}\".", file=sys.stderr)


def run_validation(conn, console: Console) -> None:
    """Interactive validation: show 30 random classifications for user review."""

    rows = conn.execute("SELECT * FROM classifications").fetchall()
    all_cls = [dict(r) for r in rows]

    if not all_cls:
        console.print("[dim]No classifications to validate.[/dim]")
        return

    sample_size = min(30, len(all_cls))
    sample = random.sample(all_cls, sample_size)

    # Pool name lookup
    pools = cache.get_pools(conn)
    pool_names: dict[str, str] = {p["id"]: p["name"] for p in pools}

    correct = 0
    total = 0

    for i, cls in enumerate(sample, 1):
        filepath = cls["filepath"]
        pool_id = cls.get("pool_id", "")
        pool_name = pool_names.get(pool_id, pool_id)
        confidence = cls.get("confidence", 0.0)
        method = cls.get("method", "")

        panel_content = (
            f"[bold]File:[/bold] {_basename(filepath)}\n"
            f"[bold]Pool:[/bold] {pool_name}\n"
            f"[bold]Confidence:[/bold] {confidence:.2f}\n"
            f"[bold]Method:[/bold] {method}"
        )

        console.print(Panel(panel_content, title=f"Sample {i}/{sample_size}"))

        answer = Prompt.ask("Correct?", choices=["y", "n"], default="y")
        if answer.lower() == "y":
            correct += 1
        total += 1
        console.print()

    pct = (correct / total * 100) if total > 0 else 0
    console.print(f"[bold]{correct}/{total} correct ({pct:.0f}%)[/bold]")


def print_explanation(console: Console, result: dict) -> None:
    """Display why a specific image was classified into its pool."""

    filepath = result.get("filepath", "")
    pool_name = result.get("pool_name", "unknown")
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")

    console.print()
    console.print(
        f"This screenshot is [bold]{confidence:.2f}[/bold] similar to your "
        f"[bold]{pool_name}[/bold] pool [dim]\u2014[/dim] [italic]{explanation}[/italic]"
    )
    console.print()
