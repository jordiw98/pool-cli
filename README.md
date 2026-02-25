# Pool CLI

Intelligent screenshot organizer. Analyzes your screenshots and groups them into intent-aware Pools with temporal analysis and action suggestions.

## Install

```bash
cd pool-cli
pip install -e .
```

This installs the `pool` command globally. One-time model download (~350MB for SigLIP2) happens on first run.

### Requirements

- macOS (Apple Silicon recommended, uses Metal GPU)
- Python 3.10+
- API keys:

```bash
export ANTHROPIC_API_KEY=sk-...
export GOOGLE_API_KEY=...        # only needed for full mode
```

## Usage

```bash
# Full analysis (auto-detects dataset size)
pool ~/Pictures/Screenshots

# Quick mode — results in ~3 min
pool ~/Pictures/Screenshots --quick

# Estimate cost before running
pool ~/Pictures/Screenshots --cost

# Fully local, no API calls ($0)
pool ~/Pictures/Screenshots --no-api

# Export a pool as CSV
pool ~/Pictures/Screenshots --csv "Music"

# Explain why an image is in its pool
pool ~/Pictures/Screenshots --explain ~/Pictures/Screenshots/IMG_1234.png

# Validate accuracy on 30 random samples
pool ~/Pictures/Screenshots --validate

# JSON output
pool ~/Pictures/Screenshots --json
```

## How it works

### Three modes

| Mode | Trigger | Time | Cost |
|------|---------|------|------|
| **Full** | >= 200 images (default) | ~15 min | ~$6 |
| **Small** | < 200 images (auto) | ~3 min | ~$2 |
| **Quick** | `--quick` flag | ~3 min | ~$1 |

### Pipeline (full mode)

1. **Scan** — metadata, perceptual dedup, thumbnails
2. **Embed** — SigLIP2 512D vectors on Metal GPU
3. **Cluster** — UMAP + HDBSCAN discovers natural groups
4. **Discover** — Claude names pools, classifies intent
5. **Classify** — 3-tier cascade:
   - Tier 1: SigLIP zero-shot (instant, $0) — handles ~55%
   - Tier 2: Apple Vision OCR ($0) — handles ~24%
   - Tier 3: Gemini Flash tiebreaker (~$1) — handles ~10%
6. **Analyze** — temporal signatures, loop detection, Claude actions
7. **Output** — grouped by lifecycle status (Active / Comes & Goes / Moved On / One-Time)

### Caching

Everything is cached in SQLite (`.pool_cache.db` in the source directory). Re-running on the same folder completes in ~30 seconds. Use `--no-cache` to force a full re-analysis.

## CLI flags

| Flag | Description |
|------|-------------|
| `--quick` | Quick mode (~3 min) |
| `--json` | JSON output |
| `--csv <name>` | Export pool as CSV |
| `--explain <path>` | Explain one image's classification |
| `--validate` | Validate 30 random classifications |
| `--top-k <n>` | Top N matches per pool (default 10) |
| `--min-confidence <f>` | Confidence threshold (default 0.3) |
| `--cost` | Estimate cost without running |
| `--no-cache` | Force re-analysis |
| `--no-api` | Local only, $0 |
