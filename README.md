# Pool CLI

Intelligent screenshot organizer. Analyzes your screenshots and groups them into intent-aware Pools with temporal analysis (active/abandoned/cyclical loops) and action suggestions.

Most tools treat this as image classification. Pool treats it as **intention archaeology** — screenshots are frozen moments of caring, and Pool detects the lifecycle of that caring.

## Setup (macOS, Apple Silicon)

```bash
# 1. Clone
git clone https://github.com/jordiw98/pool-cli.git
cd pool-cli

# 2. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create venv + install dependencies
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .

# 4. Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AIza...
```

First run downloads the SigLIP2 model (~350MB one-time).

## Run it

```bash
# Point it at your screenshots folder
pool ~/Pictures/Screenshots
```

The tool scans your images, shows a cost estimate, and asks for confirmation before any API call:

```
Found 9,247 screenshots (8,891 unique after dedup).
Estimated cost: ~$5.80 (Anthropic ~$4.60 · Google ~$1.20)
Proceed? [Y/n]
```

## Modes

| Mode | When | Time (9K images) | Cost |
|------|------|-------------------|------|
| **Full** | >= 200 images (auto) | ~15 min | ~$6 |
| **Small** | < 200 images (auto) | ~3 min | ~$2 |
| **Quick** | `--quick` flag | ~3 min | ~$1 |

Quick mode samples 40 images for a fast preview. Full mode runs the complete pipeline with temporal analysis and the 3-tier classification cascade.

## What the full pipeline does

1. **Scan** — walk directory, extract metadata, perceptual dedup (pHash), generate thumbnails
2. **Embed** — SigLIP2 512D image embeddings on Metal GPU
3. **Cluster** — UMAP dimensionality reduction + HDBSCAN auto-clustering
4. **Discover** — Claude sees cluster representatives, names pools with human-friendly names, classifies intent (aspirational/functional/emotional/social/investigative/creative)
5. **Classify** — 3-tier cascade:
   - Tier 1: SigLIP zero-shot (instant, $0) — typically handles ~55%
   - Tier 2: Apple Vision OCR for text-heavy screenshots ($0) — handles ~24%
   - Tier 3: Gemini 2.0 Flash tiebreaker (~$1) — handles ~10%
   - Unresolvable images go to a charming catch-all pool
6. **Analyze** — compute temporal signatures per pool, detect open loops (active/abandoned/cyclical/one-shot), generate Claude-powered action suggestions with restraint (some pools deliberately get no action)
7. **Output** — pools grouped by lifecycle status, classification breakdown, opening insight

## Example output

```
Classification
  SigLIP zero-shot (high confidence):  4,892  (55%)
  SigLIP + OCR confirmed:             2,104  (24%)
  Gemini Flash tiebreaker:               847  (10%)
  Unresolved → "Not Sure" pool:          157  ( 2%)
  Duplicates skipped:                    356

"You care about music the way some people care about food — in bursts, deeply, then you move on."

─── Active ───

Your Shazam Moments — 234 matches
  Action: "Build a playlist from your last 6 months — 38 tracks identified"
  Notes: Most active Nov-Jan, slowed down but still going

─── Comes & Goes ───

Hyrox Season — 82 matches
  Action: "Build a weekly plan from these workouts"
  Notes: Peaks every 3 months — looks like competition prep

─── Moved On ───

Stuff You Almost Bought — 45 matches
  Action: (none)
  Notes: You browsed hard in September and moved on completely
```

## All CLI flags

```bash
pool ~/path                          # Full run (auto-selects mode)
pool ~/path --quick                  # Quick preview (~3 min, ~$1)
pool ~/path --cost                   # Just show cost estimate
pool ~/path --no-api                 # Fully local, no API, $0
pool ~/path --json                   # JSON output
pool ~/path --csv "Music"            # Export one pool as CSV
pool ~/path --explain ~/path/img.png # Why is this image in this pool?
pool ~/path --validate               # Review 30 random classifications
pool ~/path --top-k 20               # Show 20 matches per pool
pool ~/path --min-confidence 0.5     # Stricter noise threshold
pool ~/path --no-cache               # Wipe cache, full reprocess
```

## Caching

Everything is cached in SQLite (`.pool_cache.db` in the source directory). Re-running the same folder completes in under 1 second. Use `--no-cache` to force a full re-analysis.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) — uses Metal GPU for embeddings
- Python 3.10+ (3.12 recommended)
- [uv](https://github.com/astral-sh/uv) for dependency management
- `ANTHROPIC_API_KEY` — for Claude (pool discovery + actions)
- `GOOGLE_API_KEY` — for Gemini Flash (Tier 3 tiebreaker, full mode only)

## Cost safety

- Always shows an estimate and asks for confirmation before API calls
- `--cost` flag for dry-run estimation
- `--no-api` for $0 local-only mode
- `--quick` caps at ~$1
- Full run on 9K images: ~$6
