# Pool CLI

**Intelligent screenshot organizer that treats your camera roll as intention archaeology.**

Most approaches treat screenshot organization as image classification — sort pictures into folders. Pool does something different. It treats screenshots as **frozen moments of caring** and detects the lifecycle of that caring: what you're actively into, what comes and goes in cycles, and what you looked at hard and moved on from.

It doesn't just categorize. It names your pools like a friend would ("Stuff You Almost Bought", not "Products Pool"), detects temporal patterns (active / abandoned / cyclical / one-shot), and suggests actions only when they'd genuinely help — some pools deliberately get no action, because that's the right call.

## Architecture

```
Phase 1: Scan + dedup                    ~2 min     $0
    Walk directory, metadata, pHash dedup, thumbnails

Phase 2: SigLIP2 embeddings              ~5 min     $0
    512D image vectors on Metal GPU (Apple Silicon)

Phase 3: UMAP + HDBSCAN clustering       ~1 min     $0
    Auto-discover natural groups, no predefined categories

Phase 4: Claude pool discovery            ~1 min     ~$2
    Cluster representatives → Claude → human-named pools

Phase 5: 3-tier classification cascade    ~4 min     ~$1
    Tier 1: SigLIP zero-shot (instant, $0)     ~55% of images
    Tier 2: Apple Vision OCR ($0)              ~24% of images
    Tier 3: Gemini 2.0 Flash tiebreaker        ~10% of images
    Unresolvable → catch-all pool              ~2% of images

Phase 6: Temporal analysis + actions      ~2 min     ~$3
    Loop detection, Claude-powered actions + opening insight

TOTAL (9K images, first run)              ~15 min    ~$6
TOTAL (cached re-run)                     <1 sec     $0
```

The key insight: no single model handles screenshots well. The text ON a screenshot is often a stronger signal than the pixels. So SigLIP handles the visually obvious ones, OCR catches the text-heavy ones, and Gemini Flash resolves the rest. ~85% is classified locally for $0.

## Setup

### What you need

- **macOS** with Apple Silicon (M1/M2/M3/M4) — uses Metal GPU for embeddings
- **Python 3.10+** (if you have a recent macOS + Homebrew, you probably have this)
- **Two API keys:**
  - `ANTHROPIC_API_KEY` — for Claude (pool naming, actions, narratives)
  - `GOOGLE_API_KEY` — for Gemini Flash (Tier 3 classification tiebreaker)

### Install

```bash
git clone https://github.com/jordiw98/pool-cli.git
cd pool-cli
```

**If you have Python 3.10+ already:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**If you're on an older Python (check with `python3 --version`):**

```bash
# Install uv — it handles Python versions for you
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv with Python 3.12 (downloads automatically if needed)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

### First run: model download

The first time you run `pool`, it downloads the **SigLIP2 vision model** (~350MB). This is a one-time download that gets cached in `~/.cache/huggingface/`. Subsequent runs load it from disk in a few seconds.

You might see a warning about unauthenticated HuggingFace requests — this is fine, the model is public. If you want to suppress it, set `HF_TOKEN` to any HuggingFace token.

### Set your API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

Or put them in a `.env` file in the project root (gitignored):

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' > .env
echo 'export GOOGLE_API_KEY="AIza..."' >> .env
source .env
```

## Usage

### Quick start

```bash
# Activate the venv
source .venv/bin/activate

# Run it on your screenshots folder
pool ~/Pictures/Screenshots
```

The tool scans your images, shows a cost estimate, and asks for confirmation before spending anything:

```
pool

Scanning screenshots… 39.7s
Found 9,247 screenshots (8,891 unique after dedup).
Mode: full

Estimated cost: ~$5.80 (Anthropic ~$4.60 · Google ~$1.20)
Proceed? [Y/n]
```

### Three modes

| Mode | Trigger | Time | Cost | Best for |
|------|---------|------|------|----------|
| **Full** | >= 200 images (auto) | ~15 min | ~$6 | The real deal. Full cascade + temporal analysis. |
| **Small** | < 200 images (auto) | ~3 min | ~$2 | Small collections. Skips HDBSCAN, sends all to Claude. |
| **Quick** | `--quick` flag | ~3 min | ~$1 | Fast preview. Samples 40 images, skips temporal analysis. |

**Recommended:** Run `--quick` first to see if the pools make sense, then do a full run.

### All flags

```bash
pool ~/path                          # Full run (auto-selects mode by image count)
pool ~/path --quick                  # Quick preview (~3 min, ~$1)
pool ~/path --cost                   # Show cost estimate without running
pool ~/path --no-api                 # Fully local mode, no API calls, $0
pool ~/path --json                   # Full structured JSON output
pool ~/path --csv "Music"            # Export one pool as CSV (partial name match)
pool ~/path --explain ~/path/img.png # Why is this image in this pool?
pool ~/path --validate               # Review 30 random classifications, report accuracy
pool ~/path --top-k 20               # Show 20 matches per pool (default: 10)
pool ~/path --min-confidence 0.5     # Stricter noise threshold (default: 0.3)
pool ~/path --no-cache               # Wipe cache, force full reprocessing
```

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

Stuff You Wanted to Buy — 45 matches
  Action: "Extract product names + best buy links"
  Notes: Your style changed a lot — only kept the 45 most recent

─── Comes & Goes ───

Hyrox Season — 82 matches
  Action: "Build a weekly plan from these workouts"
  Notes: Peaks every 3 months — looks like competition prep

─── Moved On ───

The Job Hunt — 31 matches
  Action: (none)
  Notes: Intense for 3 weeks in March, then you found what you needed

─── ¯\_(ツ)_/¯ ───

A Sunset, a Receipt, and Someone's Cat — 157 matches
  Action: (none)
  Notes: Screenshots that don't connect to anything — and that's fine
```

## Caching

Everything is cached in SQLite (`.pool_cache.db` inside the screenshots folder). This means:

- **Re-running** the same folder completes in under 1 second
- **Adding new screenshots** only processes the new ones
- **`--no-cache`** wipes the cache and reprocesses everything
- Cache is ~50MB for 9K images (metadata + embeddings + classifications)

## Tech stack

| Layer | Tool | Role |
|-------|------|------|
| Image embeddings | SigLIP2 (ViT-B-16-512) via `open_clip` | Cluster discovery + Tier 1 classification |
| Clustering | UMAP + HDBSCAN | Auto-discover natural groups |
| OCR | Apple Vision via `ocrmac` | Tier 2 — text extraction for text-heavy screenshots |
| Tiebreaker | Gemini 2.0 Flash | Tier 3 — low-confidence images |
| Discovery + narrative | Claude Sonnet 4 | Pool naming, intent, actions — the taste work |
| Dedup | `imagehash` (pHash) | Near-duplicate detection |
| Cache | SQLite | Everything cached, re-run = instant |
| CLI | Typer + Rich | Progress bars, styled terminal output |

## Cost safety

The tool never spends money without asking:

- Shows cost estimate before every run
- `--cost` for dry-run estimation
- `--no-api` for fully local $0 mode
- `--quick` caps at ~$1
- Full run on 9K images: ~$6 total
