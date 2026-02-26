"""Phase 3 — Pool Discovery.

Sends cluster representatives (full mode) or sampled images (small/quick mode)
to Claude for pool naming, intent classification, and SigLIP description generation.

Public API:
    discover_pools(conn, source_dir, mode, no_api)
    quick_discover(conn, source_dir, no_api)
"""

from __future__ import annotations

import base64
import json
import logging
import random
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pool import cache
from pool.models import (
    ClassificationMethod,
    IntentType,
    Pool,
    PoolAction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each attempt

# Representative selection per cluster (full mode)
CENTROID_REPS = 4   # 3-5 closest to centroid
EDGE_REPS = 2       # 2-3 from the edges

# Sampling caps
SMALL_MODE_SAMPLE = 50
QUICK_MODE_SAMPLE = 40

# Thumbnail max dimension for base64 payload (pixels)
THUMB_MAX_DIM = 512


# ---------------------------------------------------------------------------
# Generic / no-API fallback pools
# ---------------------------------------------------------------------------

_GENERIC_POOLS: list[dict[str, Any]] = [
    {
        "id": "music",
        "name": "Songs & Playlists",
        "description": "Music apps, songs, albums, and playlists you captured",
        "intent": IntentType.ASPIRATIONAL.value,
        "siglip_description": (
            "screenshot of a music streaming app showing a song album or playlist "
            "such as Spotify Apple Music YouTube Music or SoundCloud"
        ),
        "source_clusters": [],
        "is_noise": False,
    },
    {
        "id": "shopping",
        "name": "Stuff You Wanted to Buy",
        "description": "Products, prices, carts, and wishlists",
        "intent": IntentType.ASPIRATIONAL.value,
        "siglip_description": (
            "screenshot of an online store product page shopping cart or wishlist "
            "showing an item with a price on Amazon eBay or a retail website"
        ),
        "source_clusters": [],
        "is_noise": False,
    },
    {
        "id": "fitness",
        "name": "Workout & Health Tracking",
        "description": "Fitness stats, workout plans, health metrics",
        "intent": IntentType.FUNCTIONAL.value,
        "siglip_description": (
            "screenshot of a fitness or health tracking app showing workout stats "
            "steps calories heart rate or exercise routine"
        ),
        "source_clusters": [],
        "is_noise": False,
    },
    {
        "id": "recipes",
        "name": "Recipes You'll Totally Make",
        "description": "Recipes, food inspiration, and cooking instructions",
        "intent": IntentType.ASPIRATIONAL.value,
        "siglip_description": (
            "screenshot of a recipe showing ingredients cooking instructions or a "
            "food photo with preparation steps"
        ),
        "source_clusters": [],
        "is_noise": False,
    },
    {
        "id": "places",
        "name": "Places & Directions",
        "description": "Maps, addresses, travel plans, and saved locations",
        "intent": IntentType.FUNCTIONAL.value,
        "siglip_description": (
            "screenshot of a map navigation directions address or location "
            "from Google Maps Apple Maps or a travel app"
        ),
        "source_clusters": [],
        "is_noise": False,
    },
    {
        "id": "social",
        "name": "Social Media Moments",
        "description": "Posts, stories, DMs, and memes you saved from social apps",
        "intent": IntentType.SOCIAL.value,
        "siglip_description": (
            "screenshot of a social media app such as Instagram Twitter TikTok "
            "Snapchat or Facebook showing a post story message or meme"
        ),
        "source_clusters": [],
        "is_noise": False,
    },
]

_GENERIC_NOISE: dict[str, Any] = {
    "id": "noise",
    "name": "The Junk Drawer",
    "description": "Screenshots that don't clearly belong anywhere else",
    "intent": IntentType.UNKNOWN.value,
    "siglip_description": "miscellaneous screenshot that does not fit any specific category",
    "source_clusters": [],
    "is_noise": True,
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _image_to_base64(filepath: str, max_dim: int = THUMB_MAX_DIM) -> tuple[str, str]:
    """Read an image, resize if needed, and return (base64_data, media_type).

    Uses PIL for resizing so that we don't send huge payloads to Claude.
    """
    from PIL import Image
    import io

    path = Path(filepath)
    suffix = path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_map.get(suffix, "image/png")

    img = Image.open(path)
    # Convert RGBA to RGB for JPEG output
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize keeping aspect ratio
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    # Always output JPEG for smaller payload
    img.save(buf, format="JPEG", quality=80)
    media_type = "image/jpeg"
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return b64, media_type


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-ish date string, returning None on failure."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _date_range_str(dates: list[Optional[datetime]]) -> str:
    """Compact date range like 'Jan 2024 – Mar 2024'."""
    valid = sorted(d for d in dates if d is not None)
    if not valid:
        return "unknown dates"
    if len(valid) == 1:
        return valid[0].strftime("%b %Y")
    return f"{valid[0].strftime('%b %Y')} – {valid[-1].strftime('%b %Y')}"


def _sample_spread(items: list[dict], n: int) -> list[dict]:
    """Sample *n* items spread across the date range.

    Items are expected to have a 'created_at' or 'modified_at' key.
    Falls back to random sampling if dates are unavailable.
    """
    if len(items) <= n:
        return list(items)

    def _sort_key(img: dict) -> str:
        return img.get("created_at") or img.get("modified_at") or ""

    sorted_items = sorted(items, key=_sort_key)
    # Pick evenly spaced indices
    indices = np.linspace(0, len(sorted_items) - 1, n, dtype=int)
    seen: set[int] = set()
    result: list[dict] = []
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            result.append(sorted_items[idx])
    # If rounding collapsed some indices, pad with random picks
    remaining = [sorted_items[i] for i in range(len(sorted_items)) if i not in seen]
    while len(result) < n and remaining:
        pick = remaining.pop(random.randrange(len(remaining)))
        result.append(pick)
    return result


# ---------------------------------------------------------------------------
# Claude API call with retry
# ---------------------------------------------------------------------------

def _call_claude(system: str, user_content: list[dict], *, max_tokens: int = 4096) -> str:
    """Send a request to Claude and return the text response.

    Retries up to MAX_RETRIES times with exponential backoff on transient errors.
    """
    import anthropic

    client = anthropic.Anthropic()

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            # Extract text from the first text block
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            raise ValueError("Claude response contained no text blocks")
        except Exception as exc:
            last_error = exc
            # Don't retry on auth or invalid request errors
            exc_name = type(exc).__name__
            if "AuthenticationError" in exc_name or "InvalidRequestError" in exc_name:
                raise
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Claude API call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

    raise RuntimeError(f"Claude API call failed after {MAX_RETRIES} attempts") from last_error


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from Claude's response.

    Claude sometimes wraps JSON in markdown code fences — strip those.
    """
    cleaned = text.strip()
    # Remove ```json ... ``` wrapper
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else 3
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are helping build an intelligent screenshot organizer called Pool.
Your job is to look at clusters of screenshots from one person's phone and \
transform raw groupings into pools that feel human, perceptive, and specific to this person.

Pool names should sound like a close friend describing your phone back to you — \
not a database schema. "Stuff You Wanted to Buy" not "Products Pool."

For each pool, classify the intent:
- Aspirational ("I want this")
- Functional ("I need this later")
- Emotional ("I want to remember this")
- Social ("I want to share this")
- Investigative ("I'm researching")
- Creative ("I'm stealing/referencing")"""


def _build_full_mode_content(
    clusters: dict[int, list[dict]],
    representative_map: dict[int, list[str]],
    all_images: list[dict],
) -> list[dict]:
    """Build the user content blocks for full-mode discovery."""
    # Compute global stats
    dates = [_parse_date(img.get("created_at") or img.get("modified_at")) for img in all_images]
    valid_dates = [d for d in dates if d]
    unique_count = len(all_images)
    earliest = min(valid_dates).strftime("%b %d, %Y") if valid_dates else "unknown"
    latest = max(valid_dates).strftime("%b %d, %Y") if valid_dates else "unknown"
    n_clusters = len(clusters)

    preamble = (
        f"I've analyzed {unique_count} screenshots from one person's phone spanning "
        f"{earliest} to {latest}. My clustering found {n_clusters} natural groups.\n\n"
        "Here's each group with representative screenshots:"
    )
    content: list[dict] = [{"type": "text", "text": preamble}]

    for cluster_id in sorted(clusters.keys()):
        imgs = clusters[cluster_id]
        cluster_dates = [
            _parse_date(im.get("created_at") or im.get("modified_at")) for im in imgs
        ]
        dr = _date_range_str(cluster_dates)
        label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        content.append({
            "type": "text",
            "text": f"\n{label} ({len(imgs)} images, {dr}):",
        })

        for fp in representative_map.get(cluster_id, []):
            try:
                b64, mtype = _image_to_base64(fp)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mtype, "data": b64},
                })
            except Exception as exc:
                logger.warning("Could not encode image %s: %s", fp, exc)

    content.append({
        "type": "text",
        "text": (
            "\n\nYour job:\n"
            "1. Name each pool (human-friendly, intent-aware)\n"
            "2. Classify intent type\n"
            "3. Write a one-sentence description\n"
            "4. Write a SigLIP-optimized text description for zero-shot classification "
            "(e.g., \"screenshot of a music streaming app showing a song, album, or playlist\")\n"
            "5. Note if any clusters should merge or split\n"
            "6. Name the noise cluster something charming\n\n"
            "Return JSON:\n"
            "{\n"
            '  "pools": [\n'
            "    {\n"
            '      "id": "music",\n'
            '      "name": "Your Shazam Moments",\n'
            '      "description": "Songs you heard and wanted to remember",\n'
            '      "intent": "aspirational",\n'
            '      "siglip_description": "screenshot of a music app showing a song album or playlist",\n'
            '      "source_clusters": [1, 4],\n'
            '      "notes": "Clusters 1 and 4 are both music — merge them"\n'
            "    }\n"
            "  ],\n"
            '  "noise_pool": {\n'
            '    "name": "Not Sure Why You Saved These",\n'
            '    "description": "screenshots that don\'t connect to anything"\n'
            "  }\n"
            "}\n\n"
            "Return ONLY valid JSON, no extra commentary."
        ),
    })
    return content


def _build_small_mode_content(
    sampled_images: list[dict],
    all_images: list[dict],
) -> list[dict]:
    """Build user content for small-mode discovery (no clustering)."""
    dates = [_parse_date(img.get("created_at") or img.get("modified_at")) for img in all_images]
    valid_dates = [d for d in dates if d]
    unique_count = len(all_images)
    earliest = min(valid_dates).strftime("%b %d, %Y") if valid_dates else "unknown"
    latest = max(valid_dates).strftime("%b %d, %Y") if valid_dates else "unknown"

    preamble = (
        f"I have {unique_count} screenshots from one person's phone spanning "
        f"{earliest} to {latest}. No clustering was performed — instead here are "
        f"{len(sampled_images)} representative samples spread across the date range.\n\n"
        "Look at all of them, then define the pools."
    )
    content: list[dict] = [{"type": "text", "text": preamble}]

    for img in sampled_images:
        try:
            b64, mtype = _image_to_base64(img["filepath"])
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mtype, "data": b64},
            })
        except Exception as exc:
            logger.warning("Could not encode image %s: %s", img["filepath"], exc)

    content.append({
        "type": "text",
        "text": (
            "\n\nYour job:\n"
            "1. Define pools (human-friendly names, intent-aware)\n"
            "2. Classify intent type for each pool\n"
            "3. Write a one-sentence description for each\n"
            "4. Write a SigLIP-optimized text description for zero-shot classification\n"
            "5. Include a noise/catch-all pool\n\n"
            "Return JSON:\n"
            "{\n"
            '  "pools": [\n'
            "    {\n"
            '      "id": "short_slug",\n'
            '      "name": "Human-Friendly Name",\n'
            '      "description": "One sentence",\n'
            '      "intent": "aspirational|functional|emotional|social|investigative|creative",\n'
            '      "siglip_description": "screenshot of ...",\n'
            '      "source_clusters": []\n'
            "    }\n"
            "  ],\n"
            '  "noise_pool": {\n'
            '    "name": "Charming Noise Name",\n'
            '    "description": "catch-all description"\n'
            "  }\n"
            "}\n\n"
            "Return ONLY valid JSON, no extra commentary."
        ),
    })
    return content


def _build_quick_mode_content(
    sampled_images: list[dict],
    all_images: list[dict],
) -> list[dict]:
    """Build user content for quick-mode discovery + inline classification."""
    dates = [_parse_date(img.get("created_at") or img.get("modified_at")) for img in all_images]
    valid_dates = [d for d in dates if d]
    unique_count = len(all_images)
    earliest = min(valid_dates).strftime("%b %d, %Y") if valid_dates else "unknown"
    latest = max(valid_dates).strftime("%b %d, %Y") if valid_dates else "unknown"

    # We label each image so Claude can reference them in classifications
    preamble = (
        f"I have {unique_count} screenshots from one person's phone spanning "
        f"{earliest} to {latest}. Here are {len(sampled_images)} samples. "
        "I need you to:\n"
        "1. Define pools\n"
        "2. Classify each of these sample images into a pool\n"
        "3. Suggest a short action/recommendation per pool\n"
    )
    content: list[dict] = [{"type": "text", "text": preamble}]

    for idx, img in enumerate(sampled_images):
        content.append({
            "type": "text",
            "text": f"\nImage {idx + 1} ({Path(img['filepath']).name}):",
        })
        try:
            b64, mtype = _image_to_base64(img["filepath"])
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mtype, "data": b64},
            })
        except Exception as exc:
            logger.warning("Could not encode image %s: %s", img["filepath"], exc)

    content.append({
        "type": "text",
        "text": (
            "\n\nReturn JSON:\n"
            "{\n"
            '  "pools": [\n'
            "    {\n"
            '      "id": "short_slug",\n'
            '      "name": "Human-Friendly Name",\n'
            '      "description": "One sentence",\n'
            '      "intent": "aspirational|functional|emotional|social|investigative|creative",\n'
            '      "siglip_description": "screenshot of ...",\n'
            '      "action": "What to do with this pool (or null)",\n'
            '      "action_why": "Why this action makes sense"\n'
            "    }\n"
            "  ],\n"
            '  "noise_pool": {\n'
            '    "name": "Charming Noise Name",\n'
            '    "description": "catch-all"\n'
            "  },\n"
            '  "classifications": [\n'
            '    {"image_index": 1, "pool_id": "short_slug", "confidence": 0.9}\n'
            "  ]\n"
            "}\n\n"
            "Return ONLY valid JSON, no extra commentary."
        ),
    })
    return content


# ---------------------------------------------------------------------------
# Representative selection (full mode)
# ---------------------------------------------------------------------------

def _select_representatives(
    cluster_images: list[dict],
    cluster_embeddings: dict[str, np.ndarray],
) -> list[str]:
    """Pick 5-7 representative filepaths: centroid-near + edge images.

    Returns fewer if the cluster is small.
    """
    fps = [img["filepath"] for img in cluster_images if img["filepath"] in cluster_embeddings]
    if not fps:
        # No embeddings available — fall back to random sample
        sample_n = min(CENTROID_REPS + EDGE_REPS, len(cluster_images))
        return [img["filepath"] for img in random.sample(cluster_images, sample_n)]

    vecs = np.array([cluster_embeddings[fp] for fp in fps])
    centroid = vecs.mean(axis=0)

    # Cosine distances to centroid
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    cos_sim = (vecs @ centroid) / (norms.squeeze() * np.linalg.norm(centroid) + 1e-9)
    order = np.argsort(-cos_sim)  # descending similarity

    n_centroid = min(CENTROID_REPS, len(fps))
    n_edge = min(EDGE_REPS, max(0, len(fps) - n_centroid))

    selected_indices: list[int] = []
    # Closest to centroid
    selected_indices.extend(order[:n_centroid].tolist())
    # Farthest from centroid (edges)
    if n_edge > 0:
        edge_candidates = order[n_centroid:]
        selected_indices.extend(edge_candidates[-n_edge:].tolist())

    return [fps[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _normalize_intent(raw: str) -> str:
    """Map a raw intent string to a valid IntentType value."""
    mapping = {v.value: v.value for v in IntentType}
    lowered = raw.strip().lower()
    if lowered in mapping:
        return lowered
    return IntentType.UNKNOWN.value


def _pools_from_response(data: dict) -> list[dict]:
    """Convert Claude's JSON response into a list of pool dicts for save_pools."""
    pools: list[dict] = []
    for p in data.get("pools", []):
        pools.append({
            "id": p.get("id", "unknown"),
            "name": p.get("name", "Unnamed Pool"),
            "description": p.get("description", ""),
            "intent": _normalize_intent(p.get("intent", "unknown")),
            "siglip_description": p.get("siglip_description", ""),
            "source_clusters": p.get("source_clusters", []),
            "match_count": 0,
            "top_matches": [],
            "is_noise": False,
        })

    # Noise pool
    noise = data.get("noise_pool", {})
    pools.append({
        "id": "noise",
        "name": noise.get("name", "The Junk Drawer"),
        "description": noise.get("description", "Screenshots that don't fit anywhere"),
        "intent": IntentType.UNKNOWN.value,
        "siglip_description": "miscellaneous screenshot that does not fit any specific category",
        "source_clusters": [],
        "match_count": 0,
        "top_matches": [],
        "is_noise": True,
    })
    return pools


# ---------------------------------------------------------------------------
# Public API — discover_pools
# ---------------------------------------------------------------------------

def discover_pools(
    conn: sqlite3.Connection,
    source_dir: str,
    mode: str = "full",
    no_api: bool = False,
) -> list[dict]:
    """Phase 3: Discover pools from clustered or raw images.

    Args:
        conn: SQLite connection (with pool cache tables).
        source_dir: Root directory of screenshots.
        mode: "full" (clustered) or "small" (unclustered).
        no_api: If True, generate generic pools without calling Claude.

    Returns:
        List of pool dicts that were saved to cache.
    """
    # Early return if pools already cached
    existing = cache.get_pools(conn)
    if existing:
        logger.info("Pools already cached (%d pools) — skipping discovery.", len(existing))
        return existing

    if no_api:
        return _discover_no_api(conn)

    all_images = cache.get_all_images(conn, include_duplicates=False)
    if not all_images:
        logger.warning("No images found in cache — cannot discover pools.")
        return []

    if mode == "full":
        pools = _discover_full(conn, all_images)
    else:
        pools = _discover_small(conn, all_images)

    cache.save_pools(conn, pools)
    conn.commit()
    return pools


def _discover_no_api(conn: sqlite3.Connection) -> list[dict]:
    """Generate generic fallback pools without any API call."""
    pools = [dict(p) for p in _GENERIC_POOLS]
    pools.append(dict(_GENERIC_NOISE))
    cache.save_pools(conn, pools)
    conn.commit()
    return pools


def _discover_full(conn: sqlite3.Connection, all_images: list[dict]) -> list[dict]:
    """Full-mode discovery: cluster representatives sent to Claude."""
    # Load embeddings and cluster assignments
    raw_embeddings = cache.get_all_embeddings(conn)
    embedding_map: dict[str, np.ndarray] = {fp: vec for fp, vec in raw_embeddings}

    # Get cluster_id per filepath from the embeddings table
    rows = conn.execute(
        "SELECT filepath, cluster_id FROM embeddings WHERE cluster_id IS NOT NULL"
    ).fetchall()
    fp_to_cluster: dict[str, int] = {r[0]: r[1] for r in rows}

    # Build image lookup
    img_lookup: dict[str, dict] = {img["filepath"]: img for img in all_images}

    # Group images by cluster
    clusters: dict[int, list[dict]] = defaultdict(list)
    for fp, cid in fp_to_cluster.items():
        if fp in img_lookup:
            clusters[cid].append(img_lookup[fp])

    # Images without a cluster assignment go to noise (-1)
    assigned_fps = set(fp_to_cluster.keys())
    for img in all_images:
        if img["filepath"] not in assigned_fps:
            clusters[-1].append(img)

    # Select representatives per cluster
    representative_map: dict[int, list[str]] = {}
    for cid, imgs in clusters.items():
        cluster_embs = {
            fp: embedding_map[fp] for fp in [im["filepath"] for im in imgs] if fp in embedding_map
        }
        representative_map[cid] = _select_representatives(imgs, cluster_embs)

    # Build prompt and call Claude
    user_content = _build_full_mode_content(clusters, representative_map, all_images)
    raw_response = _call_claude(_SYSTEM_PROMPT, user_content, max_tokens=4096)

    try:
        data = _parse_json_response(raw_response)
    except json.JSONDecodeError:
        logger.error("Failed to parse Claude discovery response — falling back to generic pools.")
        return _GENERIC_POOLS + [_GENERIC_NOISE]

    return _pools_from_response(data)


def _discover_small(conn: sqlite3.Connection, all_images: list[dict]) -> list[dict]:
    """Small-mode discovery: sample images sent to Claude (no clustering)."""
    sampled = _sample_spread(all_images, SMALL_MODE_SAMPLE)

    user_content = _build_small_mode_content(sampled, all_images)
    raw_response = _call_claude(_SYSTEM_PROMPT, user_content, max_tokens=4096)

    try:
        data = _parse_json_response(raw_response)
    except json.JSONDecodeError:
        logger.error("Failed to parse Claude discovery response — falling back to generic pools.")
        return _GENERIC_POOLS + [_GENERIC_NOISE]

    return _pools_from_response(data)


# ---------------------------------------------------------------------------
# Public API — quick_discover
# ---------------------------------------------------------------------------

def quick_discover(
    conn: sqlite3.Connection,
    source_dir: str,
    no_api: bool = False,
) -> list[dict]:
    """Quick-mode shortcut: sample images, discover pools, classify samples, generate actions.

    This replaces Phases 3-5 in a single Claude call.

    Args:
        conn: SQLite connection.
        source_dir: Root directory of screenshots.
        no_api: If True, use generic pools and skip Claude.

    Returns:
        List of pool dicts that were saved.
    """
    # Early return if pools already cached
    existing = cache.get_pools(conn)
    if existing:
        logger.info("Pools already cached (%d pools) — skipping quick discovery.", len(existing))
        return existing

    all_images = cache.get_all_images(conn, include_duplicates=False)
    if not all_images:
        logger.warning("No images found in cache — cannot discover pools.")
        return []

    if no_api:
        pools = _discover_no_api(conn)
        _generate_generic_actions(conn, pools)
        return pools

    sampled = _sample_spread(all_images, QUICK_MODE_SAMPLE)

    # Build prompt and call Claude
    user_content = _build_quick_mode_content(sampled, all_images)
    raw_response = _call_claude(_SYSTEM_PROMPT, user_content, max_tokens=8192)

    try:
        data = _parse_json_response(raw_response)
    except json.JSONDecodeError:
        logger.error("Failed to parse Claude quick-discover response — falling back to generics.")
        pools = _GENERIC_POOLS + [_GENERIC_NOISE]
        cache.save_pools(conn, pools)
        conn.commit()
        _generate_generic_actions(conn, pools)
        return pools

    # Save pools
    pools = _pools_from_response(data)
    cache.save_pools(conn, pools)

    # Save inline classifications for the sampled images
    classifications = data.get("classifications", [])
    for c in classifications:
        img_idx = c.get("image_index")
        pool_id = c.get("pool_id")
        confidence = c.get("confidence", 0.8)
        if img_idx is None or pool_id is None:
            continue
        # image_index is 1-based
        idx = img_idx - 1
        if 0 <= idx < len(sampled):
            fp = sampled[idx]["filepath"]
            cache.upsert_classification(
                conn,
                filepath=fp,
                pool_id=pool_id,
                confidence=float(confidence),
                method=ClassificationMethod.CLAUDE_DIRECT.value,
                explanation="Classified by Claude in quick mode",
                ocr_text=None,
            )

    # Save inline actions (quick mode generates them here since Phase 5 is skipped)
    _save_quick_actions(conn, data, pools)

    conn.commit()
    return pools


def _save_quick_actions(
    conn: sqlite3.Connection,
    data: dict,
    pools: list[dict],
) -> None:
    """Extract and save pool actions from Claude's quick-mode response."""
    pool_actions: dict[str, dict] = {}
    for p in data.get("pools", []):
        pid = p.get("id")
        if pid:
            pool_actions[pid] = {
                "action": p.get("action"),
                "action_why": p.get("action_why", ""),
            }

    for pool in pools:
        pid = pool["id"]
        action_info = pool_actions.get(pid, {})
        action_text = action_info.get("action")
        cache.save_action(
            conn,
            pool_id=pid,
            action=action_text,
            why=action_info.get("action_why", ""),
            notes="Generated in quick mode",
            has_action=1 if action_text else 0,
        )


def _generate_generic_actions(conn: sqlite3.Connection, pools: list[dict]) -> None:
    """Generate placeholder actions for no-api mode."""
    for pool in pools:
        cache.save_action(
            conn,
            pool_id=pool["id"],
            action=None,
            why=None,
            notes="No API mode — actions require Claude analysis",
            has_action=0,
        )
    conn.commit()
