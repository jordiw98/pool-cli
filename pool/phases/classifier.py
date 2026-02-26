"""Phase 4: 3-tier cascading classification.

Tier 1 -- SigLIP zero-shot (instant, $0)
Tier 2 -- OCR enrichment via Apple Vision / ocrmac ($0)
Tier 3 -- Gemini 2.0 Flash tiebreaker (~$0.001/image)

Also provides the --explain feature for single-image explanations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import threading
import time as time_module

import base64 as base64_module

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from pool import cache
from pool.models import ClassificationMethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter for external API calls
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, max_per_minute: int = 14):
        self._interval = 60.0 / max_per_minute
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        """Block until a request is allowed."""
        with self._lock:
            now = time_module.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time_module.sleep(wait)
            self._last = time_module.monotonic()

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE = 0.70
_MEDIUM_CONFIDENCE = 0.40

# ---------------------------------------------------------------------------
# Keyword banks for OCR Tier-2 scoring
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "music": [
        "spotify", "apple music", "shazam", "album", "playlist",
        "song", "track", "artist", "now playing", "lyrics",
    ],
    "shopping": [
        "\u20ac", "$", "add to cart", "buy now", "price", "shipping",
        "order", "checkout", "wishlist", "in stock",
    ],
    "fitness": [
        "workout", "reps", "sets", "heart rate", "km", "pace",
        "hyrox", "training", "calories", "steps", "strava",
    ],
    "recipes": [
        "ingredients", "prep time", "servings", "tbsp", "cups",
        "recipe", "cook", "bake", "minutes", "preheat",
    ],
    "hiring": [
        "linkedin", "experience", "resume", "apply", "candidate",
        "job", "position", "recruiter", "hiring", "cv",
    ],
    "travel": [
        "booking", "flight", "hotel", "check-in", "departure",
        "boarding", "itinerary", "airbnb", "reservation", "gate",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_noise_pool(pools: list[dict]) -> Optional[str]:
    """Return the pool_id of the noise / catch-all pool, if one exists."""
    for p in pools:
        if p.get("is_noise"):
            return p["id"]
    return None


def _build_keyword_map(pools: list[dict]) -> dict[str, list[str]]:
    """Build a mapping of pool_id -> list of keywords for OCR matching.

    Combines the static category keyword banks with dynamic keywords derived
    from each pool's name, description, and siglip_description.
    """
    pool_keywords: dict[str, list[str]] = {}

    # Index category keywords by lowercase for matching against pool text
    category_index: dict[str, list[str]] = {}
    for cat, kws in _CATEGORY_KEYWORDS.items():
        category_index[cat] = kws

    for pool in pools:
        if pool.get("is_noise"):
            continue

        pid = pool["id"]
        keywords: list[str] = []

        # Dynamic keywords from pool name and description
        name_lower = pool.get("name", "").lower()
        desc_lower = pool.get("description", "").lower()
        siglip_lower = (pool.get("siglip_description") or "").lower()
        combined_text = f"{name_lower} {desc_lower} {siglip_lower}"

        # Add meaningful words from pool name (>3 chars, skip stop words)
        _stop = {"the", "and", "for", "with", "from", "this", "that", "your", "pool", "screenshots"}
        for word in re.findall(r"[a-z]{4,}", name_lower):
            if word not in _stop:
                keywords.append(word)

        # Match against category keyword banks
        for cat, kws in category_index.items():
            if cat in combined_text or any(kw in combined_text for kw in kws[:3]):
                keywords.extend(kws)

        # Add words from siglip_description as keywords
        for word in re.findall(r"[a-z]{4,}", siglip_lower):
            if word not in _stop:
                keywords.append(word)

        pool_keywords[pid] = list(set(keywords))

    return pool_keywords


def _score_ocr_for_pools(
    ocr_text: str,
    pool_keywords: dict[str, list[str]],
) -> list[tuple[str, int, set[str]]]:
    """Score OCR text against each pool's keyword bank.

    Returns a sorted list of (pool_id, match_count, matched_keywords)
    in descending order of match_count.
    """
    text_lower = ocr_text.lower()
    scores: list[tuple[str, int, set[str]]] = []

    for pid, keywords in pool_keywords.items():
        matched: set[str] = set()
        for kw in keywords:
            if kw in text_lower:
                matched.add(kw)
        if matched:
            scores.append((pid, len(matched), matched))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _run_ocr(filepath: str) -> str:
    """Run Apple Vision OCR on a single image via ocrmac.

    Returns the extracted text, or an empty string on failure.
    """
    try:
        from ocrmac import ocrmac as ocrmac_mod

        annotations = ocrmac_mod.OCR(filepath, recognition_level="fast").recognize()
        extracted = " ".join([item[0] for item in annotations])
        return extracted.strip()
    except Exception:
        logger.debug("OCR failed for %s", filepath, exc_info=True)
        return ""


def _image_to_base64_for_gemini(filepath: str, max_dim: int = 512) -> tuple[str, str]:
    """Read an image, resize to thumbnail, return (base64_data, mime_type).

    Uses inline base64 instead of upload_file to avoid sending full-size
    images to Google's servers and to reduce upload time.
    """
    from PIL import Image
    import base64
    import io

    img = Image.open(filepath)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return b64, "image/jpeg"


def _call_gemini_single(
    filepath: str,
    pools_info: list[dict],
    model,
) -> dict:
    """Send a single image to Gemini Flash and parse the classification response.

    Returns a dict with keys: pool_id, confidence, explanation.
    On failure returns an empty dict.
    """
    pool_list_text = "\n".join(
        f'- id="{p["id"]}", name="{p["name"]}", description="{p["description"]}"'
        for p in pools_info
        if not p.get("is_noise")
    )

    prompt = (
        "You are classifying a screenshot into one of these content pools:\n\n"
        f"{pool_list_text}\n\n"
        "Look at this screenshot and determine which pool it belongs to.\n"
        "Respond ONLY with valid JSON (no markdown, no backticks):\n"
        '{"pool_id": "<id>", "confidence": <0.0-1.0>, "explanation": "<one sentence>"}\n'
        "If the screenshot does not clearly fit any pool, use confidence < 0.3."
    )

    try:
        img_path = Path(filepath)
        if not img_path.exists():
            return {}

        b64_data, mime_type = _image_to_base64_for_gemini(str(img_path))
        response = model.generate_content(
            [
                prompt,
                {"mime_type": mime_type, "data": base64_module.b64decode(b64_data)},
            ],
            generation_config={"temperature": 0.1, "max_output_tokens": 256},
        )

        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)
        return {
            "pool_id": parsed.get("pool_id", ""),
            "confidence": float(parsed.get("confidence", 0.0)),
            "explanation": parsed.get("explanation", ""),
        }
    except json.JSONDecodeError:
        logger.debug("Gemini returned invalid JSON for %s: %s", filepath, text if "text" in dir() else "N/A")
        return {}
    except Exception:
        logger.debug("Gemini call failed for %s", filepath, exc_info=True)
        return {}


_gemini_limiter = _RateLimiter(max_per_minute=14)  # Google free tier: 15 rpm, leave headroom


def _call_gemini_with_retry(
    filepath: str,
    pools_info: list[dict],
    model,
    retries: int = 2,
) -> dict:
    """Call Gemini with rate limiting, retries, and exponential backoff."""
    for attempt in range(1 + retries):
        _gemini_limiter.acquire()
        result = _call_gemini_single(filepath, pools_info, model)
        if result:
            return result
        if attempt < retries:
            wait = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
            logger.debug("Retrying Gemini for %s (attempt %d, wait %.1fs)", filepath, attempt + 2, wait)
            time_module.sleep(wait)
    return {}


def _make_progress() -> Progress:
    """Create a standard Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )


# ---------------------------------------------------------------------------
# Centroid-based classification (uses cluster embeddings directly)
# ---------------------------------------------------------------------------


def _compute_pool_centroids(
    conn: sqlite3.Connection,
    pools: list[dict],
    all_embeddings: list[tuple[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Compute L2-normalized centroid embeddings for each pool.

    Uses the cluster→pool mapping from discovery to aggregate embeddings
    of all images assigned to each pool's source clusters. Returns an empty
    dict when cluster data is unavailable (e.g. small mode).
    """
    # Get cluster assignments from embeddings table
    rows = conn.execute(
        "SELECT filepath, cluster_id FROM embeddings WHERE cluster_id IS NOT NULL"
    ).fetchall()
    if not rows:
        return {}

    fp_to_cluster: dict[str, int] = {r[0]: r[1] for r in rows}
    emb_lookup: dict[str, np.ndarray] = {fp: emb for fp, emb in all_embeddings}

    # Map cluster IDs to pool IDs via source_clusters
    centroids: dict[str, np.ndarray] = {}
    for pool in pools:
        if pool.get("is_noise"):
            continue
        source_clusters = pool.get("source_clusters", [])
        if not source_clusters:
            continue

        source_set = set(int(c) for c in source_clusters)

        # Collect all embeddings belonging to this pool's clusters
        pool_embs: list[np.ndarray] = []
        for fp, cid in fp_to_cluster.items():
            if cid in source_set and fp in emb_lookup:
                pool_embs.append(emb_lookup[fp])

        if pool_embs:
            centroid = np.mean(pool_embs, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[pool["id"]] = centroid

    logger.info(
        "Computed centroids for %d/%d pools from cluster data",
        len(centroids), len([p for p in pools if not p.get("is_noise")]),
    )
    return centroids


# ---------------------------------------------------------------------------
# Public API: classify
# ---------------------------------------------------------------------------


def classify(
    conn: sqlite3.Connection,
    min_confidence: float = 0.3,
    skip_gemini: bool = False,
    no_api: bool = False,
) -> None:
    """Run the 3-tier cascading classification on all unclassified images.

    Args:
        conn: SQLite connection with cache tables populated.
        min_confidence: Minimum confidence to accept (below this -> noise).
        skip_gemini: If True, skip Tier 3 and send remaining to noise.
        no_api: If True, only run Tier 1 (SigLIP); everything else -> noise.
    """
    pools = cache.get_pools(conn)
    if not pools:
        logger.warning("No pools found -- cannot classify")
        return

    noise_pool_id = _find_noise_pool(pools)

    # Build pool lookup (non-noise pools only for classification targets)
    pool_by_id: dict[str, dict] = {p["id"]: p for p in pools}
    classifiable_pools = [p for p in pools if not p.get("is_noise")]

    if not classifiable_pools:
        logger.warning("No non-noise pools found -- all images will be noise")
        return

    # Get all non-duplicate images and their embeddings
    all_embeddings = cache.get_all_embeddings(conn)
    if not all_embeddings:
        logger.warning("No embeddings found -- cannot classify")
        return

    # Filter out images that already have cached classifications
    pending: list[tuple[str, np.ndarray]] = []
    for filepath, embedding in all_embeddings:
        existing = cache.get_cached_classification(conn, filepath)
        if existing is None:
            pending.append((filepath, embedding))

    if not pending:
        logger.info("All images already classified")
        return

    logger.info("Classifying %d images across %d pools", len(pending), len(classifiable_pools))

    # ------------------------------------------------------------------
    # Tier 1: SigLIP embedding match
    # ------------------------------------------------------------------
    #
    # Strategy: when cluster data is available (full mode), classify by
    # nearest pool centroid in embedding space.  This stays in the same
    # vector space end-to-end and avoids the indirect text→embedding
    # coupling. Falls back to text-based zero-shot for small mode where
    # no clusters exist.
    # ------------------------------------------------------------------

    pool_ids_ordered: list[str] = [p["id"] for p in classifiable_pools]

    # Stack image embeddings into matrix
    filepaths = [fp for fp, _ in pending]
    image_matrix = np.stack([emb for _, emb in pending])  # (num_images, 512)

    # Normalize
    image_norms = np.linalg.norm(image_matrix, axis=1, keepdims=True)
    image_norms = np.where(image_norms == 0, 1.0, image_norms)
    image_matrix = image_matrix / image_norms

    # Try centroid-based classification first (full mode with cluster data)
    pool_centroids = _compute_pool_centroids(conn, classifiable_pools, all_embeddings)
    use_centroids = len(pool_centroids) >= len(classifiable_pools) * 0.5

    if use_centroids:
        logger.info("Using centroid-based classification (%d pool centroids)", len(pool_centroids))
        # Build centroid matrix in same order as pool_ids_ordered
        # Pools without centroids fall back to text embedding
        from pool.phases.embeddings import encode_texts

        target_vecs: list[np.ndarray] = []
        for p in classifiable_pools:
            pid = p["id"]
            if pid in pool_centroids:
                target_vecs.append(pool_centroids[pid])
            else:
                # Fallback: encode the text description for this pool
                text = p.get("siglip_description") or p["description"]
                vec = encode_texts([text])[0]
                target_vecs.append(vec)

        target_matrix = np.stack(target_vecs)
    else:
        logger.info("Using text-based zero-shot classification (no cluster data)")
        from pool.phases.embeddings import encode_texts

        pool_texts = [p.get("siglip_description") or p["description"] for p in classifiable_pools]
        target_matrix = encode_texts(pool_texts)

    # Normalize target vectors
    target_norms = np.linalg.norm(target_matrix, axis=1, keepdims=True)
    target_norms = np.where(target_norms == 0, 1.0, target_norms)
    target_matrix = target_matrix / target_norms

    # Cosine similarity: (num_images, num_pools)
    similarity = image_matrix @ target_matrix.T

    # Classify based on confidence thresholds + margin
    tier2_candidates: list[tuple[str, float, str]] = []  # (filepath, conf, pool_id)
    tier1_classified = 0

    with _make_progress() as progress:
        task = progress.add_task("Tier 1: SigLIP embedding match", total=len(filepaths))

        for i, filepath in enumerate(filepaths):
            scores = similarity[i]
            sorted_indices = np.argsort(scores)[::-1]
            best_idx = int(sorted_indices[0])
            best_score = float(scores[best_idx])
            best_pool = pool_ids_ordered[best_idx]

            # Margin: gap between best and second-best pool score.
            # A large margin means the classification is unambiguous.
            if len(sorted_indices) > 1:
                second_score = float(scores[sorted_indices[1]])
                margin = best_score - second_score
            else:
                margin = best_score

            # Effective confidence combines absolute score and margin.
            # Centroid-based scores are typically higher (same space),
            # so margin prevents false positives when pools are close.
            best_conf = min(best_score * (0.5 + margin), 0.99) if use_centroids else best_score

            if best_conf > _HIGH_CONFIDENCE and margin > 0.05:
                # High confidence -- accept immediately
                pool_info = pool_by_id[best_pool]
                explanation = pool_info.get("siglip_description") or pool_info["description"]
                cache.upsert_classification(
                    conn,
                    filepath=filepath,
                    pool_id=best_pool,
                    confidence=round(best_conf, 4),
                    method=ClassificationMethod.SIGLIP.value,
                    explanation=f"SigLIP match: {explanation}",
                    ocr_text=None,
                )
                tier1_classified += 1
            elif best_conf < min_confidence:
                if no_api:
                    # Below noise threshold and no API -- assign to noise
                    cache.upsert_classification(
                        conn,
                        filepath=filepath,
                        pool_id=noise_pool_id or best_pool,
                        confidence=round(best_conf, 4),
                        method=ClassificationMethod.NOISE.value,
                        explanation="Below minimum confidence threshold",
                        ocr_text=None,
                    )
                else:
                    # Collect for Tier 2/3
                    tier2_candidates.append((filepath, best_conf, best_pool))
            else:
                # Medium confidence -- collect for Tier 2
                tier2_candidates.append((filepath, best_conf, best_pool))

            progress.advance(task)

    conn.commit()
    logger.info(
        "Tier 1 complete: %d high confidence, %d need further classification",
        tier1_classified, len(tier2_candidates),
    )

    # In no_api mode, everything not already classified goes to noise
    if no_api:
        for filepath, conf, pool_id in tier2_candidates:
            cache.upsert_classification(
                conn,
                filepath=filepath,
                pool_id=noise_pool_id or pool_id,
                confidence=round(conf, 4),
                method=ClassificationMethod.NOISE.value,
                explanation="No API mode -- insufficient SigLIP confidence",
                ocr_text=None,
            )
        conn.commit()
        return

    if not tier2_candidates:
        return

    # ------------------------------------------------------------------
    # Tier 2: OCR enrichment
    # ------------------------------------------------------------------
    pool_keywords = _build_keyword_map(pools)
    tier3_candidates: list[tuple[str, float, str]] = []  # (filepath, best_conf, best_pool)
    tier2_classified = 0

    # Step 1: Pre-compute OCR for all candidates in parallel.
    # Apple Vision OCR is CPU-bound; 4 workers balances throughput
    # against diminishing returns from context-switching overhead.
    ocr_results: dict[str, str] = {}  # filepath -> ocr_text

    with _make_progress() as progress:
        task = progress.add_task(
            "Tier 2: OCR extraction", total=len(tier2_candidates),
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_filepath = {
                executor.submit(_run_ocr, filepath): filepath
                for filepath, _conf, _pool in tier2_candidates
            }

            for future in as_completed(future_to_filepath):
                filepath = future_to_filepath[future]
                try:
                    ocr_results[filepath] = future.result()
                except Exception:
                    logger.debug("OCR future failed for %s", filepath, exc_info=True)
                    ocr_results[filepath] = ""
                progress.advance(task)

    # Step 2: Score and classify sequentially using the pre-computed OCR text.
    with _make_progress() as progress:
        task = progress.add_task(
            "Tier 2: OCR classification", total=len(tier2_candidates),
        )

        for filepath, siglip_conf, siglip_pool in tier2_candidates:
            ocr_text = ocr_results.get(filepath, "")

            if ocr_text:
                scores = _score_ocr_for_pools(ocr_text, pool_keywords)

                if scores and scores[0][1] >= 2:
                    # Clear winner with 2+ unique keyword matches
                    ocr_pool_id = scores[0][0]
                    ocr_matched = scores[0][2]

                    # Boost confidence when OCR confirms SigLIP
                    if ocr_pool_id == siglip_pool:
                        final_conf = min(siglip_conf + 0.2, 0.95)
                    else:
                        final_conf = min(0.5 + 0.05 * scores[0][1], 0.85)

                    pool_info = pool_by_id.get(ocr_pool_id, {})
                    explanation = (
                        f"OCR keywords matched: {', '.join(sorted(ocr_matched)[:5])}"
                    )
                    cache.upsert_classification(
                        conn,
                        filepath=filepath,
                        pool_id=ocr_pool_id,
                        confidence=round(final_conf, 4),
                        method=ClassificationMethod.OCR.value,
                        explanation=explanation,
                        ocr_text=ocr_text[:2000],  # cap stored OCR text
                    )
                    tier2_classified += 1
                elif siglip_conf > 0.5:
                    # OCR found text but no clear pool match; SigLIP was decent
                    pool_info = pool_by_id[siglip_pool]
                    explanation = pool_info.get("siglip_description") or pool_info["description"]
                    cache.upsert_classification(
                        conn,
                        filepath=filepath,
                        pool_id=siglip_pool,
                        confidence=round(siglip_conf, 4),
                        method=ClassificationMethod.OCR.value,
                        explanation=f"OCR found no decisive keywords; SigLIP match: {explanation}",
                        ocr_text=ocr_text[:2000],
                    )
                    tier2_classified += 1
                else:
                    # OCR inconclusive and SigLIP weak -- move to Tier 3
                    tier3_candidates.append((filepath, siglip_conf, siglip_pool))
            else:
                # No OCR text extracted
                if siglip_conf > 0.5:
                    pool_info = pool_by_id[siglip_pool]
                    explanation = pool_info.get("siglip_description") or pool_info["description"]
                    cache.upsert_classification(
                        conn,
                        filepath=filepath,
                        pool_id=siglip_pool,
                        confidence=round(siglip_conf, 4),
                        method=ClassificationMethod.SIGLIP.value,
                        explanation=f"SigLIP match (no OCR text): {explanation}",
                        ocr_text=None,
                    )
                    tier2_classified += 1
                else:
                    tier3_candidates.append((filepath, siglip_conf, siglip_pool))

            progress.advance(task)

    conn.commit()
    logger.info(
        "Tier 2 complete: %d OCR-resolved, %d need Gemini tiebreaker",
        tier2_classified, len(tier3_candidates),
    )

    if not tier3_candidates:
        return

    # ------------------------------------------------------------------
    # Tier 3: Gemini Flash tiebreaker
    # ------------------------------------------------------------------
    if skip_gemini:
        for filepath, conf, pool_id in tier3_candidates:
            cache.upsert_classification(
                conn,
                filepath=filepath,
                pool_id=noise_pool_id or pool_id,
                confidence=round(conf, 4),
                method=ClassificationMethod.NOISE.value,
                explanation="Gemini skipped -- assigned to noise",
                ocr_text=None,
            )
        conn.commit()
        logger.info("Tier 3 skipped: %d images assigned to noise", len(tier3_candidates))
        return

    # Initialize Gemini
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set -- assigning remaining to noise")
            for filepath, conf, pool_id in tier3_candidates:
                cache.upsert_classification(
                    conn,
                    filepath=filepath,
                    pool_id=noise_pool_id or pool_id,
                    confidence=round(conf, 4),
                    method=ClassificationMethod.NOISE.value,
                    explanation="No Google API key available",
                    ocr_text=None,
                )
            conn.commit()
            return

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    except ImportError:
        logger.warning("google-generativeai not installed -- assigning remaining to noise")
        for filepath, conf, pool_id in tier3_candidates:
            cache.upsert_classification(
                conn,
                filepath=filepath,
                pool_id=noise_pool_id or pool_id,
                confidence=round(conf, 4),
                method=ClassificationMethod.NOISE.value,
                explanation="Gemini SDK not available",
                ocr_text=None,
            )
        conn.commit()
        return

    # Valid pool IDs for validation
    valid_pool_ids = {p["id"] for p in classifiable_pools}

    gemini_classified = 0
    gemini_noise = 0
    max_workers = min(5, len(tier3_candidates))  # conservative to avoid rate limits

    with _make_progress() as progress:
        task = progress.add_task("Tier 3: Gemini Flash", total=len(tier3_candidates))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_filepath: dict = {}
            for filepath, _conf, _pool in tier3_candidates:
                future = executor.submit(
                    _call_gemini_with_retry,
                    filepath,
                    pools,
                    gemini_model,
                    retries=1,
                )
                future_to_filepath[future] = (filepath, _conf, _pool)

            for future in as_completed(future_to_filepath):
                filepath, fallback_conf, fallback_pool = future_to_filepath[future]

                try:
                    result = future.result()
                except Exception:
                    logger.debug("Gemini future failed for %s", filepath, exc_info=True)
                    result = {}

                if (
                    result
                    and result.get("pool_id") in valid_pool_ids
                    and result.get("confidence", 0) >= min_confidence
                ):
                    cache.upsert_classification(
                        conn,
                        filepath=filepath,
                        pool_id=result["pool_id"],
                        confidence=round(result["confidence"], 4),
                        method=ClassificationMethod.GEMINI.value,
                        explanation=result.get("explanation", ""),
                        ocr_text=None,
                    )
                    gemini_classified += 1
                else:
                    cache.upsert_classification(
                        conn,
                        filepath=filepath,
                        pool_id=noise_pool_id or fallback_pool,
                        confidence=round(fallback_conf, 4),
                        method=ClassificationMethod.NOISE.value,
                        explanation=result.get("explanation", "Gemini could not resolve classification"),
                        ocr_text=None,
                    )
                    gemini_noise += 1

                progress.advance(task)

    conn.commit()
    logger.info(
        "Tier 3 complete: %d Gemini-classified, %d assigned to noise",
        gemini_classified, gemini_noise,
    )


# ---------------------------------------------------------------------------
# Public API: explain_image
# ---------------------------------------------------------------------------


def explain_image(
    conn: sqlite3.Connection,
    filepath: str,
    no_api: bool = False,
) -> dict:
    """Explain why a single image was classified into its pool.

    Args:
        conn: SQLite connection.
        filepath: Path to the image to explain.
        no_api: If True, do not call Gemini for explanations.

    Returns:
        dict with keys: filepath, pool_id, pool_name, confidence, method, explanation.
    """
    resolved = str(Path(filepath).expanduser().resolve())
    pools = cache.get_pools(conn)
    pool_by_id: dict[str, dict] = {p["id"]: p for p in pools}

    # Check cached classification
    cached = cache.get_cached_classification(conn, resolved)

    if cached:
        pool_id = cached["pool_id"]
        pool_info = pool_by_id.get(pool_id, {})
        pool_name = pool_info.get("name", pool_id)
        confidence = cached.get("confidence", 0.0)
        method = cached.get("method", "unknown")
        cached_explanation = cached.get("explanation", "")

        if confidence > _HIGH_CONFIDENCE:
            siglip_desc = pool_info.get("siglip_description") or pool_info.get("description", "")
            explanation = (
                f"This screenshot is {confidence:.2f} similar to your "
                f'"{pool_name}" pool -- {cached_explanation or siglip_desc}'
            )
            return {
                "filepath": resolved,
                "pool_id": pool_id,
                "pool_name": pool_name,
                "confidence": confidence,
                "method": method,
                "explanation": explanation,
            }

        # Low/medium confidence -- try Gemini for a better explanation
        if not no_api:
            gemini_explanation = _get_gemini_explanation(resolved, pools)
            if gemini_explanation:
                explanation = (
                    f"This screenshot is {confidence:.2f} similar to your "
                    f'"{pool_name}" pool. {gemini_explanation}'
                )
                return {
                    "filepath": resolved,
                    "pool_id": pool_id,
                    "pool_name": pool_name,
                    "confidence": confidence,
                    "method": method,
                    "explanation": explanation,
                }

        # Fallback to cached explanation
        explanation = (
            f"This screenshot is {confidence:.2f} similar to your "
            f'"{pool_name}" pool -- {cached_explanation or "no further details available"}'
        )
        return {
            "filepath": resolved,
            "pool_id": pool_id,
            "pool_name": pool_name,
            "confidence": confidence,
            "method": method,
            "explanation": explanation,
        }

    # No cached classification -- run a quick SigLIP match
    from pool.phases.embeddings import encode_texts

    embedding_row = cache.get_cached_embedding(conn, resolved)
    classifiable_pools = [p for p in pools if not p.get("is_noise")]

    if embedding_row is not None and classifiable_pools:
        pool_texts = []
        pool_ids_ordered = []
        for p in classifiable_pools:
            pool_texts.append(p.get("siglip_description") or p["description"])
            pool_ids_ordered.append(p["id"])

        text_embeddings = encode_texts(pool_texts)
        img_emb = embedding_row.reshape(1, -1)
        img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        scores = (img_emb @ text_embeddings.T)[0]
        best_idx = int(np.argmax(scores))
        best_conf = float(scores[best_idx])
        best_pool = pool_ids_ordered[best_idx]
        pool_info = pool_by_id[best_pool]
        pool_name = pool_info.get("name", best_pool)

        # Try Gemini for a richer explanation
        if not no_api and best_conf < _HIGH_CONFIDENCE:
            gemini_explanation = _get_gemini_explanation(resolved, pools)
            if gemini_explanation:
                return {
                    "filepath": resolved,
                    "pool_id": best_pool,
                    "pool_name": pool_name,
                    "confidence": best_conf,
                    "method": "siglip",
                    "explanation": gemini_explanation,
                }

        siglip_desc = pool_info.get("siglip_description") or pool_info.get("description", "")
        return {
            "filepath": resolved,
            "pool_id": best_pool,
            "pool_name": pool_name,
            "confidence": best_conf,
            "method": "siglip",
            "explanation": (
                f"This screenshot is {best_conf:.2f} similar to your "
                f'"{pool_name}" pool -- {siglip_desc}'
            ),
        }

    # No embedding and no classification -- cannot explain
    return {
        "filepath": resolved,
        "pool_id": "",
        "pool_name": "Unknown",
        "confidence": 0.0,
        "method": "none",
        "explanation": "This image has not been processed yet. Run the full pipeline first.",
    }


def _get_gemini_explanation(filepath: str, pools: list[dict]) -> str:
    """Request a one-line explanation from Gemini Flash for a single image.

    Returns the explanation string, or empty string on failure.
    """
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return ""

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        pool_list_text = "\n".join(
            f'- "{p["name"]}": {p["description"]}'
            for p in pools
            if not p.get("is_noise")
        )

        prompt = (
            "Look at this screenshot and explain in one sentence why it belongs "
            "to a particular content category. Here are the categories:\n\n"
            f"{pool_list_text}\n\n"
            "Respond with a single plain-text sentence, no JSON."
        )

        img_path = Path(filepath)
        if not img_path.exists():
            return ""

        b64_data, mime_type = _image_to_base64_for_gemini(str(img_path))
        response = model.generate_content(
            [
                prompt,
                {"mime_type": mime_type, "data": base64_module.b64decode(b64_data)},
            ],
            generation_config={"temperature": 0.2, "max_output_tokens": 128},
        )
        return response.text.strip()
    except Exception:
        logger.debug("Gemini explanation failed for %s", filepath, exc_info=True)
        return ""
