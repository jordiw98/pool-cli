"""Phase 5 — Temporal analysis, loop detection, and action generation.

Computes temporal signatures for each pool, classifies loop status,
generates Claude-powered action suggestions, and produces an opening insight.
"""

from __future__ import annotations

import base64
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from pool import cache
from pool.models import LoopStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BURST_WINDOW_HOURS = 72
_BURST_MIN_COUNT = 5
_MODEL = "claude-sonnet-4-20250514"
_MAX_RETRIES = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(conn: sqlite3.Connection, source_dir: str, no_api: bool = False) -> None:
    """Run the full analysis phase: temporals, loop status, actions, insight.

    Parameters
    ----------
    conn:
        Open SQLite connection with pool cache tables.
    source_dir:
        Root directory of the screenshot library (used for thumbnail paths).
    no_api:
        When True, skip all Claude API calls and produce generic fallbacks.
    """
    pools = cache.get_pools(conn)
    if not pools:
        logger.warning("No pools found — skipping analysis.")
        return

    # Build a filepath -> image metadata lookup once for all pools.
    all_images = {img["filepath"]: img for img in cache.get_all_images(conn)}

    # Step 1 + 2: Temporal signatures and loop classification
    pool_temporals = _compute_all_temporals(conn, pools, all_images)

    # Step 3: Action generation
    _generate_actions(conn, pools, pool_temporals, all_images, source_dir, no_api)

    # Step 4: Opening insight
    _generate_opening_insight(conn, pools, pool_temporals, no_api)

    conn.commit()


# ---------------------------------------------------------------------------
# Step 1 + 2: Temporal signatures & loop classification
# ---------------------------------------------------------------------------


def _compute_all_temporals(
    conn: sqlite3.Connection,
    pools: list[dict],
    all_images: dict[str, dict],
) -> dict[str, dict]:
    """Compute and persist temporal signatures for every pool.

    Returns a mapping of pool_id -> temporal dict for downstream use.
    """
    existing = {t["pool_id"]: t for t in cache.get_temporals(conn)}
    result: dict[str, dict] = {}

    for pool in pools:
        pid = pool["id"]

        # Use cached temporal if already present.
        if pid in existing:
            result[pid] = existing[pid]
            continue

        classifications = cache.get_classifications_by_pool(conn, pid)
        dates = _extract_sorted_dates(classifications, all_images)

        temporal = _build_temporal(pid, dates)
        temporal["loop_status"] = _classify_loop(temporal, dates).value

        cache.save_temporal(conn, **temporal)
        result[pid] = temporal

    return result


def _extract_sorted_dates(
    classifications: list[dict],
    all_images: dict[str, dict],
) -> list[datetime]:
    """Return sorted datetime objects for classified images that have created_at."""
    dates: list[datetime] = []
    for cls in classifications:
        img = all_images.get(cls["filepath"])
        if img and img.get("created_at"):
            try:
                dates.append(datetime.fromisoformat(img["created_at"]))
            except (ValueError, TypeError):
                continue
    dates.sort()
    return dates


def _build_temporal(pool_id: str, dates: list[datetime]) -> dict:
    """Build a temporal-signature dict from sorted dates."""
    if not dates:
        return {
            "pool_id": pool_id,
            "first_date": None,
            "last_date": None,
            "span_days": 0,
            "total_count": 0,
            "frequency_per_month": 0.0,
            "burst_count": 0,
            "longest_gap_days": 0,
        }

    first = dates[0]
    last = dates[-1]
    span = (last - first).days
    total = len(dates)
    freq = total / max(span / 30.0, 1.0)

    bursts = _count_bursts(dates)
    longest_gap = _longest_gap(dates)

    return {
        "pool_id": pool_id,
        "first_date": first.isoformat(),
        "last_date": last.isoformat(),
        "span_days": span,
        "total_count": total,
        "frequency_per_month": round(freq, 2),
        "burst_count": bursts,
        "longest_gap_days": longest_gap,
    }


def _count_bursts(dates: list[datetime]) -> int:
    """Count distinct bursts (5+ screenshots within 72 hours)."""
    if len(dates) < _BURST_MIN_COUNT:
        return 0

    bursts = 0
    window_start = 0

    while window_start < len(dates):
        window_end = window_start
        while (
            window_end + 1 < len(dates)
            and (dates[window_end + 1] - dates[window_start]).total_seconds()
            <= _BURST_WINDOW_HOURS * 3600
        ):
            window_end += 1

        count_in_window = window_end - window_start + 1
        if count_in_window >= _BURST_MIN_COUNT:
            bursts += 1
            # Advance past this burst entirely.
            window_start = window_end + 1
        else:
            window_start += 1

    return bursts


def _longest_gap(dates: list[datetime]) -> int:
    """Return the longest gap in days between consecutive dates."""
    if len(dates) < 2:
        return 0
    return max((dates[i + 1] - dates[i]).days for i in range(len(dates) - 1))


# ---------------------------------------------------------------------------
# Step 2: Loop classification
# ---------------------------------------------------------------------------

def _classify_loop(temporal: dict, dates: list[datetime]) -> LoopStatus:
    """Determine loop status from temporal signature and raw dates."""
    total = temporal["total_count"]

    if total < 3:
        return LoopStatus.UNKNOWN

    if not dates:
        return LoopStatus.UNKNOWN

    now = datetime.now(tz=dates[-1].tzinfo)
    last = dates[-1]
    first = dates[0]
    span_days = temporal["span_days"]
    longest_gap = temporal["longest_gap_days"]
    days_since_last = (now - last).days
    burst_count = temporal["burst_count"]

    # ONE_SHOT: all within 14 days, ended 30+ days ago.
    if span_days <= 14 and days_since_last >= 30:
        return LoopStatus.ONE_SHOT

    # ABANDONED: 10+ images, 90+ days silence since last screenshot.
    if total >= 10 and days_since_last >= 90:
        return LoopStatus.ABANDONED

    # CYCLICAL: 2+ bursts separated by gaps of 30+ days.
    if burst_count >= 2 and longest_gap >= 30:
        return LoopStatus.CYCLICAL

    # ACTIVE: last within 30 days, or consistent 3+ months with no gap > 60 days.
    if days_since_last <= 30:
        return LoopStatus.ACTIVE

    span_months = span_days / 30.0
    if span_months >= 3 and longest_gap <= 60:
        return LoopStatus.ACTIVE

    return LoopStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Step 3: Action generation
# ---------------------------------------------------------------------------


def _generate_actions(
    conn: sqlite3.Connection,
    pools: list[dict],
    pool_temporals: dict[str, dict],
    all_images: dict[str, dict],
    source_dir: str,
    no_api: bool,
) -> None:
    """Generate and persist pool actions (Claude or generic fallback)."""
    existing_actions = {a["pool_id"] for a in cache.get_actions(conn)}
    pools_needing_actions: list[dict] = []

    for pool in pools:
        pid = pool["id"]
        if pid in existing_actions:
            continue

        temporal = pool_temporals.get(pid, {})
        count = temporal.get("total_count", 0)

        # Skip noise pools and tiny pools — save generic note.
        if pool.get("is_noise") or count < 3:
            cache.save_action(
                conn,
                pool_id=pid,
                action=None,
                why=None,
                notes="Too few images to detect a pattern." if count < 3 else "Uncategorized screenshots.",
                has_action=0,
            )
            continue

        pools_needing_actions.append(pool)

    if not pools_needing_actions:
        return

    if no_api:
        _save_generic_actions(conn, pools_needing_actions, pool_temporals)
        return

    _call_claude_for_actions(conn, pools_needing_actions, pool_temporals, all_images, source_dir)


def _save_generic_actions(
    conn: sqlite3.Connection,
    pools: list[dict],
    pool_temporals: dict[str, dict],
) -> None:
    """Save simple template actions when API is unavailable."""
    for pool in pools:
        pid = pool["id"]
        temporal = pool_temporals.get(pid, {})
        status = temporal.get("loop_status", "unknown")
        count = temporal.get("total_count", 0)

        note = f"{count} screenshots"
        if temporal.get("first_date") and temporal.get("last_date"):
            note += f" from {temporal['first_date'][:10]} to {temporal['last_date'][:10]}"
        note += f" ({status})"

        cache.save_action(
            conn,
            pool_id=pid,
            action=None,
            why=None,
            notes=note,
            has_action=0,
        )


def _call_claude_for_actions(
    conn: sqlite3.Connection,
    pools: list[dict],
    pool_temporals: dict[str, dict],
    all_images: dict[str, dict],
    source_dir: str,
) -> None:
    """Batch all pools into one Claude call for action generation."""
    import anthropic

    client = anthropic.Anthropic()

    # Build the user message with pool info and thumbnails.
    user_parts: list[dict] = []
    pool_id_order: list[str] = []

    for pool in pools:
        pid = pool["id"]
        temporal = pool_temporals.get(pid, {})
        pool_id_order.append(pid)

        header = (
            f'Pool: "{pool["name"]}" ({temporal.get("total_count", 0)} images)\n'
            f'Span: {temporal.get("first_date", "unknown")[:10] if temporal.get("first_date") else "unknown"}'
            f' to {temporal.get("last_date", "unknown")[:10] if temporal.get("last_date") else "unknown"}\n'
            f'Loop status: {temporal.get("loop_status", "unknown")}\n'
            f'Intent: {pool.get("intent", "unknown")}\n'
            f'Pool ID: {pid}\n'
        )
        user_parts.append({"type": "text", "text": header})

        # Attach up to 10 representative thumbnails (highest confidence).
        thumbnails = _get_top_thumbnails(pid, conn, all_images, source_dir, limit=10)
        for media_type, b64 in thumbnails:
            user_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            })

    system_prompt = (
        "You are writing action suggestions and notes for Pool, an intelligent screenshot organizer.\n\n"
        "For each pool below, suggest ONE specific action that would be genuinely useful.\n\n"
        "Rules:\n"
        "- If the content feels personal or emotional, suggest no action. Just acknowledge what's there.\n"
        "- Be specific, not generic. \"Build a playlist from your last 6 months — 38 tracks identified\" "
        "not \"Create Spotify playlist\"\n"
        "- If the loop is abandoned, be gentle: acknowledge it ended, offer to reopen only if relevant.\n"
        "- If the loop is cyclical, note the pattern.\n"
        "- Some pools should have NO action. Set has_action to false for those.\n\n"
        "For Notes: write a one-liner that references temporal patterns or behavioral observations. "
        "Sound like a perceptive friend, not a database.\n\n"
        "Return a JSON array (no markdown fences). Each element:\n"
        "{\n"
        '  "pool_id": "...",\n'
        '  "action": "..." or null,\n'
        '  "why": "..." or null,\n'
        '  "notes": "...",\n'
        '  "has_action": true/false\n'
        "}"
    )

    user_text_intro = (
        "Here are the pools with their temporal data and representative screenshots:\n\n"
    )
    content: list[dict] = [{"type": "text", "text": user_text_intro}] + user_parts

    parsed = _claude_call_with_retry(
        client,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
        pool_ids=pool_id_order,
    )

    # Map results by pool_id for reliable lookup.
    actions_by_id: dict[str, dict] = {}
    if parsed:
        for item in parsed:
            actions_by_id[item.get("pool_id", "")] = item

    for pool in pools:
        pid = pool["id"]
        item = actions_by_id.get(pid)
        if item:
            cache.save_action(
                conn,
                pool_id=pid,
                action=item.get("action"),
                why=item.get("why"),
                notes=item.get("notes", ""),
                has_action=1 if item.get("has_action") else 0,
            )
        else:
            # Claude omitted this pool — save generic.
            cache.save_action(
                conn,
                pool_id=pid,
                action=None,
                why=None,
                notes="",
                has_action=0,
            )


def _get_top_thumbnails(
    pool_id: str,
    conn: sqlite3.Connection,
    all_images: dict[str, dict],
    source_dir: str,
    limit: int = 10,
) -> list[tuple[str, str]]:
    """Return up to *limit* (media_type, base64) pairs for a pool's best images."""
    classifications = cache.get_classifications_by_pool(conn, pool_id)
    # Already sorted by confidence DESC from cache layer.

    results: list[tuple[str, str]] = []
    for cls in classifications:
        if len(results) >= limit:
            break
        img = all_images.get(cls["filepath"])
        if not img:
            continue

        # Prefer thumbnail, fall back to original.
        thumb_path = img.get("thumbnail_path") or cls["filepath"]
        path = Path(thumb_path)
        if not path.is_absolute():
            path = Path(source_dir) / path

        if not path.exists():
            # Try the original if thumbnail is missing.
            path = Path(cls["filepath"])
            if not path.exists():
                continue

        suffix = path.suffix.lower()
        media_type = _suffix_to_media_type(suffix)
        if not media_type:
            continue

        try:
            data = path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            results.append((media_type, b64))
        except OSError:
            continue

    return results


def _suffix_to_media_type(suffix: str) -> Optional[str]:
    """Map file extension to MIME type supported by Claude vision."""
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mapping.get(suffix)


# ---------------------------------------------------------------------------
# Step 4: Opening insight
# ---------------------------------------------------------------------------


def _generate_opening_insight(
    conn: sqlite3.Connection,
    pools: list[dict],
    pool_temporals: dict[str, dict],
    no_api: bool,
) -> None:
    """Generate a single-sentence opening observation about the library."""
    existing = cache.get_state(conn, "opening_insight")
    if existing is not None:
        return

    if no_api:
        cache.set_state(conn, "opening_insight", "")
        return

    import anthropic

    client = anthropic.Anthropic()

    summaries: list[str] = []
    for pool in pools:
        pid = pool["id"]
        t = pool_temporals.get(pid, {})
        first = (t.get("first_date") or "?")[:10]
        last = (t.get("last_date") or "?")[:10]
        line = (
            f'- "{pool["name"]}": {t.get("total_count", 0)} images, '
            f'{first} to {last}, '
            f'loop={t.get("loop_status", "unknown")}, '
            f'intent={pool.get("intent", "unknown")}'
        )
        summaries.append(line)

    user_text = (
        "Here's a summary of one person's screenshot library organized into pools:\n\n"
        + "\n".join(summaries)
        + "\n\n"
        "Write ONE opening observation about this person — something perceptive that connects "
        "patterns across pools. One sentence. Sound like a friend who just saw your phone for "
        "the first time and noticed something real."
    )

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": user_text}],
        )
        insight = response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Opening-insight Claude call failed: %s", exc)
        insight = ""

    cache.set_state(conn, "opening_insight", insight)


# ---------------------------------------------------------------------------
# Claude call helper
# ---------------------------------------------------------------------------


def _claude_call_with_retry(
    client,
    *,
    system: str,
    messages: list[dict],
    max_tokens: int,
    pool_ids: list[str],
) -> Optional[list[dict]]:
    """Call Claude, parse JSON response, retry once on failure."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if Claude added them despite instruction.
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError, KeyError) as exc:
            logger.warning("Action-generation parse error (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("Action-generation API error (attempt %d): %s", attempt + 1, exc)

    return None
