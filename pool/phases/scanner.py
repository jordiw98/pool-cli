"""Phase 1: Scanner — directory walking, metadata extraction, perceptual dedup, thumbnails.

Scans a directory of screenshots, extracts image metadata, computes perceptual
hashes for near-duplicate detection, generates thumbnails, and stores everything
in the SQLite cache layer.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import imagehash
from PIL import Image

# Register HEIC/HEIF support with Pillow
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)

from pool import cache

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".heic", ".bmp", ".tiff"}
)

THUMBNAIL_DIR_NAME = ".pool_thumbnails"
THUMBNAIL_MAX_EDGE = 512
THUMBNAIL_QUALITY = 85
DUPLICATE_HAMMING_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan(conn: sqlite3.Connection, source_dir: str) -> dict[str, int]:
    """Run the full Phase 1 pipeline.

    1. Walk *source_dir* for image files.
    2. Extract metadata and compute perceptual hashes (cache-aware).
    3. Generate thumbnails.
    4. Detect near-duplicate groups via hamming distance on pHash.

    Returns ``{"total": int, "unique": int, "duplicates": int}``.
    """
    source = Path(source_dir)
    thumb_dir = source / THUMBNAIL_DIR_NAME
    thumb_dir.mkdir(exist_ok=True)

    image_paths = _collect_image_paths(source)

    if not image_paths:
        return {"total": 0, "unique": 0, "duplicates": 0}

    # --- Stage 1: metadata + hash + thumbnail ---
    _scan_images(conn, image_paths, thumb_dir)
    conn.commit()

    # --- Stage 2: perceptual dedup ---
    duplicates_found = _deduplicate(conn)
    conn.commit()

    total, unique = cache.get_image_count(conn)
    return {
        "total": total,
        "unique": unique,
        "duplicates": duplicates_found,
    }


# ---------------------------------------------------------------------------
# Directory walking
# ---------------------------------------------------------------------------

def _collect_image_paths(source: Path) -> list[Path]:
    """Recursively collect image file paths, skipping hidden dirs and thumbnails."""
    paths: list[Path] = []
    for root, dirs, files in os.walk(source):
        # Prune hidden directories and the thumbnail directory in-place so
        # os.walk does not descend into them.
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d != THUMBNAIL_DIR_NAME
        ]
        for fname in files:
            if fname.startswith("."):
                continue
            p = Path(root) / fname
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _scan_images(
    conn: sqlite3.Connection,
    paths: list[Path],
    thumb_dir: Path,
) -> None:
    """Extract metadata, compute pHash, and generate thumbnail for each image."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning images", total=len(paths))
        for path in paths:
            try:
                _process_single_image(conn, path, thumb_dir)
            except Exception:
                logger.warning("Skipping %s — failed to process", path)
            progress.advance(task)


def _process_single_image(
    conn: sqlite3.Connection,
    path: Path,
    thumb_dir: Path,
) -> None:
    """Process one image: check cache, extract metadata, hash, thumbnail."""
    filepath_str = str(path)
    stat = path.stat()
    file_size = stat.st_size

    # --- Cache check: same path + same size means nothing changed ---
    cached = cache.get_cached_image(conn, filepath_str)
    if cached and cached.get("file_size") == file_size and cached.get("phash"):
        # Ensure thumbnail still exists on disk; if not, regenerate.
        thumb_path = cached.get("thumbnail_path")
        if thumb_path and Path(thumb_path).exists():
            return
        # Thumbnail is missing — fall through to regenerate it, but keep the
        # rest of the cached data intact.

    # --- Open image ---
    img = Image.open(path)
    # Some formats are lazy-loaded; accessing size forces header parse.
    width, height = img.size

    # --- Dates ---
    created_at = _extract_created_at(img, stat)
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    # --- Perceptual hash ---
    phash_value = str(imagehash.phash(img))

    # --- Thumbnail ---
    thumbnail_path = _generate_thumbnail(img, path, thumb_dir)

    # --- Upsert ---
    cache.upsert_image(
        conn,
        filepath=filepath_str,
        filename=path.name,
        file_size=file_size,
        width=width,
        height=height,
        created_at=created_at,
        modified_at=modified_at,
        phash=phash_value,
        is_duplicate=0,
        duplicate_of=None,
        thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
    )


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _extract_created_at(img: Image.Image, stat: os.stat_result) -> str:
    """Try EXIF DateTimeOriginal / DateTimeDigitized, fall back to filesystem."""
    try:
        exif_data = img.getexif()
        if exif_data:
            # DateTimeOriginal (36867) or DateTimeDigitized (36868)
            for tag_id in (36867, 36868):
                val = exif_data.get(tag_id)
                if val:
                    dt = datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                    return dt.replace(tzinfo=timezone.utc).isoformat()
            # Fallback: generic DateTime tag (306)
            val = exif_data.get(306)
            if val:
                dt = datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        pass  # EXIF parsing can fail on many image types; that is fine.

    # Filesystem fallback: birthtime if available, else mtime.
    ts = getattr(stat, "st_birthtime", None) or stat.st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

def _generate_thumbnail(
    img: Image.Image,
    original_path: Path,
    thumb_dir: Path,
) -> Optional[Path]:
    """Resize to <=512px max edge, save as JPEG in the thumbnail directory."""
    try:
        thumb_name = original_path.stem + ".jpg"
        thumb_path = thumb_dir / thumb_name

        # Handle name collisions (different source subdirectories may share names).
        if thumb_path.exists():
            # Append a deterministic hash of the full original path for uniqueness.
            import hashlib
            path_hash = hashlib.md5(str(original_path).encode()).hexdigest()[:8]
            thumb_name = f"{original_path.stem}_{path_hash}.jpg"
            thumb_path = thumb_dir / thumb_name

        copy = img.copy()
        copy.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE), Image.LANCZOS)
        # Ensure RGB for JPEG output (handles RGBA, palette, etc.).
        if copy.mode not in ("RGB", "L"):
            copy = copy.convert("RGB")
        copy.save(str(thumb_path), format="JPEG", quality=THUMBNAIL_QUALITY)
        return thumb_path
    except Exception:
        logger.warning(
            "Failed to generate thumbnail for %s", original_path, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Perceptual deduplication
# ---------------------------------------------------------------------------

def _deduplicate(conn: sqlite3.Connection) -> int:
    """Mark near-duplicate images using hamming distance on pHash.

    Groups images by hash similarity. The first image encountered in each
    group (sorted by filepath for determinism) becomes the representative;
    all others are flagged ``is_duplicate=1`` with ``duplicate_of`` pointing
    to the representative.

    Returns the number of images marked as duplicates.
    """
    rows = conn.execute(
        "SELECT filepath, phash FROM images WHERE phash IS NOT NULL ORDER BY filepath"
    ).fetchall()

    if not rows:
        return 0

    # Parse hashes up front.
    entries: list[tuple[str, imagehash.ImageHash]] = []
    for row in rows:
        try:
            entries.append((row["filepath"], imagehash.hex_to_hash(row["phash"])))
        except Exception:
            logger.warning("Invalid phash for %s — skipping dedup", row["filepath"])

    # Union-Find for grouping.
    parent: dict[str, str] = {fp: fp for fp, _ in entries}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the lexicographically smaller path as root so the
            # representative is deterministic.
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    # Pairwise comparison — O(n^2) but fine for typical screenshot collections
    # (a few thousand images).  For very large sets a VP-tree or BK-tree would
    # be more appropriate, but this keeps the dependency footprint small.
    n = len(entries)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        total_comparisons = n * (n - 1) // 2
        task = progress.add_task("Deduplicating", total=total_comparisons)
        for i in range(n):
            fp_a, hash_a = entries[i]
            for j in range(i + 1, n):
                fp_b, hash_b = entries[j]
                if hash_a - hash_b < DUPLICATE_HAMMING_THRESHOLD:
                    union(fp_a, fp_b)
                progress.advance(task)

    # Build groups.
    groups: dict[str, list[str]] = {}
    for fp, _ in entries:
        root = find(fp)
        groups.setdefault(root, []).append(fp)

    # Mark duplicates.
    duplicate_count = 0
    for representative, members in groups.items():
        for member in members:
            if member == representative:
                # Representative — ensure it is not marked as a duplicate.
                conn.execute(
                    "UPDATE images SET is_duplicate = 0, duplicate_of = NULL WHERE filepath = ?",
                    (representative,),
                )
            else:
                conn.execute(
                    "UPDATE images SET is_duplicate = 1, duplicate_of = ? WHERE filepath = ?",
                    (representative, member),
                )
                duplicate_count += 1

    return duplicate_count
