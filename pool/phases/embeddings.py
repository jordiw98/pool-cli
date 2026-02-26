"""Phase 2: Embeddings, dimensionality reduction, and clustering.

Loads SigLIP2 (ViT-B-16-SigLIP2-512) to compute 512-D image embeddings,
then runs UMAP + HDBSCAN for cluster discovery.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from pool import cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton for the SigLIP model, transforms, and tokenizer.
# Loaded once on first use and reused across all calls.
# ---------------------------------------------------------------------------

_model: Optional[torch.nn.Module] = None
_preprocess = None
_tokenizer = None
_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    """Return MPS device when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model() -> tuple:
    """Load SigLIP2 model, transforms, and tokenizer (cached as singleton).

    Returns:
        (model, preprocess, tokenizer, device)
    """
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        return _model, _preprocess, _tokenizer, _device

    import open_clip

    _device = _get_device()
    logger.info("Loading SigLIP2 on %s", _device)

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-512",
        pretrained="webli",
    )
    _model = _model.to(_device)
    _model.eval()

    _tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2-512")

    logger.info("SigLIP2 loaded successfully")
    return _model, _preprocess, _tokenizer, _device


# ---------------------------------------------------------------------------
# Public helpers for Phase 4 (classification)
# ---------------------------------------------------------------------------


def get_model_and_tokenizer():
    """Return the loaded SigLIP model and tokenizer.

    The model is loaded on first call and cached for subsequent use.

    Returns:
        (model, tokenizer)
    """
    model, _pp, tokenizer, _dev = _load_model()
    return model, tokenizer


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of text descriptions into normalized embedding vectors.

    Args:
        texts: Text strings to encode.

    Returns:
        np.ndarray of shape (len(texts), 512) with L2-normalized rows.
    """
    if not texts:
        return np.empty((0, 512), dtype=np.float32)

    model, _pp, tokenizer, device = _load_model()

    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        text_features = model.encode_text(tokens)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    return text_features.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 2a: Compute embeddings
# ---------------------------------------------------------------------------


def _load_and_preprocess(thumbnail_path: str, preprocess) -> Optional[torch.Tensor]:
    """Safely load a thumbnail and apply the model's preprocessing.

    Returns None if the thumbnail is missing or corrupt.
    """
    path = Path(thumbnail_path)
    if not path.exists():
        logger.warning("Thumbnail not found: %s", thumbnail_path)
        return None
    try:
        img = Image.open(path).convert("RGB")
        return preprocess(img)
    except Exception:
        logger.warning("Failed to load thumbnail: %s", thumbnail_path, exc_info=True)
        return None


def compute_embeddings(conn: sqlite3.Connection) -> None:
    """Compute SigLIP2 embeddings for all non-duplicate images.

    Skips images that already have a cached embedding. Stores each result via
    the cache layer and commits periodically to avoid data loss on interruption.
    """
    images = cache.get_all_images(conn)
    if not images:
        logger.info("No images to embed")
        return

    # Filter to images that still need embeddings
    pending: list[dict] = []
    for img in images:
        if not img.get("thumbnail_path"):
            logger.debug("Skipping %s — no thumbnail", img["filepath"])
            continue
        cached = cache.get_cached_embedding(conn, img["filepath"])
        if cached is not None:
            continue
        pending.append(img)

    if not pending:
        logger.info("All embeddings already cached (%d images)", len(images))
        return

    model, preprocess, _tok, device = _load_model()
    batch_size = 32 if device.type == "mps" else 16

    logger.info(
        "Computing embeddings for %d images (batch_size=%d, device=%s)",
        len(pending), batch_size, device,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Embedding", total=len(pending))
        batches_since_commit = 0

        for batch_start in range(0, len(pending), batch_size):
            batch_images = pending[batch_start : batch_start + batch_size]

            # Preprocess thumbnails, tracking which ones succeeded
            tensors: list[torch.Tensor] = []
            valid_indices: list[int] = []
            for i, img in enumerate(batch_images):
                tensor = _load_and_preprocess(img["thumbnail_path"], preprocess)
                if tensor is not None:
                    tensors.append(tensor)
                    valid_indices.append(i)

            if tensors:
                batch_tensor = torch.stack(tensors).to(device)

                with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                    features = model.encode_image(batch_tensor)
                    features = torch.nn.functional.normalize(features, dim=-1)

                embeddings_np = features.cpu().numpy().astype(np.float32)

                for j, idx in enumerate(valid_indices):
                    filepath = batch_images[idx]["filepath"]
                    cache.upsert_embedding(conn, filepath, embeddings_np[j])

            progress.advance(task, advance=len(batch_images))
            batches_since_commit += 1

            if batches_since_commit >= 10:
                conn.commit()
                batches_since_commit = 0

    # Final commit for any remaining writes
    conn.commit()
    logger.info("Embedding computation complete")


# ---------------------------------------------------------------------------
# Phase 2b: UMAP + HDBSCAN clustering
# ---------------------------------------------------------------------------


def cluster(conn: sqlite3.Connection) -> None:
    """Run UMAP dimensionality reduction followed by HDBSCAN clustering.

    Reads all cached embeddings, reduces to 15 dimensions with UMAP (cosine
    metric), then clusters with HDBSCAN. Cluster IDs (including -1 for noise)
    are written back to the cache.
    """
    import hdbscan
    import umap

    all_embeddings = cache.get_all_embeddings(conn)
    if not all_embeddings:
        logger.warning("No embeddings found — skipping clustering")
        return

    filepaths = [fp for fp, _emb in all_embeddings]
    matrix = np.stack([emb for _fp, emb in all_embeddings])
    n = len(filepaths)

    if n < 30:
        logger.warning("Too few images for clustering (%d) — skipping", n)
        return

    # Adaptive parameters based on dataset size
    n_neighbors = min(30, max(10, n // 20))      # 10-30, scales with data
    n_components = min(15, max(5, n // 50))       # 5-15, avoid over-reducing small sets
    min_cluster_size = max(5, n // 200)           # at least 5, ~0.5% of dataset
    min_samples = max(2, min_cluster_size // 3)   # ~1/3 of min_cluster_size

    logger.info(
        "UMAP/HDBSCAN params for n=%d: n_neighbors=%d, n_components=%d, min_cluster=%d, min_samples=%d",
        n, n_neighbors, n_components, min_cluster_size, min_samples,
    )

    logger.info("Clustering %d embeddings", n)

    # Dimensionality reduction
    reducer = umap.UMAP(
        n_components=n_components,
        metric="cosine",
        n_neighbors=n_neighbors,
        random_state=42,
    )
    reduced = reducer.fit_transform(matrix)

    # Clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(reduced)

    # Build mapping and write back
    filepath_to_cluster: dict[str, int] = {}
    for fp, label in zip(filepaths, labels):
        filepath_to_cluster[fp] = int(label)

    cache.update_cluster_ids(conn, filepath_to_cluster)
    conn.commit()

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    logger.info("Clustering complete: %d clusters, %d noise points", n_clusters, n_noise)
