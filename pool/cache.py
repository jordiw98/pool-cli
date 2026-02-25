"""SQLite cache layer for Pool CLI.

Everything is cached: metadata, hashes, embeddings, OCR text, classifications, pool definitions.
Re-run on same folder = ~30 seconds.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


DB_NAME = ".pool_cache.db"


def get_db_path(source_dir: str) -> Path:
    """Cache lives inside the source directory as .pool_cache.db."""
    return Path(source_dir) / DB_NAME


def connect(source_dir: str) -> sqlite3.Connection:
    db_path = get_db_path(source_dir)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            filepath TEXT PRIMARY KEY,
            filename TEXT,
            file_size INTEGER,
            width INTEGER,
            height INTEGER,
            created_at TEXT,
            modified_at TEXT,
            phash TEXT,
            is_duplicate INTEGER DEFAULT 0,
            duplicate_of TEXT,
            thumbnail_path TEXT
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            filepath TEXT PRIMARY KEY,
            embedding BLOB,
            cluster_id INTEGER,
            FOREIGN KEY (filepath) REFERENCES images(filepath)
        );

        CREATE TABLE IF NOT EXISTS classifications (
            filepath TEXT PRIMARY KEY,
            pool_id TEXT,
            confidence REAL,
            method TEXT,
            explanation TEXT,
            ocr_text TEXT,
            FOREIGN KEY (filepath) REFERENCES images(filepath)
        );

        CREATE TABLE IF NOT EXISTS pools (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            intent TEXT,
            siglip_description TEXT,
            source_clusters TEXT,  -- JSON array
            match_count INTEGER DEFAULT 0,
            top_matches TEXT,     -- JSON array of filepaths
            is_noise INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS temporals (
            pool_id TEXT PRIMARY KEY,
            first_date TEXT,
            last_date TEXT,
            span_days INTEGER,
            total_count INTEGER,
            frequency_per_month REAL,
            burst_count INTEGER,
            longest_gap_days INTEGER,
            loop_status TEXT,
            FOREIGN KEY (pool_id) REFERENCES pools(id)
        );

        CREATE TABLE IF NOT EXISTS actions (
            pool_id TEXT PRIMARY KEY,
            action TEXT,
            why TEXT,
            notes TEXT,
            has_action INTEGER DEFAULT 0,
            FOREIGN KEY (pool_id) REFERENCES pools(id)
        );

        CREATE TABLE IF NOT EXISTS pipeline_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
        CREATE INDEX IF NOT EXISTS idx_classifications_pool ON classifications(pool_id);
        CREATE INDEX IF NOT EXISTS idx_images_duplicate ON images(is_duplicate);
    """)
    conn.commit()


# --- Image metadata ---

def get_cached_image(conn: sqlite3.Connection, filepath: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM images WHERE filepath = ?", (filepath,)).fetchone()
    return dict(row) if row else None


def upsert_image(conn: sqlite3.Connection, **kwargs) -> None:
    cols = list(kwargs.keys())
    placeholders = ", ".join(["?"] * len(cols))
    updates = ", ".join([f"{c} = excluded.{c}" for c in cols if c != "filepath"])
    sql = f"""
        INSERT INTO images ({', '.join(cols)}) VALUES ({placeholders})
        ON CONFLICT(filepath) DO UPDATE SET {updates}
    """
    conn.execute(sql, list(kwargs.values()))


def get_all_images(conn: sqlite3.Connection, include_duplicates: bool = False) -> list[dict]:
    if include_duplicates:
        rows = conn.execute("SELECT * FROM images").fetchall()
    else:
        rows = conn.execute("SELECT * FROM images WHERE is_duplicate = 0").fetchall()
    return [dict(r) for r in rows]


def get_image_count(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (total, unique) image counts."""
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    unique = conn.execute("SELECT COUNT(*) FROM images WHERE is_duplicate = 0").fetchone()[0]
    return total, unique


# --- Embeddings ---

def get_cached_embedding(conn: sqlite3.Connection, filepath: str) -> Optional[np.ndarray]:
    row = conn.execute("SELECT embedding FROM embeddings WHERE filepath = ?", (filepath,)).fetchone()
    if row and row[0]:
        return np.frombuffer(row[0], dtype=np.float32)
    return None


def upsert_embedding(conn: sqlite3.Connection, filepath: str, embedding: np.ndarray, cluster_id: Optional[int] = None) -> None:
    blob = embedding.astype(np.float32).tobytes()
    conn.execute("""
        INSERT INTO embeddings (filepath, embedding, cluster_id) VALUES (?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET embedding = excluded.embedding, cluster_id = excluded.cluster_id
    """, (filepath, blob, cluster_id))


def get_all_embeddings(conn: sqlite3.Connection) -> list[tuple[str, np.ndarray]]:
    rows = conn.execute("""
        SELECT e.filepath, e.embedding FROM embeddings e
        JOIN images i ON e.filepath = i.filepath
        WHERE i.is_duplicate = 0 AND e.embedding IS NOT NULL
    """).fetchall()
    return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]


def update_cluster_ids(conn: sqlite3.Connection, filepath_to_cluster: dict[str, int]) -> None:
    conn.executemany(
        "UPDATE embeddings SET cluster_id = ? WHERE filepath = ?",
        [(cid, fp) for fp, cid in filepath_to_cluster.items()]
    )


# --- Classifications ---

def get_cached_classification(conn: sqlite3.Connection, filepath: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM classifications WHERE filepath = ?", (filepath,)).fetchone()
    return dict(row) if row else None


def upsert_classification(conn: sqlite3.Connection, **kwargs) -> None:
    cols = list(kwargs.keys())
    placeholders = ", ".join(["?"] * len(cols))
    updates = ", ".join([f"{c} = excluded.{c}" for c in cols if c != "filepath"])
    sql = f"""
        INSERT INTO classifications ({', '.join(cols)}) VALUES ({placeholders})
        ON CONFLICT(filepath) DO UPDATE SET {updates}
    """
    conn.execute(sql, list(kwargs.values()))


def get_classifications_by_pool(conn: sqlite3.Connection, pool_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM classifications WHERE pool_id = ? ORDER BY confidence DESC",
        (pool_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_classification_breakdown(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("""
        SELECT method, COUNT(*) as cnt FROM classifications GROUP BY method
    """).fetchall()
    return {r[0]: r[1] for r in rows}


# --- Pools ---

def save_pools(conn: sqlite3.Connection, pools: list[dict]) -> None:
    for p in pools:
        conn.execute("""
            INSERT INTO pools (id, name, description, intent, siglip_description, source_clusters, match_count, top_matches, is_noise)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, description=excluded.description, intent=excluded.intent,
                siglip_description=excluded.siglip_description, source_clusters=excluded.source_clusters,
                match_count=excluded.match_count, top_matches=excluded.top_matches, is_noise=excluded.is_noise
        """, (
            p["id"], p["name"], p["description"], p.get("intent", "unknown"),
            p.get("siglip_description", ""), json.dumps(p.get("source_clusters", [])),
            p.get("match_count", 0), json.dumps(p.get("top_matches", [])),
            1 if p.get("is_noise", False) else 0
        ))


def get_pools(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM pools").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["source_clusters"] = json.loads(d["source_clusters"]) if d["source_clusters"] else []
        d["top_matches"] = json.loads(d["top_matches"]) if d["top_matches"] else []
        d["is_noise"] = bool(d["is_noise"])
        result.append(d)
    return result


# --- Temporals & Actions ---

def save_temporal(conn: sqlite3.Connection, **kwargs) -> None:
    conn.execute("""
        INSERT INTO temporals (pool_id, first_date, last_date, span_days, total_count, frequency_per_month, burst_count, longest_gap_days, loop_status)
        VALUES (:pool_id, :first_date, :last_date, :span_days, :total_count, :frequency_per_month, :burst_count, :longest_gap_days, :loop_status)
        ON CONFLICT(pool_id) DO UPDATE SET
            first_date=excluded.first_date, last_date=excluded.last_date, span_days=excluded.span_days,
            total_count=excluded.total_count, frequency_per_month=excluded.frequency_per_month,
            burst_count=excluded.burst_count, longest_gap_days=excluded.longest_gap_days, loop_status=excluded.loop_status
    """, kwargs)


def save_action(conn: sqlite3.Connection, **kwargs) -> None:
    conn.execute("""
        INSERT INTO actions (pool_id, action, why, notes, has_action)
        VALUES (:pool_id, :action, :why, :notes, :has_action)
        ON CONFLICT(pool_id) DO UPDATE SET
            action=excluded.action, why=excluded.why, notes=excluded.notes, has_action=excluded.has_action
    """, kwargs)


def update_pool_counts(conn: sqlite3.Connection, top_k: int = 10) -> None:
    """Update match_count and top_matches for all pools based on classifications."""
    pools = get_pools(conn)
    for pool in pools:
        pid = pool["id"]
        count_row = conn.execute(
            "SELECT COUNT(*) FROM classifications WHERE pool_id = ?", (pid,)
        ).fetchone()
        match_count = count_row[0] if count_row else 0

        top_rows = conn.execute(
            "SELECT filepath FROM classifications WHERE pool_id = ? ORDER BY confidence DESC LIMIT ?",
            (pid, top_k)
        ).fetchall()
        top_matches = json.dumps([r[0] for r in top_rows])

        conn.execute(
            "UPDATE pools SET match_count = ?, top_matches = ? WHERE id = ?",
            (match_count, top_matches, pid)
        )
    conn.commit()


def get_temporals(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM temporals").fetchall()]


def get_actions(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM actions").fetchall()]


# --- Pipeline state ---

def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO pipeline_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value)
    )
    conn.commit()


def get_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def clear_all(conn: sqlite3.Connection) -> None:
    """Clear all cached data (--no-cache)."""
    for table in ["images", "embeddings", "classifications", "pools", "temporals", "actions", "pipeline_state"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
