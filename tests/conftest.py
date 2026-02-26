"""Shared fixtures for pool-cli tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from pool import cache


@pytest.fixture
def conn():
    """In-memory SQLite connection with all pool cache tables created."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    cache._ensure_tables(c)
    return c


@pytest.fixture
def now_utc():
    """Current UTC datetime."""
    return datetime.now(tz=timezone.utc)


def make_dates(
    base: datetime,
    offsets_days: list[int],
) -> list[datetime]:
    """Build a sorted list of datetimes from a base and day-offsets."""
    dates = [base + timedelta(days=d) for d in offsets_days]
    dates.sort()
    return dates


def make_temporal(
    pool_id: str = "test",
    total_count: int = 10,
    span_days: int = 60,
    burst_count: int = 0,
    longest_gap_days: int = 10,
    first_date: str | None = "2025-01-01T00:00:00+00:00",
    last_date: str | None = "2025-03-01T00:00:00+00:00",
) -> dict:
    """Build a temporal dict matching the structure used by analyzer."""
    return {
        "pool_id": pool_id,
        "first_date": first_date,
        "last_date": last_date,
        "span_days": span_days,
        "total_count": total_count,
        "frequency_per_month": round(total_count / max(span_days / 30.0, 1.0), 2),
        "burst_count": burst_count,
        "longest_gap_days": longest_gap_days,
    }
