"""Comprehensive unit tests for pool-cli core logic.

Tests cover: temporal analysis, OCR scoring, discovery helpers,
cache round-trips, scanner helpers, and model utilities.

No heavy dependencies (torch, open_clip, anthropic, google-generativeai, ocrmac)
are imported or exercised.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pool import cache
from pool.models import (
    ClassificationBreakdown,
    ClassificationMethod,
    CostEstimate,
    IntentType,
    LoopStatus,
    PipelineMode,
)
from pool.phases.analyzer import (
    _build_temporal,
    _classify_loop,
    _count_bursts,
    _longest_gap,
)
from pool.phases.classifier import (
    _build_keyword_map,
    _find_noise_pool,
    _score_ocr_for_pools,
)
from pool.phases.discovery import (
    _date_range_str,
    _normalize_intent,
    _parse_date,
    _parse_json_response,
    _pools_from_response,
    _sample_spread,
)

from tests.conftest import make_dates, make_temporal


# =========================================================================
# 1. Temporal classification
# =========================================================================


class TestClassifyLoop:
    """Tests for _classify_loop."""

    def test_unknown_when_fewer_than_3_images(self):
        temporal = make_temporal(total_count=2, span_days=5)
        dates = make_dates(datetime(2025, 1, 1, tzinfo=timezone.utc), [0, 1])
        assert _classify_loop(temporal, dates) == LoopStatus.UNKNOWN

    def test_active_when_recent(self, now_utc):
        """Last screenshot within 30 days -> ACTIVE."""
        last = now_utc - timedelta(days=5)
        first = last - timedelta(days=60)
        dates = make_dates(first, [0, 20, 40, 55, 60])
        temporal = make_temporal(
            total_count=5,
            span_days=60,
            longest_gap_days=20,
            first_date=first.isoformat(),
            last_date=last.isoformat(),
        )
        assert _classify_loop(temporal, dates) == LoopStatus.ACTIVE

    def test_active_consistent_3plus_months(self, now_utc):
        """Span >= 3 months, no gap > 60 days, even if last > 30 days ago."""
        last = now_utc - timedelta(days=45)
        first = last - timedelta(days=120)
        dates = make_dates(first, [0, 30, 60, 90, 120])
        temporal = make_temporal(
            total_count=5,
            span_days=120,
            longest_gap_days=30,
            first_date=first.isoformat(),
            last_date=last.isoformat(),
        )
        assert _classify_loop(temporal, dates) == LoopStatus.ACTIVE

    def test_abandoned_90_plus_days_silence(self, now_utc):
        """10+ images, 90+ days since last -> ABANDONED."""
        last = now_utc - timedelta(days=100)
        first = last - timedelta(days=200)
        offsets = list(range(0, 200, 20))  # 10 images
        dates = make_dates(first, offsets)
        temporal = make_temporal(
            total_count=len(dates),
            span_days=200,
            longest_gap_days=20,
            first_date=first.isoformat(),
            last_date=last.isoformat(),
        )
        assert _classify_loop(temporal, dates) == LoopStatus.ABANDONED

    def test_cyclical_2_bursts_with_gaps(self, now_utc):
        """2+ bursts separated by 30+ day gap -> CYCLICAL."""
        last = now_utc - timedelta(days=5)
        first = last - timedelta(days=100)
        # Burst 1: days 0-2 (5 images within 72 hours)
        # Gap of 50 days
        # Burst 2: days 52-54 (5 images within 72 hours)
        offsets = [0, 0, 1, 1, 2, 52, 52, 53, 53, 54, 100]
        dates = make_dates(first, offsets)
        temporal = make_temporal(
            total_count=len(dates),
            span_days=100,
            burst_count=2,
            longest_gap_days=50,
            first_date=first.isoformat(),
            last_date=last.isoformat(),
        )
        assert _classify_loop(temporal, dates) == LoopStatus.CYCLICAL

    def test_one_shot_all_within_14_days_ended(self, now_utc):
        """All within 14 days, ended 30+ days ago -> ONE_SHOT."""
        last = now_utc - timedelta(days=60)
        first = last - timedelta(days=10)
        dates = make_dates(first, [0, 3, 5, 7, 10])
        temporal = make_temporal(
            total_count=5,
            span_days=10,
            longest_gap_days=3,
            first_date=first.isoformat(),
            last_date=last.isoformat(),
        )
        assert _classify_loop(temporal, dates) == LoopStatus.ONE_SHOT

    def test_boundary_exactly_30_days_since_last_is_active(self, now_utc):
        """Exactly 30 days since last screenshot -> ACTIVE (<=30)."""
        last = now_utc - timedelta(days=30)
        first = last - timedelta(days=60)
        dates = make_dates(first, [0, 20, 40, 60])
        temporal = make_temporal(
            total_count=4,
            span_days=60,
            longest_gap_days=20,
            first_date=first.isoformat(),
            last_date=last.isoformat(),
        )
        assert _classify_loop(temporal, dates) == LoopStatus.ACTIVE

    def test_boundary_exactly_90_days_silence_with_10_images_is_abandoned(self, now_utc):
        """Exactly 90 days silence with 10+ images -> ABANDONED."""
        last = now_utc - timedelta(days=90)
        first = last - timedelta(days=180)
        offsets = list(range(0, 181, 18))  # ~10+ images
        dates = make_dates(first, offsets)
        temporal = make_temporal(
            total_count=len(dates),
            span_days=180,
            longest_gap_days=18,
            first_date=first.isoformat(),
            last_date=last.isoformat(),
        )
        assert _classify_loop(temporal, dates) == LoopStatus.ABANDONED


class TestCountBursts:
    """Tests for _count_bursts."""

    def test_no_bursts_when_fewer_than_5_images(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = make_dates(base, [0, 1, 2, 3])
        assert _count_bursts(dates) == 0

    def test_single_burst_5_within_72_hours(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # 5 images within 2 days (48 hours < 72 hours)
        dates = make_dates(base, [0, 0, 1, 1, 2])
        assert _count_bursts(dates) == 1

    def test_multiple_bursts_separated_by_gap(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # Burst 1: days 0-2 (5 images), Burst 2: days 30-32 (5 images)
        offsets = [0, 0, 1, 1, 2, 30, 30, 31, 31, 32]
        dates = make_dates(base, offsets)
        assert _count_bursts(dates) == 2

    def test_exactly_at_threshold_5_within_72_hours(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # 5 images spread exactly across 3 days (72 hours boundary)
        dates = [base + timedelta(hours=h) for h in [0, 18, 36, 54, 72]]
        dates.sort()
        assert _count_bursts(dates) == 1


class TestLongestGap:
    """Tests for _longest_gap."""

    def test_single_date_returns_zero(self):
        dates = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        assert _longest_gap(dates) == 0

    def test_multiple_dates_finds_largest_gap(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = make_dates(base, [0, 5, 6, 50, 55])
        assert _longest_gap(dates) == 44  # gap between day 6 and day 50

    def test_uniform_spacing(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = make_dates(base, [0, 10, 20, 30])
        assert _longest_gap(dates) == 10


# =========================================================================
# 2. OCR scoring
# =========================================================================


class TestScoreOcrForPools:
    """Tests for _score_ocr_for_pools."""

    def test_matching_keywords(self):
        pool_keywords = {"music_pool": ["spotify", "playlist", "album"]}
        scores = _score_ocr_for_pools("Open Spotify and check your playlist", pool_keywords)
        assert len(scores) == 1
        assert scores[0][0] == "music_pool"
        assert scores[0][1] == 2  # "spotify" and "playlist"

    def test_no_matches_returns_empty(self):
        pool_keywords = {"music_pool": ["spotify", "playlist"]}
        scores = _score_ocr_for_pools("Hello world, nothing to see here", pool_keywords)
        assert scores == []

    def test_ranking_most_matches_first(self):
        pool_keywords = {
            "music": ["spotify", "playlist", "album"],
            "shopping": ["cart", "price"],
        }
        text = "spotify playlist album cart"
        scores = _score_ocr_for_pools(text, pool_keywords)
        assert scores[0][0] == "music"
        assert scores[0][1] == 3
        assert scores[1][0] == "shopping"
        assert scores[1][1] == 1


class TestBuildKeywordMap:
    """Tests for _build_keyword_map."""

    def test_static_and_dynamic_keywords(self):
        pools = [{
            "id": "music_pool",
            "name": "Songs & Playlists",
            "description": "Music from spotify and streaming apps",
            "siglip_description": "screenshot of music streaming interface",
            "is_noise": False,
        }]
        kw_map = _build_keyword_map(pools)
        assert "music_pool" in kw_map
        keywords = kw_map["music_pool"]
        # Should include dynamic words from name like "songs", "playlists"
        assert "songs" in keywords
        # Should include static category keywords because "music" matches
        assert "spotify" in keywords

    def test_skips_noise_pools(self):
        pools = [
            {"id": "noise", "name": "Junk Drawer", "description": "misc", "siglip_description": "", "is_noise": True},
            {"id": "music", "name": "Music Pool", "description": "songs", "siglip_description": "", "is_noise": False},
        ]
        kw_map = _build_keyword_map(pools)
        assert "noise" not in kw_map
        assert "music" in kw_map


class TestFindNoisePool:
    """Tests for _find_noise_pool."""

    def test_finds_noise_pool(self):
        pools = [
            {"id": "music", "is_noise": False},
            {"id": "noise", "is_noise": True},
        ]
        assert _find_noise_pool(pools) == "noise"

    def test_returns_none_when_no_noise(self):
        pools = [{"id": "music", "is_noise": False}]
        assert _find_noise_pool(pools) is None


# =========================================================================
# 3. Discovery helpers
# =========================================================================


class TestParseDate:
    """Tests for _parse_date."""

    def test_iso_format(self):
        result = _parse_date("2025-06-15T10:30:00")
        assert result == datetime(2025, 6, 15, 10, 30, 0)

    def test_timezone_aware_format(self):
        result = _parse_date("2025-06-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6

    def test_plain_date(self):
        result = _parse_date("2025-06-15")
        assert result == datetime(2025, 6, 15)

    def test_invalid_string_returns_none(self):
        assert _parse_date("not-a-date") is None

    def test_none_input_returns_none(self):
        assert _parse_date(None) is None


class TestDateRangeStr:
    """Tests for _date_range_str."""

    def test_valid_date_range(self):
        dates = [datetime(2024, 1, 15), datetime(2024, 3, 20)]
        result = _date_range_str(dates)
        assert "Jan 2024" in result
        assert "Mar 2024" in result

    def test_single_date(self):
        dates = [datetime(2024, 6, 1)]
        result = _date_range_str(dates)
        assert result == "Jun 2024"

    def test_empty_list(self):
        assert _date_range_str([]) == "unknown dates"

    def test_none_values_filtered(self):
        dates = [None, datetime(2024, 5, 1), None]
        result = _date_range_str(dates)
        assert result == "May 2024"


class TestParseJsonResponse:
    """Tests for _parse_json_response."""

    def test_clean_json(self):
        data = _parse_json_response('{"pools": []}')
        assert data == {"pools": []}

    def test_markdown_fenced_json(self):
        text = '```json\n{"pools": [{"id": "music"}]}\n```'
        data = _parse_json_response(text)
        assert data["pools"][0]["id"] == "music"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("this is not json at all")


class TestNormalizeIntent:
    """Tests for _normalize_intent."""

    def test_valid_intent(self):
        assert _normalize_intent("aspirational") == "aspirational"
        assert _normalize_intent("functional") == "functional"

    def test_case_insensitive(self):
        assert _normalize_intent("  Aspirational  ") == "aspirational"

    def test_invalid_returns_unknown(self):
        assert _normalize_intent("bogus") == "unknown"


class TestPoolsFromResponse:
    """Tests for _pools_from_response."""

    def test_builds_pools_correctly(self):
        data = {
            "pools": [
                {
                    "id": "music",
                    "name": "My Music",
                    "description": "Songs I saved",
                    "intent": "aspirational",
                    "siglip_description": "screenshot of music app",
                    "source_clusters": [1, 2],
                }
            ],
            "noise_pool": {
                "name": "Junk Drawer",
                "description": "Random stuff",
            },
        }
        pools = _pools_from_response(data)
        assert len(pools) == 2

        music = pools[0]
        assert music["id"] == "music"
        assert music["name"] == "My Music"
        assert music["intent"] == "aspirational"
        assert music["is_noise"] is False

        noise = pools[1]
        assert noise["id"] == "noise"
        assert noise["is_noise"] is True
        assert noise["name"] == "Junk Drawer"


class TestSampleSpread:
    """Tests for _sample_spread."""

    def test_returns_all_when_n_exceeds_items(self):
        items = [{"created_at": f"2025-01-0{i}"} for i in range(1, 4)]
        result = _sample_spread(items, 10)
        assert len(result) == 3

    def test_samples_correct_count(self):
        items = [{"created_at": f"2025-01-{i:02d}"} for i in range(1, 21)]
        result = _sample_spread(items, 5)
        assert len(result) == 5


# =========================================================================
# 4. Cache round-trips
# =========================================================================


class TestCacheImages:
    """Tests for image upsert/get round-trips."""

    def test_upsert_and_get_image(self, conn):
        cache.upsert_image(
            conn,
            filepath="/tmp/test.png",
            filename="test.png",
            file_size=1024,
            width=800,
            height=600,
            created_at="2025-01-01T00:00:00+00:00",
            modified_at="2025-01-02T00:00:00+00:00",
            phash="abcdef1234567890",
            is_duplicate=0,
            duplicate_of=None,
            thumbnail_path="/tmp/thumb.jpg",
        )
        conn.commit()

        img = cache.get_cached_image(conn, "/tmp/test.png")
        assert img is not None
        assert img["filename"] == "test.png"
        assert img["file_size"] == 1024
        assert img["width"] == 800
        assert img["phash"] == "abcdef1234567890"


class TestCacheEmbeddings:
    """Tests for embedding upsert/get round-trips (numpy)."""

    def test_upsert_and_get_embedding(self, conn):
        # Insert an image first (foreign key target)
        cache.upsert_image(
            conn, filepath="/tmp/emb.png", filename="emb.png",
            file_size=100, width=10, height=10, created_at=None,
            modified_at=None, phash=None, is_duplicate=0,
            duplicate_of=None, thumbnail_path=None,
        )

        original = np.random.rand(512).astype(np.float32)
        cache.upsert_embedding(conn, "/tmp/emb.png", original, cluster_id=3)
        conn.commit()

        recovered = cache.get_cached_embedding(conn, "/tmp/emb.png")
        assert recovered is not None
        np.testing.assert_allclose(recovered, original, atol=1e-7)


class TestCacheClassifications:
    """Tests for classification upsert and pool-based retrieval."""

    def test_upsert_and_get_by_pool(self, conn):
        cache.upsert_classification(
            conn,
            filepath="/tmp/cls.png",
            pool_id="music",
            confidence=0.85,
            method="siglip",
            explanation="SigLIP match",
            ocr_text=None,
        )
        conn.commit()

        results = cache.get_classifications_by_pool(conn, "music")
        assert len(results) == 1
        assert results[0]["filepath"] == "/tmp/cls.png"
        assert results[0]["confidence"] == 0.85


class TestCachePools:
    """Tests for pool save/get with JSON fields."""

    def test_save_and_get_pools(self, conn):
        pools = [
            {
                "id": "music",
                "name": "My Music",
                "description": "Songs",
                "intent": "aspirational",
                "siglip_description": "screenshot of music app",
                "source_clusters": [1, 2, 3],
                "match_count": 42,
                "top_matches": ["/tmp/a.png", "/tmp/b.png"],
                "is_noise": False,
            },
            {
                "id": "noise",
                "name": "Junk",
                "description": "Misc",
                "intent": "unknown",
                "siglip_description": "",
                "source_clusters": [],
                "match_count": 0,
                "top_matches": [],
                "is_noise": True,
            },
        ]
        cache.save_pools(conn, pools)
        conn.commit()

        retrieved = cache.get_pools(conn)
        assert len(retrieved) == 2

        music = next(p for p in retrieved if p["id"] == "music")
        assert music["source_clusters"] == [1, 2, 3]
        assert music["top_matches"] == ["/tmp/a.png", "/tmp/b.png"]
        assert music["is_noise"] is False

        noise = next(p for p in retrieved if p["id"] == "noise")
        assert noise["is_noise"] is True


class TestCachePipelineState:
    """Tests for pipeline state set/get."""

    def test_set_and_get_state(self, conn):
        cache.set_state(conn, "phase", "discovery")
        assert cache.get_state(conn, "phase") == "discovery"

    def test_get_nonexistent_key_returns_none(self, conn):
        assert cache.get_state(conn, "nonexistent") is None

    def test_overwrite_state(self, conn):
        cache.set_state(conn, "step", "1")
        cache.set_state(conn, "step", "2")
        assert cache.get_state(conn, "step") == "2"


# =========================================================================
# 5. Scanner helpers
# =========================================================================


class TestCollectImagePaths:
    """Tests for _collect_image_paths (via scanner.IMAGE_EXTENSIONS)."""

    def test_skips_hidden_dirs_and_non_image_files(self, tmp_path):
        """Create a directory tree and verify hidden dirs and non-images are skipped."""
        from pool.phases.scanner import _collect_image_paths

        # Visible image
        (tmp_path / "photo.png").write_bytes(b"\x89PNG")
        # Hidden directory with an image inside
        hidden = tmp_path / ".hidden_dir"
        hidden.mkdir()
        (hidden / "secret.png").write_bytes(b"\x89PNG")
        # Non-image file
        (tmp_path / "notes.txt").write_text("hello")
        # Hidden file
        (tmp_path / ".DS_Store").write_bytes(b"\x00")
        # Thumbnail dir
        thumb = tmp_path / ".pool_thumbnails"
        thumb.mkdir()
        (thumb / "thumb.jpg").write_bytes(b"\xff\xd8")

        paths = _collect_image_paths(tmp_path)
        names = [p.name for p in paths]
        assert "photo.png" in names
        assert "secret.png" not in names
        assert "notes.txt" not in names
        assert ".DS_Store" not in names
        assert "thumb.jpg" not in names

    def test_image_extension_matching(self, tmp_path):
        """All supported extensions should be collected."""
        from pool.phases.scanner import _collect_image_paths, IMAGE_EXTENSIONS

        for ext in IMAGE_EXTENSIONS:
            (tmp_path / f"test{ext}").write_bytes(b"\x00")

        paths = _collect_image_paths(tmp_path)
        assert len(paths) == len(IMAGE_EXTENSIONS)

    def test_nested_directories(self, tmp_path):
        """Images in subdirectories should be found."""
        from pool.phases.scanner import _collect_image_paths

        sub = tmp_path / "subdir" / "deeper"
        sub.mkdir(parents=True)
        (sub / "nested.jpg").write_bytes(b"\xff\xd8")

        paths = _collect_image_paths(tmp_path)
        names = [p.name for p in paths]
        assert "nested.jpg" in names


# =========================================================================
# 6. Model utilities
# =========================================================================


class TestClassificationBreakdownSummaryLines:
    """Tests for ClassificationBreakdown.summary_lines."""

    def test_percentages(self):
        bd = ClassificationBreakdown(
            siglip_high=50,
            ocr_confirmed=30,
            gemini_resolved=10,
            unresolved=10,
            total_unique=100,
        )
        lines = bd.summary_lines()
        labels = [l[0] for l in lines]
        assert "SigLIP zero-shot (high confidence)" in labels

        siglip_line = next(l for l in lines if "SigLIP zero-shot" in l[0])
        assert siglip_line[1] == 50
        assert abs(siglip_line[2] - 0.5) < 1e-9

    def test_zero_total_no_division_error(self):
        bd = ClassificationBreakdown(
            siglip_high=5,
            total_unique=0,
        )
        lines = bd.summary_lines()
        # Should use max(total_unique, 1) to avoid division by zero
        assert len(lines) == 1
        assert lines[0][1] == 5

    def test_empty_breakdown_returns_empty(self):
        bd = ClassificationBreakdown()
        assert bd.summary_lines() == []


class TestCostEstimate:
    """Tests for CostEstimate.display."""

    def test_display_formatting(self):
        cost = CostEstimate(anthropic_usd=1.50, google_usd=0.25)
        text = cost.display()
        assert "~$1.75" in text
        assert "Anthropic ~$1.50" in text
        assert "Google ~$0.25" in text

    def test_total_usd(self):
        cost = CostEstimate(anthropic_usd=0.10, google_usd=0.05)
        assert abs(cost.total_usd - 0.15) < 1e-9


class TestEnumValues:
    """Tests for enum value consistency."""

    def test_loop_status_values(self):
        assert LoopStatus.ACTIVE.value == "active"
        assert LoopStatus.CYCLICAL.value == "cyclical"
        assert LoopStatus.ABANDONED.value == "abandoned"
        assert LoopStatus.ONE_SHOT.value == "one_shot"
        assert LoopStatus.UNKNOWN.value == "unknown"

    def test_intent_type_values(self):
        assert IntentType.ASPIRATIONAL.value == "aspirational"
        assert IntentType.FUNCTIONAL.value == "functional"
        assert IntentType.EMOTIONAL.value == "emotional"
        assert IntentType.SOCIAL.value == "social"
        assert IntentType.INVESTIGATIVE.value == "investigative"
        assert IntentType.CREATIVE.value == "creative"
        assert IntentType.UNKNOWN.value == "unknown"

    def test_classification_method_values(self):
        assert ClassificationMethod.SIGLIP.value == "siglip"
        assert ClassificationMethod.OCR.value == "ocr"
        assert ClassificationMethod.GEMINI.value == "gemini"
        assert ClassificationMethod.CLAUDE_DIRECT.value == "claude"
        assert ClassificationMethod.NOISE.value == "noise"


class TestBuildTemporal:
    """Tests for _build_temporal."""

    def test_empty_dates(self):
        result = _build_temporal("pool_x", [])
        assert result["total_count"] == 0
        assert result["span_days"] == 0
        assert result["first_date"] is None

    def test_with_dates(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dates = make_dates(base, [0, 10, 20, 30])
        result = _build_temporal("pool_y", dates)
        assert result["pool_id"] == "pool_y"
        assert result["total_count"] == 4
        assert result["span_days"] == 30
        assert result["first_date"] == dates[0].isoformat()
        assert result["last_date"] == dates[-1].isoformat()
        assert result["frequency_per_month"] == 4.0  # 4 images / (30/30 = 1 month)
