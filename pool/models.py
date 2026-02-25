"""Shared data models for Pool CLI."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class PipelineMode(str, Enum):
    FULL = "full"       # >=200 unique images, complete pipeline
    SMALL = "small"     # <200 unique images, skip HDBSCAN
    QUICK = "quick"     # --quick flag, sample → Claude → done


class LoopStatus(str, Enum):
    ACTIVE = "active"           # last screenshot within 30 days or consistent 3+ months
    CYCLICAL = "cyclical"       # 2+ distinct bursts with rhythm
    ABANDONED = "abandoned"     # clear activity followed by 90+ days silence
    ONE_SHOT = "one_shot"       # all within 2 weeks, never revisited
    UNKNOWN = "unknown"


class IntentType(str, Enum):
    ASPIRATIONAL = "aspirational"       # "I want this"
    FUNCTIONAL = "functional"           # "I need this later"
    EMOTIONAL = "emotional"             # "I want to remember this"
    SOCIAL = "social"                   # "I want to share this"
    INVESTIGATIVE = "investigative"     # "I'm researching"
    CREATIVE = "creative"              # "I'm stealing/referencing"
    UNKNOWN = "unknown"


class ClassificationMethod(str, Enum):
    SIGLIP = "siglip"           # Tier 1: zero-shot embedding match
    OCR = "ocr"                 # Tier 2: OCR keyword confirmation/override
    GEMINI = "gemini"           # Tier 3: Gemini Flash tiebreaker
    CLAUDE_DIRECT = "claude"    # Small/quick mode: Claude classified directly
    NOISE = "noise"             # Unresolvable


# --- Image models ---

class ImageMeta(BaseModel):
    filepath: str
    filename: str
    file_size: int                          # bytes
    width: int
    height: int
    created_at: Optional[str] = None        # ISO datetime
    modified_at: Optional[str] = None       # ISO datetime
    phash: Optional[str] = None             # perceptual hash hex string
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None      # filepath of representative
    thumbnail_path: Optional[str] = None


class ImageEmbedding(BaseModel):
    filepath: str
    embedding: Optional[list[float]] = None  # 512D SigLIP vector (stored as blob in SQLite)
    cluster_id: Optional[int] = None


class ImageClassification(BaseModel):
    filepath: str
    pool_id: str
    confidence: float = 0.0
    method: ClassificationMethod = ClassificationMethod.NOISE
    explanation: Optional[str] = None
    ocr_text: Optional[str] = None


# --- Pool models ---

class Pool(BaseModel):
    id: str
    name: str
    description: str
    intent: IntentType = IntentType.UNKNOWN
    siglip_description: Optional[str] = None    # optimized text for zero-shot
    source_clusters: list[int] = Field(default_factory=list)
    match_count: int = 0
    top_matches: list[str] = Field(default_factory=list)   # filepaths
    is_noise: bool = False


class TemporalSignature(BaseModel):
    pool_id: str
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    span_days: int = 0
    total_count: int = 0
    frequency_per_month: float = 0.0
    burst_count: int = 0           # 5+ screenshots in 72hrs
    longest_gap_days: int = 0
    loop_status: LoopStatus = LoopStatus.UNKNOWN


class PoolAction(BaseModel):
    pool_id: str
    action: Optional[str] = None       # None = deliberate no-action
    why: Optional[str] = None          # one-liner explanation
    notes: str = ""                    # temporal/behavioral observation
    has_action: bool = False


# --- Pipeline state ---

class ClassificationBreakdown(BaseModel):
    siglip_high: int = 0        # Tier 1 high confidence
    ocr_confirmed: int = 0      # Tier 2 OCR
    gemini_resolved: int = 0    # Tier 3 Gemini
    claude_direct: int = 0      # Quick/small mode
    unresolved: int = 0         # noise
    duplicates_skipped: int = 0
    total_unique: int = 0

    def summary_lines(self) -> list[tuple[str, int, float]]:
        """Return (label, count, percentage) tuples for display."""
        lines = []
        if self.siglip_high > 0:
            lines.append(("SigLIP zero-shot (high confidence)", self.siglip_high, self.siglip_high / max(self.total_unique, 1)))
        if self.ocr_confirmed > 0:
            lines.append(("SigLIP + OCR confirmed", self.ocr_confirmed, self.ocr_confirmed / max(self.total_unique, 1)))
        if self.gemini_resolved > 0:
            lines.append(("Gemini Flash tiebreaker", self.gemini_resolved, self.gemini_resolved / max(self.total_unique, 1)))
        if self.claude_direct > 0:
            lines.append(("Claude direct classification", self.claude_direct, self.claude_direct / max(self.total_unique, 1)))
        if self.unresolved > 0:
            lines.append(('Unresolved → "Not Sure" pool', self.unresolved, self.unresolved / max(self.total_unique, 1)))
        if self.duplicates_skipped > 0:
            lines.append(("Duplicates skipped", self.duplicates_skipped, 0.0))
        return lines


class CostEstimate(BaseModel):
    anthropic_usd: float = 0.0
    google_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.anthropic_usd + self.google_usd

    def display(self) -> str:
        return f"~${self.total_usd:.2f} (Anthropic ~${self.anthropic_usd:.2f} · Google ~${self.google_usd:.2f})"


class PipelineState(BaseModel):
    mode: PipelineMode = PipelineMode.FULL
    source_dir: str = ""
    total_images: int = 0
    unique_images: int = 0
    duplicates: int = 0
    pools: list[Pool] = Field(default_factory=list)
    classifications: list[ImageClassification] = Field(default_factory=list)
    temporals: list[TemporalSignature] = Field(default_factory=list)
    actions: list[PoolAction] = Field(default_factory=list)
    breakdown: ClassificationBreakdown = Field(default_factory=ClassificationBreakdown)
    cost: CostEstimate = Field(default_factory=CostEstimate)
    opening_insight: str = ""
