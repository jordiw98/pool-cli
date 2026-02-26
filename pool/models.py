"""Shared data models for Pool CLI."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

CLAUDE_MODEL = "claude-sonnet-4-20250514"

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
