"""Evidence assessment: grade retrieval as strong / medium / weak.

Only STRONG evidence proceeds to answer generation. MEDIUM and WEAK route to a
safe fallback. The grade uses the top Cohere rerank relevance score against
thresholds loaded from ``config/evidence_thresholds.json``.

The shipped thresholds are conservative placeholders (``calibrated: false``).
``scripts/calibrate_evidence.py`` derives real values from a committed
question set; until then a warning is logged at startup.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.workflow.contracts import EvidenceTier, RetrievedPassage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceThresholds:
    strong: float
    medium: float
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.medium <= self.strong <= 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 <= medium <= strong <= 1 "
                f"(got medium={self.medium}, strong={self.strong})"
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "EvidenceThresholds":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            strong=float(data["strong"]),
            medium=float(data["medium"]),
            calibrated=bool(data.get("calibrated", False)),
        )

    def to_json(self) -> dict:
        return {"strong": self.strong, "medium": self.medium, "calibrated": self.calibrated}


class EvidenceAssessor:
    def __init__(self, thresholds: EvidenceThresholds) -> None:
        self.thresholds = thresholds
        if not thresholds.calibrated:
            logger.warning(
                "Evidence thresholds are uncalibrated placeholders (strong=%.2f, medium=%.2f). "
                "Run scripts/calibrate_evidence.py once the in-scope question set exists.",
                thresholds.strong,
                thresholds.medium,
            )

    @property
    def calibrated(self) -> bool:
        return self.thresholds.calibrated

    def assess(
        self, passages: Sequence[RetrievedPassage], rerank_applied: bool = True
    ) -> EvidenceTier:
        """Grade the evidence.

        No passages -> WEAK. When the Cohere rerank did not run, scores are RRF
        scores on a different scale, so the tier is capped at MEDIUM (fallback).
        """
        if not passages:
            return EvidenceTier.WEAK
        top = max(p.score for p in passages)
        t = self.thresholds
        if top >= t.strong:
            tier = EvidenceTier.STRONG
        elif top >= t.medium:
            tier = EvidenceTier.MEDIUM
        else:
            tier = EvidenceTier.WEAK
        if not rerank_applied:
            # Scores are RRF scores here, not comparable to the rerank thresholds. Report
            # MEDIUM ("retrieved something, could not confirm relevance") so the degraded
            # case is distinguishable from genuinely weak evidence. Both fall back safely.
            return EvidenceTier.MEDIUM
        return tier
