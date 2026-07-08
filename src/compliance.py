"""
Compliance policy: turn raw detections into a checklist, a score, and an
access decision. Pure functions, no I/O, so this is trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .detector import Detection


@dataclass(frozen=True)
class ChecklistItem:
    name: str
    present: bool
    required: bool
    confidence: float | None  # best confidence among satisfying detections


@dataclass(frozen=True)
class ComplianceResult:
    checklist: list[ChecklistItem]
    compliance_pct: float
    access_granted: bool
    missing: list[str]
    detected_labels: list[str]


def evaluate(detections: list[Detection]) -> ComplianceResult:
    """Apply the PPE policy in config to a set of detections."""
    # Best confidence seen per raw class.
    best_conf: dict[str, float] = {}
    for det in detections:
        best_conf[det.label] = max(best_conf.get(det.label, 0.0), det.confidence)

    checklist: list[ChecklistItem] = []
    for name, spec in config.PPE_REQUIREMENTS.items():
        satisfying = [c for c in spec["satisfied_by"] if c in best_conf]
        present = len(satisfying) > 0
        conf = max((best_conf[c] for c in satisfying), default=None)
        checklist.append(
            ChecklistItem(
                name=name,
                present=present,
                required=spec["required"],
                confidence=conf,
            )
        )

    required_items = [i for i in checklist if i.required]
    total = len(required_items)
    detected = sum(1 for i in required_items if i.present)
    compliance_pct = (detected / total * 100.0) if total else 100.0

    missing = [i.name for i in required_items if not i.present]
    access_granted = len(missing) == 0

    return ComplianceResult(
        checklist=checklist,
        compliance_pct=round(compliance_pct, 1),
        access_granted=access_granted,
        missing=missing,
        detected_labels=sorted(best_conf.keys()),
    )
