"""
Central configuration for LabPPE AI.

Everything that a user might want to tune lives here so the rest of the
codebase never hard-codes a class name, colour, or threshold.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DATA_DIR: Path = PROJECT_ROOT / "data"

# The weights produced by train.py. Falls back to a pretrained COCO model only
# so the app can boot before you have trained anything (it will detect nothing
# useful, but it will not crash).
TRAINED_WEIGHTS: Path = PROJECT_ROOT/"models" / "runs" / "labppe" / "weights" / "best.pt"
# Platform-optimized exported models. `train.py` picks the right export format
# for the OS; the detector loads whichever one exists. Naming matches the paths
# ultralytics produces automatically, so no post-export renaming is needed.
EXPORTED_MODEL_COREML: Path = MODELS_DIR / "labppe.mlpackage"        # macOS / Apple Silicon
EXPORTED_MODEL_OPENVINO: Path = MODELS_DIR / "labppe_openvino_model" # Intel CPU (Linux/Windows)
# Pretrained checkpoint auto-downloaded when no trained weights exist. yolo11n
# is a solid CPU baseline (universally available; faster than yolov8n via
# CoreML/OpenVINO). Swap for "yolo26n.pt" (Jan 2026) for another ~40% CPU speed
# if your ultralytics version supports it.
FALLBACK_WEIGHTS: str = "yolo11n.pt"

# --------------------------------------------------------------------------- #
# Model / inference
# --------------------------------------------------------------------------- #
# CPPE-5 raw class names, in the exact order the dataset defines them.
CPPE5_CLASSES: list[str] = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]

# Minimum confidence for a detection to count.
CONFIDENCE_THRESHOLD: float = 0.35
IOU_THRESHOLD: float = 0.45
# 416 is the CPU sweet spot: ~35% fewer FLOPs than 640 with negligible mAP loss
# on close-up PPE (large objects). Training and inference must match, so this
# value is used for both.
INFERENCE_IMG_SIZE: int = 416

# --------------------------------------------------------------------------- #
# PPE requirement policy
# --------------------------------------------------------------------------- #
# We map CPPE-5's raw classes to the human-facing PPE requirements of a
# biotech lab. Each requirement is satisfied if ANY of its `satisfied_by`
# classes is detected. This makes the policy easy to extend (e.g. allow a
# face shield to satisfy eye protection) without touching detection code.
#
# Keys are the labels shown on the checklist.
# `spoken` is the lowercase phrase the voice announcer reads out (it gets
# sentence-capitalised at runtime). Defaults to the lowercased key if omitted.
PPE_REQUIREMENTS: dict[str, dict] = {
    "Lab Coat": {"satisfied_by": ["Coverall"], "required": True, "spoken": "lab coat"},
    "Gloves": {"satisfied_by": ["Gloves"], "required": True, "spoken": "gloves"},
    "Goggles": {"satisfied_by": ["Goggles", "Face_Shield"], "required": True, "spoken": "safety goggles"},
    "Face Mask": {"satisfied_by": ["Mask"], "required": True, "spoken": "face mask"},
}

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
# BGR tuples for OpenCV drawing.
COLOR_PRESENT_BGR: tuple[int, int, int] = (60, 180, 75)   # green
COLOR_MISSING_BGR: tuple[int, int, int] = (60, 60, 220)   # red
COLOR_BOX_BGR: tuple[int, int, int] = (255, 170, 0)       # amber for boxes

# Hex equivalents for Streamlit / HTML.
COLOR_PRESENT_HEX: str = "#3CB44B"
COLOR_MISSING_HEX: str = "#DC3C3C"
COLOR_ACCENT_HEX: str = "#00AAFF"


def ensure_dirs() -> None:
    """Create runtime directories if they do not exist."""
    for d in (MODELS_DIR, REPORTS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
