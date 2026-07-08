"""
PPE detection: a thin, well-behaved wrapper around an Ultralytics YOLOv8 model.

The rest of the app only ever sees plain Python dicts (`Detection`), never the
Ultralytics `Results` object, which keeps the UI and compliance logic decoupled
from the model library.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Detection:
    """A single detected PPE item."""
    label: str                       # raw CPPE-5 class name, e.g. "Gloves"
    confidence: float                # 0..1
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels


class PPEDetector:
    """Loads a YOLOv8 model once and runs inference on RGB numpy images."""

    def __init__(self, weights: str | Path | None = None) -> None:
        # Imported lazily so `import detector` is cheap and so a missing
        # ultralytics install produces a clear error at construction time.
        from ultralytics import YOLO

        if weights is None:
            weights = (
                config.TRAINED_WEIGHTS
                if config.TRAINED_WEIGHTS.exists()
                else config.FALLBACK_WEIGHTS
            )
        self.weights = str(weights)
        self.using_fallback = self.weights == config.FALLBACK_WEIGHTS
        self.model = YOLO(self.weights)
        # {index: name}; comes from the trained model itself, so it is always
        # correct even if class ordering ever changes.
        self.names: dict[int, str] = self.model.names

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Run inference on an RGB image and return filtered detections."""
        results = self.model.predict(
            source=image_rgb,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for xyxy, conf, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            detections.append(
                Detection(
                    label=self.names[int(cls)],
                    confidence=float(conf),
                    box=(x1, y1, x2, y2),
                )
            )
        return detections

    @staticmethod
    def draw(image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Return a copy of the image with labelled bounding boxes drawn."""
        canvas = image_rgb.copy()
        h = canvas.shape[0]
        thickness = max(2, h // 300)
        font_scale = max(0.5, h / 1000)

        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(
                canvas, (x1, y1), (x2, y2), config.COLOR_BOX_BGR[::-1], thickness
            )
            caption = f"{det.label} {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            top = max(0, y1 - th - baseline - 4)
            cv2.rectangle(
                canvas,
                (x1, top),
                (x1 + tw + 4, top + th + baseline + 4),
                config.COLOR_BOX_BGR[::-1],
                -1,
            )
            cv2.putText(
                canvas,
                caption,
                (x1 + 2, top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        return canvas
