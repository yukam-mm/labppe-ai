"""
Evaluate the trained LabPPE detector.

Reports two things that both matter for a lab-entry gate:

1. **Detection quality** — mAP50 and mAP50-95 on the CPPE-5 validation split.
   These are the standard COCO-style metrics. mAP50 is the more forgiving
   (IoU ≥ 0.5); mAP50-95 averages over IoU thresholds 0.5..0.95.

2. **CPU latency** — median / p95 / mean inference time per image and the
   resulting throughput. When both the PyTorch .pt and the exported OpenVINO
   model are on disk, both are benchmarked side by side so you see the
   actual speedup on YOUR CPU.

Usage:
    python evaluate.py                      # full eval (mAP + latency, both formats)
    python evaluate.py --skip-map           # latency only (fast, ~1 min)
    python evaluate.py --iters 50           # more latency samples for stability
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src import config


def benchmark_latency(model, sample_images, warmup: int = 3, iters: int = 20) -> dict:
    """Time `iters` predict() calls after `warmup` untimed calls."""
    for _ in range(warmup):
        model.predict(
            sample_images[0], verbose=False, imgsz=config.INFERENCE_IMG_SIZE
        )

    times_ms: list[float] = []
    for i in range(iters):
        img = sample_images[i % len(sample_images)]
        t0 = time.perf_counter()
        model.predict(img, verbose=False, imgsz=config.INFERENCE_IMG_SIZE)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    return {
        "median_ms": round(statistics.median(times_ms), 2),
        "p95_ms": round(times_ms[max(0, int(len(times_ms) * 0.95) - 1)], 2),
        "mean_ms": round(statistics.mean(times_ms), 2),
        "throughput_fps": round(1000.0 / statistics.median(times_ms), 1),
    }


def eval_map(model, data_yaml: str) -> dict:
    """Run ultralytics' validator and pull out the headline metrics."""
    metrics = model.val(
        data=data_yaml,
        imgsz=config.INFERENCE_IMG_SIZE,
        verbose=False,
        plots=False,
    )
    return {
        "mAP50": round(float(metrics.box.map50), 3),
        "mAP50_95": round(float(metrics.box.map), 3),
    }


def _load_sample_images(n: int) -> list[np.ndarray]:
    val_dir = config.DATA_DIR / "cppe5" / "images" / "val"
    if val_dir.exists():
        paths = sorted(val_dir.glob("*.jpg"))[:n]
        if paths:
            return [np.array(Image.open(p).convert("RGB")) for p in paths]
    print("[warn] No validation images found; latency will use random arrays.")
    return [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(min(n, 5))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(config.DATA_DIR / "cppe5" / "cppe5.yaml"),
        help="dataset YAML from train.py (needed for mAP)",
    )
    parser.add_argument("--iters", type=int, default=20, help="latency iterations")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--skip-map", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_yaml_exists = Path(args.data).exists()
    if not data_yaml_exists and not args.skip_map:
        print(f"[warn] {args.data} missing → skipping mAP evaluation.")
        args.skip_map = True

    samples = _load_sample_images(args.n_samples)
    print(f"\nCPU eval — imgsz={config.INFERENCE_IMG_SIZE}, "
          f"latency n={args.iters}, mAP={'skip' if args.skip_map else 'on'}\n")

    rows: list[dict] = []

    # PyTorch .pt (baseline)
    if config.TRAINED_WEIGHTS.exists():
        print(f"→ PyTorch .pt   ({config.TRAINED_WEIGHTS.name})")
        m = YOLO(str(config.TRAINED_WEIGHTS))
        row = {"format": "PyTorch FP32", **benchmark_latency(m, samples, iters=args.iters)}
        if not args.skip_map:
            row.update(eval_map(m, args.data))
        rows.append(row)

    # CoreML (macOS / Apple Silicon)
    if config.EXPORTED_MODEL_COREML.exists():
        print(f"→ CoreML        ({config.EXPORTED_MODEL_COREML.name})")
        m = YOLO(str(config.EXPORTED_MODEL_COREML))
        row = {"format": "CoreML (ANE)", **benchmark_latency(m, samples, iters=args.iters)}
        if not args.skip_map:
            row.update(eval_map(m, args.data))
        rows.append(row)

    # OpenVINO (Intel CPU)
    if config.EXPORTED_MODEL_OPENVINO.exists():
        print(f"→ OpenVINO      ({config.EXPORTED_MODEL_OPENVINO.name})")
        m = YOLO(str(config.EXPORTED_MODEL_OPENVINO))
        row = {"format": "OpenVINO FP16", **benchmark_latency(m, samples, iters=args.iters)}
        if not args.skip_map:
            row.update(eval_map(m, args.data))
        rows.append(row)

    if not rows:
        print("\nNo trained model on disk. Run `python train.py` first.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    if len(rows) >= 2:
        # rows[0] is always PyTorch when present; compare against the fastest accel.
        base = next((r for r in rows if r["format"] == "PyTorch FP32"), None)
        accel = min(
            (r for r in rows if r["format"] != "PyTorch FP32"),
            key=lambda r: r["median_ms"],
            default=None,
        )
        if base and accel:
            speedup = base["median_ms"] / accel["median_ms"]
            print(f"\n{accel['format']} vs. PyTorch: {speedup:.2f}× (median latency)")
            if not args.skip_map:
                dm = base["mAP50_95"] - accel["mAP50_95"]
                print(f"mAP50-95 delta (PyTorch − {accel['format']}): {dm:+.3f}  "
                      f"(near zero = clean export)")


if __name__ == "__main__":
    main()
