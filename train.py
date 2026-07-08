"""
Train a YOLOv8 PPE detector on the CPPE-5 dataset via transfer learning.

Usage
-----
    python train.py --epochs 60 --model yolov8n.pt --imgsz 640

CPPE-5 ships in COCO JSON format. Ultralytics needs YOLO-txt labels + a data
YAML. This script downloads CPPE-5 from the Hugging Face Hub, converts the
annotations once (cached), writes `data/cppe5/cppe5.yaml`, then fine-tunes a
pretrained YOLOv8 model. The best weights are copied to models/labppe_yolov8.pt
so the Streamlit app picks them up automatically.

Dependencies for training only:
    pip install ultralytics datasets pycocotools tqdm
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src import config

# CPPE-5 category_id (1-based in COCO) -> zero-based YOLO index.
# CPPE-5 order: 1 Coverall, 2 Face_Shield, 3 Gloves, 4 Goggles, 5 Mask
COCO_ID_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CPPE5_DIR = config.DATA_DIR / "cppe5"


def _convert_split(split, images_root: Path, labels_root: Path) -> int:
    """Write YOLO-format label txts + copy images for one HF dataset split."""
    from PIL import Image

    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    count = 0

    for sample in split:
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        img_id = sample["image_id"]
        img_name = f"{img_id:06d}.jpg"
        image.save(images_root / img_name, quality=95)

        # HF CPPE-5 objects: dict of parallel lists.
        objs = sample["objects"]
        lines = []
        for bbox, cat in zip(objs["bbox"], objs["category"]):
            # HF stores category as 0-based already; bbox is [x, y, w, h] in px.
            yolo_cls = int(cat)
            x, y, bw, bh = bbox
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (labels_root / f"{img_id:06d}.txt").write_text("\n".join(lines))
        count += 1
    return count


def prepare_dataset() -> Path:
    """Download + convert CPPE-5. Returns path to the data YAML (cached)."""
    yaml_path = CPPE5_DIR / "cppe5.yaml"
    if yaml_path.exists():
        print(f"[data] Using cached dataset at {CPPE5_DIR}")
        return yaml_path

    from datasets import load_dataset

    print("[data] Downloading CPPE-5 from the Hugging Face Hub ...")
    ds = load_dataset("rishitdagli/cppe-5")

    n_train = _convert_split(
        ds["train"], CPPE5_DIR / "images/train", CPPE5_DIR / "labels/train"
    )
    # CPPE-5 has train/test; carve a val view from test.
    n_val = _convert_split(
        ds["test"], CPPE5_DIR / "images/val", CPPE5_DIR / "labels/val"
    )
    print(f"[data] Converted {n_train} train / {n_val} val images.")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(config.CPPE5_CLASSES))
    yaml_path.write_text(
        f"path: {CPPE5_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names}\n"
    )
    print(f"[data] Wrote {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LabPPE YOLOv8 model.")
    parser.add_argument("--model", default="yolov8n.pt", help="pretrained weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    args = parser.parse_args()

    config.ensure_dirs()
    from ultralytics import YOLO

    data_yaml = prepare_dataset()

    print(f"[train] Fine-tuning {args.model} for {args.epochs} epochs ...")
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(config.MODELS_DIR / "runs"),
        name="labppe",
        patience=15,
        seed=42,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, config.TRAINED_WEIGHTS)
        print(f"[train] Copied best weights -> {config.TRAINED_WEIGHTS}")
    else:
        print("[train] WARNING: best.pt not found; check the run directory.")


if __name__ == "__main__":
    main()
