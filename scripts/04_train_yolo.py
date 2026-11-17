"""Fine-tune a pretrained YOLO detector on the beach bird training split.

Example commands
    python scripts/04_train_yolo.py --model yolov8n.pt --name yolov8n_plain --aug plain
    python scripts/04_train_yolo.py --model yolov8n.pt --name yolov8n_aug --aug default
    python scripts/04_train_yolo.py --model yolo11n.pt --name yolo11n_aug --aug default
    python scripts/04_train_yolo.py --model yolov8s.pt --name yolov8s_aug --aug default
    python scripts/04_train_yolo.py --model yolov8n.pt --name yolov8n_aug_960 --aug default --imgsz 960

An input size of 512 matches the original images. Upsampling to 960 gives
small birds more input pixels and costs more time and memory. Lower the batch
size if memory is limited. The plain preset disables image augmentation.
The default preset uses the usual YOLO augmentations. The strong preset
also sets mixup, copy-paste, and rotation. Checkpoint selection uses validation
during training.
"""

# Five checkpoints are trained for seven YOLO comparisons.
# TTA and 2x2 tiled inference both reuse the augmented YOLOv8n best.pt.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow the script to import src when launched from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA_ROOT, METRICS_DIR, MODELS_DIR, ensure_dirs  # noqa: E402


def build_aug_kwargs(level: str) -> dict:
    """Return the Ultralytics options for an augmentation preset.

    Mosaic combines four training images and close_mosaic sets how many final
    epochs use normal layouts. HSV jitter varies lighting and colour. Scale,
    translation and horizontal flips vary the geometry. Mixup blends images
    and copy-paste adds instances.
    """
    if level == "plain":
        return dict(mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
                    degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.0,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0)
    if level == "default":
        # Set the baseline augmentation parameters explicitly for reproducibility.
        return dict(mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
                    degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5,
                    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.0)
    if level == "strong":
        return dict(mosaic=1.0, close_mosaic=10, mixup=0.15, copy_paste=0.1,
                    degrees=10.0, translate=0.1, scale=0.6, shear=2.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5,
                    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, erasing=0.2)
    raise ValueError(f"未知增强等级: {level}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="预训练权重，如 yolov8n.pt / yolov8s.pt / yolo11n.pt")
    ap.add_argument("--name", required=True, help="本次实验名（结果保存到 runs/detect/<name>）")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aug", choices=["plain", "default", "strong"], default="default")
    ap.add_argument("--device", default="0", help="'0' 用第一块 GPU，'cpu' 用 CPU")
    ap.add_argument("--patience", type=int, default=15, help="验证集 mAP 连续 N 轮不升就早停")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()
    from ultralytics import YOLO

    # Write a dataset YAML with an absolute root and relative split paths.
    data_yaml = MODELS_DIR / "beach_bird_local.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {DATA_ROOT.as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\nnames:\n  0: bird\n")

    aug_kwargs = build_aug_kwargs(args.aug)
    print(f"[train] model={args.model} name={args.name} imgsz={args.imgsz} "
          f"epochs={args.epochs} batch={args.batch} aug={args.aug}")

    model = YOLO(args.model)
    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(MODELS_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        plots=True,
        save=True,
        val=True,
        **aug_kwargs,
    )
    train_seconds = time.time() - t0

    # Record the training configuration and time for the results table.
    best = MODELS_DIR / "runs" / args.name / "weights" / "best.pt"
    meta = {
        "name": args.name, "model": args.model, "imgsz": args.imgsz,
        "epochs": args.epochs, "batch": args.batch, "aug": args.aug,
        "train_seconds": train_seconds, "best_weights": str(best),
        "train_minutes": train_seconds / 60.0,
    }
    with open(METRICS_DIR / f"train_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] best weights: {best}")
    print(f"[done] training time: {train_seconds/60:.1f} min")


if __name__ == "__main__":
    main()
