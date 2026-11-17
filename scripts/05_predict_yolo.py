"""Generate prediction JSONs from a trained YOLO checkpoint.

Example commands
    python scripts/05_predict_yolo.py --weights models/runs/yolov8n_aug/weights/best.pt --name yolov8n_aug --split val

Add --tta for multi-scale and flip inference. Use --tiles 2 --overlap 96
--imgsz 640 for overlapping 2x2 tiles. Both settings reuse the same weights.
Use a low confidence cutoff (default 0.001) for AP evaluation, which ranks
detections by confidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (METRICS_DIR, PRED_DIR, ensure_dirs, nms, save_predictions)


def predict_standard(model, image_path, args):
    """Run whole-image inference and return an (N, 5) array."""
    r = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf,
                      iou=args.iou, max_det=args.max_det, device=args.device,
                      augment=args.tta, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 5), np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = r.boxes.conf.cpu().numpy().astype(np.float32)
    return np.concatenate([xyxy, conf[:, None]], axis=1)


def predict_tiled(model, image, args):
    """Detect in overlapping tiles and map the boxes back to image coordinates.

    Upsampling each crop gives small birds more input pixels. Run the detector
    on all n x n tiles and merge their boxes with NMS.
    """
    h, w = image.shape[:2]
    n = args.tiles
    overlap = args.overlap
    tile_w = int(np.ceil(w / n)) + overlap
    tile_h = int(np.ceil(h / n)) + overlap
    tile_w, tile_h = min(tile_w, w), min(tile_h, h)
    xs = np.linspace(0, max(w - tile_w, 0), n).astype(int)
    ys = np.linspace(0, max(h - tile_h, 0), n).astype(int)
    all_boxes, all_scores = [], []
    for y0 in ys:
        for x0 in xs:
            tile = image[y0:y0 + tile_h, x0:x0 + tile_w]
            r = model.predict(source=tile, imgsz=args.imgsz, conf=args.conf,
                              iou=args.iou, max_det=args.max_det, device=args.device,
                              augment=args.tta, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.cpu().numpy().astype(np.float32)
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            all_boxes.append(xyxy)
            all_scores.append(conf)
    if not all_boxes:
        return np.zeros((0, 5), np.float32)
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    keep = nms(boxes, scores, args.merge_iou)
    keep = keep[np.argsort(-scores[keep])][: args.max_det]
    return np.concatenate([boxes[keep], scores[keep, None]], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True, help="实验名，用于输出文件名")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7, help="模型内部 NMS 的 IoU")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tta", action="store_true", help="测试时增强")
    ap.add_argument("--tiles", type=int, default=1, help="切图数 n（n×n），1=不切")
    ap.add_argument("--overlap", type=int, default=96)
    ap.add_argument("--merge-iou", type=float, default=0.5, help="跨 tile 合并的 NMS IoU")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    ensure_dirs()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    image_dir = Path(__file__).resolve().parents[1] / "data" / "Beach Bird Detection Dataset v1.0" / "images" / args.split
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if args.limit:
        image_paths = image_paths[: args.limit]

    preds = {}
    t0 = time.time()
    for i, p in enumerate(image_paths, start=1):
        if args.tiles > 1:
            import cv2
            img = cv2.imread(str(p))
            det = predict_tiled(model, img, args)
        else:
            det = predict_standard(model, p, args)
        preds[p.stem] = det
        if i % 50 == 0:
            print(f"  {i}/{len(image_paths)} images, "
                  f"{sum(len(v) for v in preds.values())} boxes", flush=True)
    elapsed = time.time() - t0

    tag = args.tag or args.name
    meta = {"method": f"yolo_{args.name}", "split": args.split, "weights": args.weights,
            "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "tta": args.tta,
            "tiles": args.tiles, "overlap": args.overlap,
            "inference_seconds": elapsed,
            "ms_per_image": 1000 * elapsed / len(image_paths),
            "num_images": len(image_paths),
            "mean_detections": float(np.mean([len(v) for v in preds.values()]))}
    save_predictions(PRED_DIR / f"yolo_{tag}_{args.split}.json", preds, meta)
    with open(METRICS_DIR / f"yolo_{tag}_{args.split}_runtime.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {len(image_paths)} images, {meta['ms_per_image']:.1f} ms/img, "
          f"mean boxes/image={meta['mean_detections']:.1f}")


if __name__ == "__main__":
    main()
