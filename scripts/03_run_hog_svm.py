"""
03_run_hog_svm.py —— 用训练好的 HOG/LBP 模型在 val 或 test 上做滑窗检测
=======================================================================

示例：
    # 在验证集上评估（调 stride / 窗口尺度 / NMS）
    .venv\\Scripts\\python.exe scripts\\03_run_hog_svm.py --name hog_linear --split val --stride 16

    # 最终测试集（只跑一次）
    .venv\\Scripts\\python.exe scripts\\03_run_hog_svm.py --name hog_linear --split test --stride 16

输出：
    results/predictions/hog_<name>_<split>.json
    results/metrics/hog_<name>_<split>.json

注意：stride、window_sizes、NMS 都是**推理时**参数，不需要重新训练模型。
      这正是做"步长/尺度对精度与速度影响"实验的方便之处。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (METRICS_DIR, MODELS_DIR, PRED_DIR, ensure_dirs, evaluate_ap,
                        load_split, save_predictions, select_operating_threshold)
from src.hog_svm import FeatureExtractor, HogParams, WindowDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--sizes", default="32,48,64,96,128")
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--score-threshold", type=float, default=-0.6)
    ap.add_argument("--size-penalty", type=float, default=0.0,
                    help="大窗口分数惩罚系数，0.3~0.6 可抑制过大的框")
    ap.add_argument("--box-shrink", type=float, default=0.77,
                    help="输出框向中心收缩比例，默认 0.77（对应训练时的 1.3 倍上下文）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 张（调试）")
    ap.add_argument("--tag", default=None, help="输出文件名后缀（默认与 name 相同）")
    args = ap.parse_args()
    ensure_dirs()

    ckpt = joblib.load(MODELS_DIR / "hog_svm" / f"{args.name}.joblib")
    extractor = FeatureExtractor(64, ckpt["feature"]["use_hog"], ckpt["feature"]["use_lbp"])
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    params = HogParams(window_sizes=sizes, stride=args.stride, nms_iou=args.nms,
                       score_threshold=args.score_threshold,
                       size_score_penalty=args.size_penalty,
                       box_shrink=args.box_shrink,
                       use_hog=ckpt["feature"]["use_hog"],
                       use_lbp=ckpt["feature"]["use_lbp"])
    detector = WindowDetector(ckpt["model"], extractor, params)

    records = load_split(args.split)
    if args.limit:
        records = records[: args.limit]
    print(f"[hog] {args.name} on {args.split}: {len(records)} images, stride={args.stride}, "
          f"sizes={sizes}, nms={args.nms}")

    preds = {}
    t0 = time.time()
    for i, rec in enumerate(records, start=1):
        preds[rec.image_id] = detector.detect(rec.load_image())
        if i % 50 == 0:
            print(f"  {i}/{len(records)}", flush=True)
    elapsed = time.time() - t0

    tag = args.tag or args.name
    meta = {"method": f"hog_{args.name}", "split": args.split, "stride": args.stride,
            "window_sizes": list(sizes), "nms_iou": args.nms,
            "inference_seconds": elapsed, "ms_per_image": 1000 * elapsed / len(records),
            "num_images": len(records), "mean_detections": float(np.mean([len(v) for v in preds.values()]))}
    save_predictions(PRED_DIR / f"hog_{tag}_{args.split}.json", preds, meta)

    ap50 = evaluate_ap(preds, records, 0.5)
    thr, best = select_operating_threshold(preds, records)
    metrics = {**meta, "AP@0.5": ap50, "best_F1": best["f1"], "best_threshold": thr,
               "precision": best["precision"], "recall": best["recall"],
               "tp": best["tp"], "fp": best["fp"], "fn": best["fn"]}
    print(f"[hog] AP@0.5={ap50:.4f} best_F1={best['f1']:.3f} (thr={thr:.3f}) "
          f"boxes/img={meta['mean_detections']:.1f}  {meta['ms_per_image']:.1f} ms/img")
    with open(METRICS_DIR / f"hog_{tag}_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
