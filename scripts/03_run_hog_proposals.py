"""
03_run_hog_proposals.py —— 混合检测器：传统方法出候选框 + HOG/SVM 打分
======================================================================

动机：
纯滑窗 HOG 在本数据集上定位很差——一只 20 px 的鸟要被 32/48/64 的窗口
以 IoU≥0.5 套住，对步长和窗口尺寸极其敏感。传统黑帽方法虽然误检多，
但它给出的候选框通常**贴合目标轮廓**。于是我们做一个级联：

    黑帽/顶帽多尺度分割 → 候选框（~30 个/图）
    → 每个候选框裁 patch → HOG 特征 → SVM 打分
    → 阈值 + NMS → 最终检测

这样既保留了传统方法良好的定位，又用学习到的特征过滤掉大量礁石/浪花误检。
速度也比全图滑窗快一个数量级（每图只需几十次 HOG 计算）。

用法：
    python scripts/03_run_hog_proposals.py --model hog_linear --split val
    python scripts/03_run_hog_proposals.py --model hog_linear --split test
"""

# 模型测试
# 加载训练好的模型，使用滑动窗口检测鸟类并评估性能

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (METRICS_DIR, MODELS_DIR, PRED_DIR, box_iou, clip_boxes,
                        ensure_dirs, evaluate_ap, load_split, nms, save_predictions,
                        select_operating_threshold)
from src.classical import ClassicalParams, detect_image
from src.hog_svm import FeatureExtractor, crop_square


def detect_proposals(img, model, extractor, classical_params: ClassicalParams,
                     candidate_threshold: float = 0.10, nms_iou: float = 0.4,
                     max_det: int = 100):
    """返回 (N,5) 的 [x1,y1,x2,y2,svm_score]。"""
    cand = detect_image(img, "multiscale_blackhat", classical_params)
    if len(cand) == 0:
        return np.zeros((0, 5), np.float32)
    # 只保留传统方法分数不太低的候选，减少 HOG 计算量
    cand = cand[cand[:, 4] >= candidate_threshold]
    if len(cand) == 0:
        return np.zeros((0, 5), np.float32)
    # 按传统分数保留前 80 个，避免密集图里候选爆炸
    if len(cand) > 80:
        cand = cand[np.argsort(-cand[:, 4])[:80]]

    patches = []
    for b in cand[:, :4]:
        x1, y1, x2, y2 = b
        side = max(x2 - x1, y2 - y1) * 1.3
        patches.append(crop_square(img, (x1 + x2) / 2, (y1 + y2) / 2, side, 64))
    X = extractor.batch(patches)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = model.predict_proba(X)[:, 1]

    boxes = cand[:, :4].astype(np.float32)
    scores = np.asarray(scores, np.float32)
    keep = nms(boxes, scores, nms_iou)
    keep = keep[np.argsort(-scores[keep])][:max_det]
    h, w = img.shape[:2]
    return np.concatenate([clip_boxes(boxes[keep], w, h), scores[keep, None]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hog_linear")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--candidate-threshold", type=float, default=0.10)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="hog_prop")
    args = ap.parse_args()
    ensure_dirs()

    ckpt = joblib.load(MODELS_DIR / "hog_svm" / f"{args.model}.joblib")
    extractor = FeatureExtractor(64, ckpt["feature"]["use_hog"], ckpt["feature"]["use_lbp"])
    model = ckpt["model"]
    cp = ClassicalParams()

    records = load_split(args.split)
    if args.limit:
        records = records[: args.limit]
    preds = {}
    t0 = time.time()
    for i, rec in enumerate(records, start=1):
        preds[rec.image_id] = detect_proposals(rec.load_image(), model, extractor, cp,
                                               args.candidate_threshold, args.nms)
        if i % 50 == 0:
            print(f"  {i}/{len(records)}", flush=True)
    elapsed = time.time() - t0
    meta = {"method": "hog_proposals", "model": args.model, "split": args.split,
            "inference_seconds": elapsed, "ms_per_image": 1000 * elapsed / len(records),
            "mean_detections": float(np.mean([len(v) for v in preds.values()])),
            "candidate_threshold": args.candidate_threshold, "nms_iou": args.nms}
    save_predictions(PRED_DIR / f"hog_{args.tag}_{args.split}.json", preds, meta)
    ap50 = evaluate_ap(preds, records, 0.5)
    thr, best = select_operating_threshold(preds, records)
    print(f"[{args.tag}/{args.split}] AP@0.5={ap50:.4f} F1={best['f1']:.3f} thr={thr:.2f} "
          f"boxes/img={meta['mean_detections']:.1f} {meta['ms_per_image']:.1f} ms/img")
    with open(METRICS_DIR / f"hog_{args.tag}_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "AP@0.5": ap50, "best_F1": best["f1"],
                   "best_threshold": thr, **best}, f, indent=2)


if __name__ == "__main__":
    main()
