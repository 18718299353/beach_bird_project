"""
06_evaluate_all.py —— 汇总所有方法，计算统一指标并生成结果表
=============================================================

用法：
    .venv\\Scripts\\python.exe scripts\\06_evaluate_all.py

做的事情：
1. 读取每个方法在 **val** 上的预测，选出使 F1 最大的置信度阈值；
2. 把这个阈值原样用到 **test** 上（test 不参与任何调参）；
3. 计算 AP@0.5 / AP@0.75 / AP@[.5:.95]、P/R/F1、计数误差、负样本误检、速度；
4. 按目标尺寸（small/medium/large）和鸟群密度分组统计；
5. 保存 CSV 表格、PR 曲线图、每张图的 TP/FP/FN（用于失败案例分析）。

为什么要"在 val 上选阈值，再用于 test"？
----------------------------------------
如果直接在 test 上挑一个最好的阈值，相当于用测试集调参，
得到的分数会虚高，不能代表真实泛化能力。作业明确要求避免这种数据泄漏。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (FIG_DIR, METRICS_DIR, PRED_DIR, TABLE_DIR, box_areas, box_iou,
                        count_metrics, density_breakdown, ensure_dirs, evaluate_ap,
                        evaluate_ap_range, load_predictions, load_split,
                        negative_scene_fp, pr_curve, prf_at_threshold,
                        select_operating_threshold, size_breakdown)

# 每个方法：显示名、所属类别、预测文件前缀、训练耗时文件（可选）
METHODS = [
    dict(key="classical_global_otsu", display="Global Otsu", family="Non-learning",
         prefix="classical_global_otsu", train_json=None),
    dict(key="classical_multiscale_blackhat", display="Multi-scale Black-hat",
         family="Non-learning", prefix="classical_multiscale_blackhat", train_json=None),
    dict(key="classical_multiscale_blackhat_ws", display="Black-hat + Watershed",
         family="Non-learning", prefix="classical_multiscale_blackhat_ws", train_json=None),
    dict(key="hog_linear", display="HOG + Linear SVM (sliding)", family="Machine learning",
         prefix="hog_hog_linear", train_json="hog_hog_linear.json"),
    dict(key="hog_prop", display="HOG + SVM (proposals)", family="Machine learning",
         prefix="hog_hog_prop", train_json="hog_hog_linear.json"),
    dict(key="hog_lbp_prop", display="HOG+LBP + SVM (proposals)", family="Machine learning",
         prefix="hog_hog_lbp_prop", train_json="hog_hog_lbp_linear.json"),
    dict(key="hog_rf_prop", display="HOG + RF (proposals)", family="Machine learning",
         prefix="hog_hog_rf_prop", train_json="hog_hog_rf.json"),
    dict(key="yolov8n_plain", display="YOLOv8n (no aug)", family="Deep learning",
         prefix="yolo_yolov8n_plain", train_json="train_yolov8n_plain.json"),
    dict(key="yolov8n_aug", display="YOLOv8n + aug", family="Deep learning",
         prefix="yolo_yolov8n_aug", train_json="train_yolov8n_aug.json"),
    dict(key="yolo11n_aug", display="YOLO11n + aug", family="Deep learning",
         prefix="yolo_yolo11n_aug", train_json="train_yolo11n_aug.json"),
    dict(key="yolov8s_aug", display="YOLOv8s + aug", family="Deep learning",
         prefix="yolo_yolov8s_aug", train_json="train_yolov8s_aug.json"),
    dict(key="yolov8n_aug_960", display="YOLOv8n + aug @960", family="Deep learning",
         prefix="yolo_yolov8n_aug_960", train_json="train_yolov8n_aug_960.json"),
    # 后处理改进（同一权重，不同推理策略）
    dict(key="yolov8n_aug_tta", display="YOLOv8n + aug + TTA", family="Deep learning (post-hoc)",
         prefix="yolo_yolov8n_aug_tta", train_json="train_yolov8n_aug.json"),
    dict(key="yolov8n_aug_tile2", display="YOLOv8n + aug + 2×2 tiles",
         family="Deep learning (post-hoc)", prefix="yolo_yolov8n_aug_tile2",
         train_json="train_yolov8n_aug.json"),
]


def per_image_stats(preds, records, threshold: float, iou_thr: float = 0.5) -> List[Dict]:
    """逐图统计 TP/FP/FN 和定位质量，供失败案例分析使用。"""
    rows = []
    for rec in records:
        det = preds.get(rec.image_id, np.zeros((0, 5), np.float32))
        if len(det):
            det = det[det[:, 4] >= threshold]
            det = det[np.argsort(-det[:, 4])]
        gt = rec.boxes
        matched = np.zeros(len(gt), dtype=bool)
        tp = fp = 0
        ious_tp = []
        for d in range(len(det)):
            if len(gt) == 0:
                fp += 1
                continue
            ious = box_iou(det[d, :4][None, :], gt)[0]
            cand = np.where(~matched)[0]
            if cand.size and ious[cand].max() >= iou_thr:
                b = cand[np.argmax(ious[cand])]
                matched[b] = True
                tp += 1
                ious_tp.append(float(ious[b]))
            else:
                fp += 1
        rows.append({"image_id": rec.image_id, "gt": int(len(gt)), "pred": int(len(det)),
                     "tp": tp, "fp": fp, "fn": int((~matched).sum()),
                     "mean_tp_iou": float(np.mean(ious_tp)) if ious_tp else 0.0,
                     "density": int(len(gt))})
    return rows


def main() -> None:
    ensure_dirs()
    val_records = load_split("val")
    test_records = load_split("test")
    summary = {}
    main_rows = []
    size_rows = []
    density_rows = []

    plt.figure(figsize=(6.4, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))

    for method, color in zip(METHODS, colors):
        val_path = PRED_DIR / f"{method['prefix']}_val.json"
        test_path = PRED_DIR / f"{method['prefix']}_test.json"
        if not val_path.exists() or not test_path.exists():
            print(f"[skip] {method['display']}: missing val/test predictions")
            continue
        val_preds, val_meta = load_predictions(val_path)
        test_preds, test_meta = load_predictions(test_path)

        # 1) 在 val 上选阈值
        thr, val_best = select_operating_threshold(val_preds, val_records)
        # 2) 原样用于 test
        ap = evaluate_ap(test_preds, test_records, 0.5)
        aps = evaluate_ap_range(test_preds, test_records,
                                iou_thresholds=np.arange(0.5, 1.0, 0.05))
        prf = prf_at_threshold(test_preds, test_records, thr)
        cm = count_metrics(test_preds, test_records, thr)
        neg = negative_scene_fp(test_preds, test_records, thr)
        sizes = size_breakdown(test_preds, test_records, thr)
        dens = density_breakdown(test_preds, test_records, thr)

        train_minutes = None
        train_json = method.get("train_json")
        if train_json and (METRICS_DIR / train_json).exists():
            with open(METRICS_DIR / train_json, encoding="utf-8") as f:
                tm = json.load(f)
            train_minutes = tm.get("train_minutes", tm.get("total_seconds", 0) / 60)

        row = {
            "method": method["display"], "family": method["family"],
            "AP@0.5": ap, "AP@0.75": aps["AP@0.75"], "AP@0.5:0.95": aps["AP@0.5:0.95"],
            "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
            "threshold": thr, "count_mae": cm["count_mae"], "count_bias": cm["count_bias"],
            "neg_fp_per_image": neg["neg_fp_per_image"],
            "ms_per_image": test_meta.get("ms_per_image", float("nan")),
            "train_minutes": train_minutes,
            "mean_detections": test_meta.get("mean_detections", float("nan")),
        }
        main_rows.append(row)
        summary[method["key"]] = {"main": row, "size": sizes, "density": dens,
                                  "val_best": val_best, "neg": neg, "count": cm}
        for s in sizes:
            size_rows.append({"method": method["display"], **s})
        for d in dens:
            density_rows.append({"method": method["display"], **d})

        # 逐图统计
        with open(METRICS_DIR / f"per_image_{method['key']}_test.json", "w", encoding="utf-8") as f:
            json.dump(per_image_stats(test_preds, test_records, thr), f)

        # PR 曲线
        rec, prec = pr_curve(test_preds, test_records, 0.5)
        plt.plot(rec, prec, label=f"{method['display']} ({ap:.3f})", color=color, lw=1.6)

        print(f"[{method['display']}] AP50={ap:.3f} AP75={aps['AP@0.75']:.3f} "
              f"P={prf['precision']:.3f} R={prf['recall']:.3f} F1={prf['f1']:.3f} "
              f"negFP/img={neg['neg_fp_per_image']:.2f} {row['ms_per_image']:.0f}ms")

    # ---------------- 保存表格 ----------------
    import csv
    def write_csv(path, rows):
        if not rows:
            return
        fields = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    write_csv(TABLE_DIR / "main_results.csv", main_rows)
    write_csv(TABLE_DIR / "size_breakdown.csv", size_rows)
    write_csv(TABLE_DIR / "density_breakdown.csv", density_rows)
    with open(METRICS_DIR / "summary_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves on the test set (IoU=0.5)")
    plt.xlim(0, 1); plt.ylim(0, 1.01); plt.grid(alpha=0.3)
    plt.legend(fontsize=6.5, loc="lower left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_pr_curves.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {TABLE_DIR/'main_results.csv'}")
    print(f"Saved: {FIG_DIR/'fig_pr_curves.png'}")


if __name__ == "__main__":
    main()
