"""Evaluate all settings with both validation and test predictions.

Run python scripts/06_evaluate_all.py from the repository root. Select the
confidence threshold that maximizes F1 on validation and use it for the
test-set operating point. The outputs include AP, precision, recall, F1,
counting errors, false positives in empty scenes, and runtime. The script
also saves size and density breakdowns, a PR plot, and per-image counts.
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

# Each entry gives the display name, family, prediction prefix, and optional training metrics file.
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
    # These inference variants share the augmented YOLOv8n checkpoint.
    dict(key="yolov8n_aug_tta", display="YOLOv8n + aug + TTA", family="Deep learning (post-hoc)",
         prefix="yolo_yolov8n_aug_tta", train_json="train_yolov8n_aug.json"),
    dict(key="yolov8n_aug_tile2", display="YOLOv8n + aug + 2×2 tiles",
         family="Deep learning (post-hoc)", prefix="yolo_yolov8n_aug_tile2",
         train_json="train_yolov8n_aug.json"),
]


def per_image_stats(preds, records, threshold: float, iou_thr: float = 0.5) -> List[Dict]:
    """Collect per-image TP, FP, FN, and localization statistics."""
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

        # Choose the operating threshold on validation.
        thr, val_best = select_operating_threshold(val_preds, val_records)
        # Use the validation threshold for the test split.
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

        # Save per-image statistics for failure analysis.
        with open(METRICS_DIR / f"per_image_{method['key']}_test.json", "w", encoding="utf-8") as f:
            json.dump(per_image_stats(test_preds, test_records, thr), f)

        # Add the method to the test PR plot.
        rec, prec = pr_curve(test_preds, test_records, 0.5)
        plt.plot(rec, prec, label=f"{method['display']} ({ap:.3f})", color=color, lw=1.6)

        print(f"[{method['display']}] AP50={ap:.3f} AP75={aps['AP@0.75']:.3f} "
              f"P={prf['precision']:.3f} R={prf['recall']:.3f} F1={prf['f1']:.3f} "
              f"negFP/img={neg['neg_fp_per_image']:.2f} {row['ms_per_image']:.0f}ms")

    # Write the results tables.
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
