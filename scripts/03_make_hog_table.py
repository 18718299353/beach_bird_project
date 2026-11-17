"""Write the HOG ablation table to results/tables/hog_ablation.csv.

The comparisons use the validation split and the same AP implementation.
They cover hard negatives (0 or 6,000), HOG and HOG+LBP, linear SVM and
random forest, and sliding windows and region proposals.
"""


from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (METRICS_DIR, PRED_DIR, SMALL_MAX_AREA, TABLE_DIR, box_areas,
                        ensure_dirs, evaluate_ap, load_predictions, load_split)

CONFIGS = [
    # (factor, setting, prediction file, metric file)
    ("Hard negatives", "0", "hog_hog_nohard_prop_val.json", "hog_hog_nohard_prop_val.json"),
    ("Hard negatives", "6,000", "hog_hog_prop_val.json", "hog_hog_prop_val.json"),
    ("Feature", "HOG", "hog_hog_prop_val.json", "hog_hog_prop_val.json"),
    ("Feature", "HOG + LBP", "hog_hog_lbp_prop_val.json", "hog_hog_lbp_prop_val.json"),
    ("Classifier", "Linear SVM", "hog_hog_prop_val.json", "hog_hog_prop_val.json"),
    ("Classifier", "Random Forest", "hog_hog_rf_prop_val.json", "hog_hog_rf_prop_val.json"),
    ("Inference", "Sliding window", "hog_hog_linear_val.json", "hog_hog_linear_val.json"),
    ("Inference", "Black-hat proposals", "hog_hog_prop_val.json", "hog_hog_prop_val.json"),
]


def main() -> None:
    ensure_dirs()
    recs = load_split("val")
    rows = []
    for factor, setting, pred_name, metric_name in CONFIGS:
        pp = PRED_DIR / pred_name
        if not pp.exists():
            print(f"[skip] {pred_name}")
            continue
        preds, meta = load_predictions(pp)
        ap = evaluate_ap(preds, recs, 0.5)
        ap_small = evaluate_ap(preds, recs, 0.5,
                               gt_filter=lambda b: box_areas(b) < SMALL_MAX_AREA)
        ms = meta.get("ms_per_image")
        if ms is None:
            mp = METRICS_DIR / metric_name
            if mp.exists():
                with open(mp, encoding="utf-8") as f:
                    ms = json.load(f).get("ms_per_image")
        rows.append({"factor": factor, "setting": setting,
                     "AP@0.5": round(float(ap), 4),
                     "AP_small": round(float(ap_small), 4),
                     "ms_per_image": round(float(ms), 1) if ms is not None else ""})
        print(f"{factor:16s} {setting:22s} AP={ap:.4f} AP_small={ap_small:.4f} ms={ms}")

    out = TABLE_DIR / "hog_ablation.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["factor", "setting", "AP@0.5", "AP_small", "ms_per_image"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
