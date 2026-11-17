"""Generate figures for the report and README.

Run scripts/06_evaluate_all.py before plotting. The figures under
results/figures/ include test AP, AP versus runtime, size and density
breakdowns, qualitative examples, failure cases, validation curves for the
five YOLO checkpoints, the HOG ablations and the TTA and tiling comparison.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (FIG_DIR, METRICS_DIR, PRED_DIR, TABLE_DIR, draw_boxes,
                        ensure_dirs, load_predictions, load_split)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "figure.dpi": 160, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25,
})

FAMILY_COLORS = {
    "Non-learning": "#9e9e9e",
    "Machine learning": "#e07b39",
    "Deep learning": "#2f6db3",
    "Deep learning (post-hoc)": "#59a14f",
}
# Try the prediction files in this order when picking a method for the qualitative plot.
QUAL_METHODS = [
    ("YOLOv8n + aug", "yolo_yolov8n_aug_test.json", (255, 0, 0)),
    ("HOG + Linear SVM", "hog_hog_linear_test.json", (0, 140, 255)),
    ("Black-hat + Watershed", "classical_multiscale_blackhat_ws_test.json", (180, 0, 180)),
]


def read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_pred(prefix: str, split: str = "test"):
    p = PRED_DIR / f"{prefix}_{split}.json"
    if not p.exists():
        return None, None
    return load_predictions(p)


# ---------------------------------------------------------------------------
# 1. main AP bar
# ---------------------------------------------------------------------------
def fig_main_ap(rows):
    rows = sorted(rows, key=lambda r: float(r["AP@0.5"]))
    names = [r["method"] for r in rows]
    vals = [float(r["AP@0.5"]) for r in rows]
    colors = [FAMILY_COLORS.get(r["family"], "#888") for r in rows]
    fig, ax = plt.subplots(figsize=(3.45, 0.24 * len(rows) + 0.8))
    ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=6.5)
    ax.set_xlim(0, min(1.0, max(vals) * 1.18 + 0.02))
    ax.set_xlabel("AP@0.5 (test set)")
    ax.set_title("Detection accuracy by method")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), fontsize=5.8, loc="lower right",
              frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_main_ap.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. AP vs time
# ---------------------------------------------------------------------------
def fig_ap_vs_time(rows):
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for r in rows:
        x = max(float(r["ms_per_image"]), 0.1)
        y = float(r["AP@0.5"])
        c = FAMILY_COLORS.get(r["family"], "#888")
        ax.scatter(x, y, s=22, color=c, edgecolor="black", linewidth=0.3, zorder=3)
        ax.annotate(r["method"], (x, y), fontsize=4.6, xytext=(2, 2),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per image (ms, log scale)")
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy–runtime trade-off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ap_vs_time.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. size breakdown
# ---------------------------------------------------------------------------
def fig_size_ap(rows):
    selected = ["YOLOv8n + aug", "HOG + RF (proposals)", "Black-hat + Watershed"]
    groups = ["small (<32² px)", "medium (32²-96² px)", "large (≥96² px)"]
    data = {g: [] for g in groups}
    labels = []
    for name in selected:
        sub = [r for r in rows if r["method"] == name]
        if not sub:
            continue
        labels.append(name)
        for g in groups:
            match = [r for r in sub if r["group"] == g]
            data[g].append(float(match[0]["AP@0.5"]) if match else 0.0)
    if not labels:
        return
    x = np.arange(len(labels)); width = 0.26
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for i, g in enumerate(groups):
        vals = data[g]
        ax.bar(x + (i - 1) * width, vals, width, label=g, edgecolor="black", linewidth=0.3)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=5.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("AP@0.5"); ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by object size")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_size_ap.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. density recall
# ---------------------------------------------------------------------------
def fig_density(rows):
    selected = ["YOLOv8n + aug", "HOG + RF (proposals)", "Black-hat + Watershed"]
    order = ["1-5 (sparse)", "6-20 (moderate)", "21-50 (dense)", "51+ (very dense)"]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for name, marker in zip(selected, ["o", "s", "^"]):
        sub = [r for r in rows if r["method"] == name]
        if not sub:
            continue
        xs, ys = [], []
        for i, g in enumerate(order):
            m = [r for r in sub if r["group"] == g]
            if m:
                xs.append(i); ys.append(float(m[0]["recall"]))
        ax.plot(xs, ys, marker=marker, label=name, linewidth=1.4, markersize=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["1-5", "6-20", "21-50", "51+"])
    ax.set_xlabel("Birds per image (density group)")
    ax.set_ylabel("Recall at operating threshold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Recall vs. flock density")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_density_recall.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5/6. qualitative examples
# ---------------------------------------------------------------------------
def _panel(ax, img, boxes, color, title, scores=None):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(draw_boxes(rgb, boxes, color=color, thickness=1, scores=scores))
    ax.set_title(title, fontsize=6.5)
    ax.axis("off")


def fig_qualitative(records):
    """Show one image per density band with ground truth and YOLO predictions side by side."""
    test = load_split("test")
    picks = []
    for lo, hi, label in [(3, 6, "sparse"), (8, 14, "moderate"), (30, 70, "dense")]:
        cand = [r for r in test if lo <= r.num_objects <= hi]
        if cand:
            cand.sort(key=lambda r: abs(r.num_objects - (lo + hi) // 2))
            picks.append((label, cand[0]))
    p, _ = find_pred("yolo_yolov8n_aug", "test")
    fig, axes = plt.subplots(len(picks), 2, figsize=(3.45, 1.45 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for row, (label, rec) in enumerate(picks):
        img = rec.load_image()
        _panel(axes[row, 0], img, rec.boxes, (0, 200, 0),
               f"GT — {label} ({rec.num_objects} birds)")
        if p is None:
            axes[row, 1].axis("off"); continue
        det = p.get(rec.image_id, np.zeros((0, 5), np.float32))
        det = det[det[:, 4] >= 0.25]
        det = det[np.argsort(-det[:, 4])][:40]
        _panel(axes[row, 1], img, det[:, :4], (255, 0, 0),
               f"YOLOv8n ({len(det)} boxes, conf≥0.25)", det[:, 4])
    fig.tight_layout(pad=0.25)
    fig.savefig(FIG_DIR / "fig_qualitative.png", bbox_inches="tight")
    plt.close(fig)


def fig_failures(records):
    """Choose examples with false positives in empty scenes, misses in dense flocks, and missed small birds."""
    test = load_split("test")
    p, _ = find_pred("yolo_yolov8n_aug", "test")
    if p is None:
        return
    per_path = METRICS_DIR / "per_image_yolov8n_aug_test.json"
    if not per_path.exists():
        return
    with open(per_path, encoding="utf-8") as f:
        per = json.load(f)
    rec_by_id = {r.image_id: r for r in test}
    cases = []
    neg = [x for x in per if x["gt"] == 0]
    if neg:
        cases.append(("False positives, negative scene", max(neg, key=lambda x: x["fp"])))
    dense = [x for x in per if x["gt"] >= 20]
    if dense:
        cases.append(("Missed birds, dense flock", max(dense, key=lambda x: x["fn"])))
    small = [x for x in per if 0 < x["gt"] < 10]
    if small:
        cases.append(("Missed small/sparse birds", max(small, key=lambda x: x["fn"])))
    # Add a medium-density image with both many false positives and many misses.
    mid = [x for x in per if 10 <= x["gt"] < 20]
    if mid:
        cases.append(("Clutter (FP) and missed birds", max(mid, key=lambda x: x["fp"] + x["fn"])))
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.45))
    axes = axes.ravel()
    for ax, (title, row) in zip(axes, cases):
        rec = rec_by_id[row["image_id"]]
        det = p.get(rec.image_id, np.zeros((0, 5), np.float32))
        det = det[det[:, 4] >= 0.25]
        img = cv2.cvtColor(rec.load_image(), cv2.COLOR_BGR2RGB)
        img = draw_boxes(img, rec.boxes, (0, 200, 0), 1)
        img = draw_boxes(img, det[:, :4], (255, 0, 0), 1, scores=det[:, 4])
        ax.imshow(img)
        ax.set_title(f"{title}\nGT={row['gt']} TP={row['tp']} FP={row['fp']} FN={row['fn']}",
                     fontsize=6)
        ax.axis("off")
    for ax in axes[len(cases):]:
        ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(FIG_DIR / "fig_failures.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. YOLO training curves
# ---------------------------------------------------------------------------
def fig_yolo_curves():
    runs = ["yolov8n_plain", "yolov8n_aug", "yolo11n_aug", "yolov8s_aug", "yolov8n_aug_960"]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    found = False
    for run in runs:
        p = Path(__file__).resolve().parents[1] / "models" / "runs" / run / "results.csv"
        if not p.exists():
            continue
        rows = read_csv(p)
        if not rows:
            continue
        epochs = [int(float(r["epoch"])) for r in rows]
        key = [k for k in rows[0] if "mAP50(B)" in k]
        if not key:
            continue
        vals = [float(r[key[0]]) for r in rows]
        ax.plot(epochs, vals, label=run, linewidth=1.3)
        found = True
    if not found:
        plt.close(fig); return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation mAP@0.5")
    ax.set_title("YOLO training curves")
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_yolo_curves.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. HOG ablation
# ---------------------------------------------------------------------------
def fig_hog_ablation():
    p = TABLE_DIR / "hog_ablation.csv"
    rows = read_csv(p)
    if not rows:
        return
    # Plot the stride, feature, and classifier ablations.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    groups = [("stride", "Window stride"), ("feature", "Feature"), ("classifier", "Classifier")]
    for ax, (key, title) in zip(axes, groups):
        sub = [r for r in rows if r.get("factor") == key]
        if not sub:
            ax.axis("off"); continue
        names = [r["setting"] for r in sub]
        vals = [float(r["AP@0.5"]) for r in sub]
        ax.bar(names, vals, color="#e07b39", edgecolor="black", linewidth=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("AP@0.5")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_hog_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. TTA / tiling
# ---------------------------------------------------------------------------
def fig_tta_tile(rows):
    names = ["YOLOv8n + aug", "YOLOv8n + aug + TTA", "YOLOv8n + aug + 2×2 tiles"]
    vals, labels = [], []
    for n in names:
        m = [r for r in rows if r["method"] == n]
        if m:
            labels.append(n.replace("YOLOv8n + aug", "baseline").replace(" + ", "\n+ "))
            vals.append(float(m[0]["AP@0.5"]))
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    bars = ax.bar(labels, vals, color=["#2f6db3", "#59a14f", "#59a14f"],
                  edgecolor="black", linewidth=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("AP@0.5"); ax.set_ylim(0, 1.0)
    ax.set_title("Inference-time improvements")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tta_tile.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    main_rows = read_csv(TABLE_DIR / "main_results.csv")
    size_rows = read_csv(TABLE_DIR / "size_breakdown.csv")
    density_rows = read_csv(TABLE_DIR / "density_breakdown.csv")
    if main_rows:
        fig_main_ap(main_rows)
        fig_ap_vs_time(main_rows)
        fig_tta_tile(main_rows)
    if size_rows:
        fig_size_ap(size_rows)
    if density_rows:
        fig_density(density_rows)
    fig_qualitative(None)
    fig_failures(None)
    fig_yolo_curves()
    fig_hog_ablation()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
