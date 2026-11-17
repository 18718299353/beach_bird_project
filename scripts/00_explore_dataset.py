"""Summarize the dataset and plot examples with ground-truth boxes.

Run python scripts/00_explore_dataset.py from the repository root. The
outputs are dataset_stats.json in results/metrics/ and the example and
distribution plots in results/figures/.

Use the size and count distributions to choose sliding-window sizes and
detector input sizes. The median box covers about 0.68% of the image, and
roughly 11% of images contain zero birds. Those images help measure false
positives.
"""

#
# The detection scripts can run independently of these statistics.
#

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (DENSITY_BINS, FIG_DIR, METRICS_DIR, ensure_dirs, load_split,
                        box_areas, draw_boxes)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def percentile(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> None:
    ensure_dirs()
    stats = {}
    all_areas, all_w, all_h = [], [], []

    for split in ("train", "val", "test"):
        recs = load_split(split)
        counts = np.array([r.num_objects for r in recs])
        areas = np.concatenate([box_areas(r.boxes) for r in recs]) if recs else np.array([])
        widths = np.concatenate([r.boxes[:, 2] - r.boxes[:, 0] for r in recs if len(r.boxes)])
        heights = np.concatenate([r.boxes[:, 3] - r.boxes[:, 1] for r in recs if len(r.boxes)])
        all_areas.append(areas); all_w.append(widths); all_h.append(heights)

        stats[split] = {
            "images": len(recs),
            "positive_images": int((counts > 0).sum()),
            "negative_images": int((counts == 0).sum()),
            "objects": int(counts.sum()),
            "boxes_per_image_mean": float(counts.mean()),
            "boxes_per_image_median": float(np.median(counts)),
            "boxes_per_image_max": int(counts.max()),
            "box_width_px": {k: percentile(widths, q) for k, q in
                             [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]},
            "box_height_px": {k: percentile(heights, q) for k, q in
                              [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]},
            "box_area_px2": {k: percentile(areas, q) for k, q in
                             [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]},
            "box_area_fraction_p50": percentile(areas / (512 * 512), 50),
            "small_boxes": int((areas < 32**2).sum()),
            "medium_boxes": int(((areas >= 32**2) & (areas < 96**2)).sum()),
            "large_boxes": int((areas >= 96**2).sum()),
            "density_hist": {name: int(((counts >= lo) & (counts <= hi)).sum())
                             for lo, hi, name in DENSITY_BINS},
        }
        print(f"[{split}] images={len(recs)} boxes={int(counts.sum())} "
              f"max/img={int(counts.max())} median_box_area={percentile(areas, 50):.1f}px^2")

    areas = np.concatenate(all_areas)
    stats["all"] = {
        "images": sum(stats[s]["images"] for s in ("train", "val", "test")),
        "objects": int(areas.size),
        "box_area_px2": {k: percentile(areas, q) for k, q in
                         [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]},
        "small_fraction": float((areas < 32**2).mean()),
        "medium_fraction": float(((areas >= 32**2) & (areas < 96**2)).mean()),
        "large_fraction": float((areas >= 96**2).mean()),
    }

    with open(METRICS_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Figure 1 groups example images by bird count, including empty scenes.
    # ------------------------------------------------------------------
    train = load_split("train")
    picks = []
    for lo, hi, name in DENSITY_BINS:
        cand = [r for r in train if lo <= r.num_objects <= hi]
        if cand:
            # Pick the image closest to the median count in each group.
            med = int(np.median([r.num_objects for r in cand]))
            cand.sort(key=lambda r: (abs(r.num_objects - med), r.image_id))
            picks.append((name, cand[0]))

    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.9))
    for ax, (name, rec) in zip(np.atleast_1d(axes), picks):
        img = cv2.cvtColor(rec.load_image(), cv2.COLOR_BGR2RGB)
        img = draw_boxes(img, rec.boxes, color=(0, 255, 0), thickness=1)
        ax.imshow(img)
        ax.set_title(f"{name}\n{rec.num_objects} birds", fontsize=9)
        ax.axis("off")
    fig.suptitle("Beach Bird dataset: examples across density levels (green = ground truth)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dataset_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2 shows the dataset distributions.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].hist(np.sqrt(areas), bins=60, color="#4c72b0", alpha=0.9)
    axes[0].axvline(32, color="red", ls="--", label="small/medium boundary (32 px)")
    axes[0].set_xlabel("box side length (px, sqrt(area))")
    axes[0].set_ylabel("count")
    axes[0].set_title("(a) Object size distribution (all 12,024 boxes)")
    axes[0].legend(fontsize=7)

    counts = np.array([r.num_objects for r in train])
    axes[1].hist(counts, bins=np.arange(0, counts.max() + 3, 3), color="#dd8452", alpha=0.9)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("birds per image")
    axes[1].set_ylabel("number of images (log)")
    axes[1].set_title(f"(b) Train density (max = {counts.max()})")

    # Plot brightness variation from reflections and changing beach lighting.
    brightness = []
    for rec in train[::5]:  # Sample every fifth image.
        g = cv2.cvtColor(rec.load_image(), cv2.COLOR_BGR2GRAY)
        brightness.append(float(g.mean()))
    axes[2].hist(brightness, bins=40, color="#55a868", alpha=0.9)
    axes[2].set_xlabel("mean gray level")
    axes[2].set_ylabel("images")
    axes[2].set_title("(c) Image brightness (train sample)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dataset_stats.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved stats -> {METRICS_DIR / 'dataset_stats.json'}")
    print(f"Saved figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
