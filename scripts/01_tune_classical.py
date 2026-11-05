"""
01_tune_classical.py —— 在验证集上为传统方法做参数扫描
=========================================================

**只在 val 上运行**，测试集保持不动，避免数据泄漏。

用法：
    .venv\\Scripts\\python.exe scripts\\01_tune_classical.py --variant multiscale_blackhat --subset 120

输出：
    results/tables/classical_tuning.csv

为什么需要这个脚本？
--------------------
传统方法有很多"魔法数字"：阈值、最小面积、结构元素大小……
我们不能凭感觉拍脑袋，而是固定其他参数、一次只改一个参数（坐标下降），
用 AP@0.5 和最佳 F1 选择。报告中的 "parameter sensitivity" 表就来自这里。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (METRICS_DIR, TABLE_DIR, ensure_dirs, evaluate_ap, load_split,
                        prf_at_threshold, select_operating_threshold)
from src.classical import ClassicalParams, detect_image


def stratified_subset(records, n: int):
    """按"每图鸟数"分层抽样，保证稀疏/密集场景都有代表。"""
    if n >= len(records):
        return records
    rng = np.random.default_rng(42)
    groups = {}
    for r in records:
        key = min(r.num_objects // 10, 6)  # 0,1-9,10-19,...
        groups.setdefault(key, []).append(r)
    out = []
    for key, items in groups.items():
        k = max(1, round(n * len(items) / len(records)))
        idx = rng.choice(len(items), size=min(k, len(items)), replace=False)
        out.extend([items[i] for i in idx])
    return out[:n]


def evaluate_config(images, records, p: ClassicalParams, variant: str) -> dict:
    preds = {}
    t0 = time.time()
    for rec, img in zip(records, images):
        preds[rec.image_id] = detect_image(img, variant, p)
    elapsed = time.time() - t0
    ap = evaluate_ap(preds, records, 0.5)
    thr, best = select_operating_threshold(preds, records)
    return {"AP@0.5": ap, "best_F1": best["f1"], "P": best["precision"],
            "R": best["recall"], "threshold": thr,
            "boxes_per_image": float(np.mean([len(v) for v in preds.values()])),
            "ms_per_image": 1000 * elapsed / len(records)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="multiscale_blackhat")
    ap.add_argument("--subset", type=int, default=120)
    args = ap.parse_args()
    ensure_dirs()

    records_all = load_split("val")
    records = stratified_subset(records_all, args.subset)
    images = [r.load_image() for r in records]
    print(f"Tuning on {len(records)} val images (variant={args.variant})")

    base = ClassicalParams()
    # 坐标下降：每个 stage 只扫描一组参数，选 AP 最高的一组进入下一阶段
    stages = [
        ("threshold", {"resp_floor": [8, 15, 22, 30], "otsu_scale": [1.0, 1.3, 1.6]}),
        ("shape", {"min_area": [80, 110, 150], "min_fill": [0.30, 0.40, 0.50]}),
        ("size", {"max_area": [12000, 18000, 25000], "max_aspect": [2.5, 3.5, 4.5]}),
        ("merge", {"nms_iou": [0.25, 0.35, 0.45], "close_ksize": [1, 3, 5]}),
    ]

    rows = []
    best_params = base
    best_ap = -1
    for stage_name, grid in stages:
        keys = list(grid.keys())
        for values in itertools.product(*[grid[k] for k in keys]):
            p = replace(best_params, **dict(zip(keys, values)))
            m = evaluate_config(images, records, p, args.variant)
            row = {"stage": stage_name, **{k: v for k, v in zip(keys, values)}, **m}
            rows.append(row)
            print(f"  [{stage_name}] " + " ".join(f"{k}={v}" for k, v in zip(keys, values)) +
                  f" -> AP={m['AP@0.5']:.4f} F1={m['best_F1']:.3f} "
                  f"boxes/img={m['boxes_per_image']:.1f}")
            if m["AP@0.5"] > best_ap:
                best_ap = m["AP@0.5"]
                best_params = p
        print(f"[stage {stage_name}] best so far: AP={best_ap:.4f}")

    # 保存所有尝试
    out_csv = TABLE_DIR / "classical_tuning.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 保存最佳参数，后续 test 直接读取
    with open(METRICS_DIR / "classical_best_params.json", "w", encoding="utf-8") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(best_params).items()}, f, indent=2)
    print(f"\nBest AP={best_ap:.4f}")
    print(f"Best params -> {METRICS_DIR / 'classical_best_params.json'}")
    print(f"Tuning table -> {out_csv}")


if __name__ == "__main__":
    main()
