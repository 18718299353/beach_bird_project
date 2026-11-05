"""
02_run_classical.py —— 在 val/test 上运行传统方法
==================================================

用法：
    # 用调好的参数在测试集上跑三个变体
    .venv\\Scripts\\python.exe scripts\\02_run_classical.py --variant all --split test --use-best

    # 快速调试（只跑 20 张）
    ... --split val --limit 20

参数来自 scripts/01_tune_classical.py 在验证集上得到的结果（--use-best），
如果没有调参文件，就用 ClassicalParams 的默认值。
"""
# 该文件
#
#
#
#

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import METRICS_DIR, ensure_dirs
from src.classical import ClassicalParams, run_split

VARIANTS = ["global_otsu", "multiscale_blackhat", "multiscale_blackhat_ws"]


def load_best() -> ClassicalParams:
    p = METRICS_DIR / "classical_best_params.json"
    if not p.exists():
        return ClassicalParams()
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if "se_sizes" in d:
        d["se_sizes"] = tuple(d["se_sizes"])
    return ClassicalParams(**d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--variant", default="all", choices=VARIANTS + ["all"])
    ap.add_argument("--use-best", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ensure_dirs()

    params = load_best() if args.use_best else ClassicalParams()
    variants = VARIANTS if args.variant == "all" else [args.variant]
    for v in variants:
        # global_otsu 不使用多尺度参数，但其余过滤参数仍适用
        run_split(args.split, v, params, limit=args.limit)


if __name__ == "__main__":
    main()
