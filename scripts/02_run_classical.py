"""Run the classical detectors on validation or test images.

Run all three variants using the settings selected on validation:
    python scripts/02_run_classical.py --variant all --split test --use-best

Add --split val --limit 20 for a short check. The script uses
ClassicalParams defaults when the tuning file is missing.
"""
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
        # Global Otsu uses the shared filters and has no multiscale parameters.
        run_split(args.split, v, params, limit=args.limit)


if __name__ == "__main__":
    main()
