"""Train a HOG or HOG+LBP classifier for bird detection.

Example commands
    python scripts/03_train_hog_svm.py --name hog_linear --hard-neg 6000
    python scripts/03_train_hog_svm.py --name hog_lbp_linear --feature hog_lbp --hard-neg 6000
    python scripts/03_train_hog_svm.py --name hog_rf --classifier rf --hard-neg 6000

Fit on the training split and use validation to check the configuration.
The test split is evaluated separately with 03_run_hog_svm.py. This script
saves the classifier under models/hog_svm/ and the training statistics and
hard-negative boxes under results/metrics/.
"""


from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (METRICS_DIR, MODELS_DIR, PRED_DIR, ensure_dirs,
                        evaluate_ap, load_split, save_predictions,
                        select_operating_threshold)
from src.hog_svm import (FeatureExtractor, HogParams, WindowDetector, build_training_data,
                         crop_square, make_classifier, mine_hard_negatives)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="实验名，如 hog_linear")
    ap.add_argument("--feature", choices=["hog", "hog_lbp"], default="hog")
    ap.add_argument("--classifier", choices=["linear", "rf", "rbf"], default="linear")
    ap.add_argument("--hard-neg", type=int, default=6000, help="难负样本数量，0 表示不做")
    ap.add_argument("--hard-images", type=int, default=200, help="用多少张训练图挖难负样本")
    ap.add_argument("--neg-per-image", type=int, default=16)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--sizes", default="32,48,64,96,128")
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--size-penalty", type=float, default=0.0)
    ap.add_argument("--train-limit", type=int, default=None, help="只用前 N 张训练图（调试）")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())
    params = HogParams(window_sizes=sizes, stride=args.stride, nms_iou=args.nms,
                       size_score_penalty=args.size_penalty,
                       neg_per_image=args.neg_per_image, use_hog=True,
                       use_lbp=(args.feature == "hog_lbp"))
    extractor = FeatureExtractor(64, params.use_hog, params.use_lbp)

    train = load_split("train")
    if args.train_limit:
        train = train[: args.train_limit]
    print(f"[train] {args.name}: feature={args.feature} classifier={args.classifier} "
          f"images={len(train)} dim={extractor.dim}")

    t0 = time.time()
    X, y = build_training_data(train, params, extractor)
    t_feat = time.time() - t0
    print(f"[train] base features: {X.shape}, {t_feat:.1f}s")

    clf = make_classifier(args.classifier)
    t1 = time.time()
    clf.fit(X, y)
    t_fit = time.time() - t1
    print(f"[train] initial fit: {t_fit:.1f}s")

    hard_boxes = []
    if args.hard_neg > 0:
        # Use the initial classifier to collect high-scoring false positives from training images.
        detector0 = WindowDetector(clf, extractor, params)
        det_params = HogParams(**{**asdict(params), "score_threshold": 0.0})
        detector0.params = det_params
        print(f"[train] mining hard negatives from {args.hard_images} train images...")
        t2 = time.time()
        # Save the boxes so other feature configurations can reuse them.
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train), size=min(args.hard_images, len(train)), replace=False)
        X_hard = []
        for i, k in enumerate(idx):
            rec = train[int(k)]
            img = rec.load_image()
            det = detector0.detect(img)
            if det.shape[0] == 0:
                continue
            for row in det[np.argsort(-det[:, 4])]:
                box, score = row[:4], float(row[4])
                if score < 0.0:
                    break
                from src.common import box_iou
                if rec.boxes.shape[0] and box_iou(box[None, :], rec.boxes)[0].max() > 0.25:
                    continue
                patch = crop_square(img, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                                    max(box[2] - box[0], box[3] - box[1]), 64)
                X_hard.append(extractor(patch))
                hard_boxes.append({"image_id": rec.image_id,
                                   "box": [round(float(v), 1) for v in box]})
                if len(hard_boxes) >= args.hard_neg:
                    break
            if len(hard_boxes) >= args.hard_neg:
                break
            if (i + 1) % 50 == 0:
                print(f"  [hard-neg] {i+1}/{len(idx)} images, {len(hard_boxes)} negatives", flush=True)
        print(f"[train] mined {len(hard_boxes)} hard negatives in {time.time()-t2:.1f}s")

        if X_hard:
            X = np.concatenate([X, np.stack(X_hard).astype(np.float32)], axis=0)
            y = np.concatenate([y, np.zeros(len(X_hard), dtype=np.int32)], axis=0)
            print(f"[train] retraining on {X.shape[0]} patches...")
            t3 = time.time()
            clf = make_classifier(args.classifier)
            clf.fit(X, y)
            t_fit += time.time() - t3
            print(f"[train] final fit: {time.time()-t3:.1f}s")

    t_total = time.time() - t0

    # Save the fitted classifier.
    import joblib
    model_dir = MODELS_DIR / "hog_svm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.name}.joblib"
    joblib.dump({"model": clf, "feature": {"use_hog": params.use_hog, "use_lbp": params.use_lbp},
                 "params": asdict(params), "name": args.name}, model_path)

    with open(METRICS_DIR / f"hard_negatives_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(hard_boxes, f)

    metrics = {"name": args.name, "feature": args.feature, "classifier": args.classifier,
               "num_train_patches": int(X.shape[0]), "dim": int(X.shape[1]),
               "feature_seconds": t_feat, "fit_seconds": t_fit, "total_seconds": t_total,
               "hard_negatives": len(hard_boxes),
               "params": asdict(params)}

    if not args.no_eval:
        val = load_split("val")
        detector = WindowDetector(clf, extractor, params)
        print(f"[eval] running sliding window on {len(val)} val images...")
        t4 = time.time()
        preds = {}
        for i, rec in enumerate(val, start=1):
            preds[rec.image_id] = detector.detect(rec.load_image())
            if i % 50 == 0:
                print(f"  [eval] {i}/{len(val)}", flush=True)
        infer = time.time() - t4
        ap = evaluate_ap(preds, val, 0.5)
        thr, best = select_operating_threshold(preds, val)
        metrics.update({"val_AP@0.5": ap, "val_best_F1": best["f1"],
                        "val_best_threshold": thr, "val_precision": best["precision"],
                        "val_recall": best["recall"],
                        "inference_seconds": infer,
                        "ms_per_image": 1000 * infer / len(val),
                        "mean_detections": float(np.mean([len(v) for v in preds.values()]))})
        save_predictions(PRED_DIR / f"hog_{args.name}_val.json", preds,
                         {"method": f"hog_{args.name}", "split": "val", **metrics})
        print(f"[eval] AP@0.5={ap:.4f} F1={best['f1']:.3f} thr={thr:.3f} "
              f"boxes/img={metrics['mean_detections']:.1f} {metrics['ms_per_image']:.1f} ms/img")

    with open(METRICS_DIR / f"hog_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] model -> {model_path}")


if __name__ == "__main__":
    main()
