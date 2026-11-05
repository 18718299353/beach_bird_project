"""
06_model_info.py —— 统计每个模型的参数量和文件大小
===================================================

报告中需要比较 "模型大小 / 参数量 / 推理速度"。
YOLO 的参数量从权重里读取；HOG+SVM 的参数就是特征维度 + 分类器系数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import METRICS_DIR, MODELS_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    info = {}

    # ---------------- YOLO ----------------
    run_dirs = {
        "yolov8n_plain": "yolov8n_plain",
        "yolov8n_aug": "yolov8n_aug",
        "yolo11n_aug": "yolo11n_aug",
        "yolov8s_aug": "yolov8s_aug",
        "yolov8n_aug_960": "yolov8n_aug_960",
    }
    try:
        from ultralytics import YOLO
        for name, run in run_dirs.items():
            w = MODELS_DIR / "runs" / run / "weights" / "best.pt"
            if not w.exists():
                continue
            model = YOLO(str(w))
            n_params = sum(p.numel() for p in model.model.parameters())
            info[name] = {"params_millions": n_params / 1e6,
                          "weights_mb": w.stat().st_size / 1e6,
                          "family": "YOLO"}
            print(f"{name}: {n_params/1e6:.2f} M params, {w.stat().st_size/1e6:.1f} MB")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] YOLO model info failed: {e}")

    # ---------------- HOG ----------------
    hog_dir = MODELS_DIR / "hog_svm"
    for name in ("hog_linear", "hog_lbp_linear", "hog_rf"):
        p = hog_dir / f"{name}.joblib"
        if not p.exists():
            continue
        import joblib
        ckpt = joblib.load(p)
        clf = ckpt["model"].named_steps["clf"]
        if hasattr(clf, "coef_"):
            n_params = int(clf.coef_.size + clf.intercept_.size)
        else:
            # 随机森林：用树的节点数近似参数量
            n_params = int(sum(t.tree_.node_count for t in clf.estimators_))
        info[name] = {"params_millions": n_params / 1e6,
                      "weights_mb": p.stat().st_size / 1e6, "family": "HOG+SVM"}
        print(f"{name}: {n_params/1e6:.3f} M params, {p.stat().st_size/1e6:.1f} MB")

    with open(METRICS_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Saved -> {METRICS_DIR / 'model_info.json'}")


if __name__ == "__main__":
    main()
