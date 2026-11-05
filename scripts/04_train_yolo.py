"""
04_train_yolo.py —— 训练 YOLO 系列模型
=======================================

用法示例（在项目根目录）：
    .venv\\Scripts\\python.exe scripts\\04_train_yolo.py --model yolov8n.pt --name yolov8n_plain --aug plain
    .venv\\Scripts\\python.exe scripts\\04_train_yolo.py --model yolov8n.pt --name yolov8n_aug   --aug default
    .venv\\Scripts\\python.exe scripts\\04_train_yolo.py --model yolo11n.pt  --name yolo11n_aug  --aug default
    .venv\\Scripts\\python.exe scripts\\04_train_yolo.py --model yolov8s.pt  --name yolov8s_aug  --aug default
    .venv\\Scripts\\python.exe scripts\\04_train_yolo.py --model yolov8n.pt --name yolov8n_960   --aug default --imgsz 960

参数说明（新手向）：
    --model   预训练权重名字。ultralytics 会自动从网上下载（约 6~50 MB）。
              n = nano（最小最快），s = small（更大更准但更慢）。
    --imgsz   训练时把图片缩放到多大。数据集原图是 512×512：
              512 不缩放；960 相当于放大，能让小目标占更多像素，但训练更慢、更吃显存。
    --epochs  训练轮数。一轮 = 把所有训练图片看一遍。太少欠拟合，太多容易过拟合。
    --batch   一次送进 GPU 的图片数。显存不够就调小（32 → 16 → 8）。
    --aug     数据增强强度：
              plain   = 完全不做几何/颜色增强，作为"干净基线"
              default = YOLO 默认增强（mosaic 拼图、HSV 颜色抖动、随机缩放平移、水平翻转）
              strong  = 在默认基础上再加 mixup / copy-paste / 小角度旋转

注意：训练集用于学习参数，验证集用于选模型和调超参数，测试集只在最后评估一次。
本脚本只会在验证集上跑 val，不会碰测试集。
"""

# 值得注意的是：
# 此处只是训练了五个模型，但是出现了7个对比项
# 包括：
# YOLOv8n + TTA ：TTA（Test-Time Augmentation）：推理时把图片翻转/缩放几次，分别预测，再合并结果。模型参数完全没变。
# YOLOv8n + 2×2 切图 ： 2×2 切图（tiled inference）：把 512×512 切成 4 块，每块放大后送进同一个模型，再把框映射回原图合并。模型参数也没变。
# 都是复用 （YOLOv8n + 增强）模型 的 best.pt （权重文件）
# 这样可以进行控制变量法进行比对操作，为报告提供数据支撑

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 把项目根目录加入 sys.path，保证无论从哪里运行都能 import src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA_ROOT, METRICS_DIR, MODELS_DIR, ensure_dirs  # noqa: E402


def build_aug_kwargs(level: str) -> dict:
    """根据增强等级返回 ultralytics 的超参数字典。

    这些参数直接对应数据增强（data augmentation）的开关：
      mosaic      把 4 张图拼成 1 张，迫使模型学习不同位置/尺度的小目标
      close_mosaic 最后 N 个 epoch 关闭 mosaic，让模型适应真实图片布局
      hsv_*       颜色抖动（色相/饱和度/亮度），模拟不同光照
      scale       随机缩放，模拟远近距离
      translate   随机平移
      fliplr      水平翻转概率（鸟没有固定朝向，翻转是安全的）
      mixup       两张图按透明度混合，增强遮挡鲁棒性
      copy_paste  把一个实例粘贴到另一张图，增加密集场景
    """
    if level == "plain":
        return dict(mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
                    degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.0,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0)
    if level == "default":
        # 大部分保持 ultralytics 默认值，显式写出便于报告复现
        return dict(mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
                    degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5,
                    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.0)
    if level == "strong":
        return dict(mosaic=1.0, close_mosaic=10, mixup=0.15, copy_paste=0.1,
                    degrees=10.0, translate=0.1, scale=0.6, shear=2.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5,
                    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, erasing=0.2)
    raise ValueError(f"未知增强等级: {level}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="预训练权重，如 yolov8n.pt / yolov8s.pt / yolo11n.pt")
    ap.add_argument("--name", required=True, help="本次实验名（结果保存到 runs/detect/<name>）")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aug", choices=["plain", "default", "strong"], default="default")
    ap.add_argument("--device", default="0", help="'0' 用第一块 GPU，'cpu' 用 CPU")
    ap.add_argument("--patience", type=int, default=15, help="验证集 mAP 连续 N 轮不升就早停")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()
    from ultralytics import YOLO

    # 数据集 YAML：ultralytics 用 path + train/val 相对路径定位图片。
    # 这里动态生成绝对路径版本，避免因工作目录不同而找不到数据。
    data_yaml = MODELS_DIR / "beach_bird_local.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {DATA_ROOT.as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\nnames:\n  0: bird\n")

    aug_kwargs = build_aug_kwargs(args.aug)
    print(f"[train] model={args.model} name={args.name} imgsz={args.imgsz} "
          f"epochs={args.epochs} batch={args.batch} aug={args.aug}")

    model = YOLO(args.model)
    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(MODELS_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        plots=True,
        save=True,
        val=True,
        **aug_kwargs,
    )
    train_seconds = time.time() - t0

    # 保存训练配置和耗时，后面生成结果表要用
    best = MODELS_DIR / "runs" / args.name / "weights" / "best.pt"
    meta = {
        "name": args.name, "model": args.model, "imgsz": args.imgsz,
        "epochs": args.epochs, "batch": args.batch, "aug": args.aug,
        "train_seconds": train_seconds, "best_weights": str(best),
        "train_minutes": train_seconds / 60.0,
    }
    with open(METRICS_DIR / f"train_{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] best weights: {best}")
    print(f"[done] training time: {train_seconds/60:.1f} min")


if __name__ == "__main__":
    main()
