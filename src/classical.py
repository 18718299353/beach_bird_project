"""
classical.py —— 非学习（non-learning）传统图像处理方法
=========================================================

这一类方法不训练任何模型，完全依靠手工设计的图像处理流程。
我们实现并比较三个变体，对应报告中的 "Category A: Non-learning methods"：

1. global_otsu
   最朴素的基线：灰度图 + 全局 Otsu 阈值（把比背景暗的像素当成鸟）+ 形态学去噪
   + 连通域找轮廓。它假设"鸟一定比背景暗"，在沙滩/水面上偶尔有效，
   但遇到深色礁石、阴影、白色海鸥就会大量误检/漏检。

2. multiscale_blackhat
   多尺度形态学"黑帽/顶帽"局部对比度检测：
     black-hat = closing(image) - image  → 突出"比周围暗的小目标"
     top-hat   = image - opening(image)  → 突出"比周围亮的小目标"
   对多个结构元素尺寸分别计算，取二者较大值，再用 Otsu 二值化。
   这样对深色鸟和白色鸟都敏感，且不受整体光照影响。

3. multiscale_blackhat_ws
   在 2 的基础上，对面积过大的连通域做**分水岭分割（watershed）**，
   把挤在一起的多只鸟拆成多个框。这是针对密集鸟群（dense flock）的设计。

为什么传统方法值得做？
----------------------
它是理解"检测"本质的最好起点：先产生候选区域，再想办法过滤。
同时它也提供了一个诚实的下界（lower bound），让我们看到深度学习方法
到底带来了多大的提升，以及提升来自哪里（更好的特征？更大的感受野？）。

重要概念
--------
- 结构元素（structuring element, SE）：形态学操作的"小刷子"。
  尺寸 k 表示只关心比 k×k 小的亮/暗结构；k 越大，能检测的目标越大，
  但也会把大块背景差异当成目标。
- 连通域（connected component）：二值图里连成一片的像素集合。
  一个连通域通常对应一个候选目标。
- 分水岭（watershed）：把"地形"（这里是距离变换）中的每个山谷/山峰
  分给不同的标记，从而把粘连的多个目标分开。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (PRED_DIR, METRICS_DIR, ImageRecord, box_iou, clip_boxes,
                        ensure_dirs, evaluate_ap, load_split, nms, prf_at_threshold,
                        save_predictions, select_operating_threshold)


@dataclass
class ClassicalParams:
    """传统方法的所有可调参数（集中放在这里，方便在验证集上做网格搜索）。

    新手提示：这些数字不是"真理"，而是根据数据统计 + 验证集实验选出来的。
    改大/改小会怎样，见每个字段后面的注释。
    """
    # --- 候选框过滤 ---
    min_area: float = 80.0        # 小于这么多像素² 的连通域丢掉（滤掉噪声点）
    max_area: float = 25000.0     # 大于这么多的丢掉（整片天空/沙滩/礁石不是一只鸟）
    min_fill: float = 0.30        # 连通域面积 / 外接矩形面积；太细长的形状丢掉
    min_aspect: float = 0.22      # 宽/高 的最小值；鸟不会极端细长
    max_aspect: float = 3.5       # 宽/高 的最大值（浪花、海岸线通常是细长条）
    # --- 多尺度形态学 ---
    se_sizes: Tuple[int, ...] = (7, 11, 17, 25, 35, 51)  # 结构元素边长（奇数）
    resp_floor: float = 8.0        # 响应阈值下限，避免 Otsu 在平坦区域选到 0
    otsu_scale: float = 1.0        # 在 Otsu 阈值上再乘一个系数（>1 更严格）
    open_ksize: int = 3            # 二值图开运算核大小（去掉孤立噪点）
    close_ksize: int = 3           # 闭运算核大小（填补目标内部空洞；太大会把两只鸟粘在一起）
    # --- 尺度一致性 ---
    # 一个尺寸为 size 的连通域，只应该由"结构元素大小与之匹配"的那一尺度检测到。
    # 如果不加这个约束，大块礁石会在小尺度上也产生强响应，造成大量误检。
    se_size_min_ratio: float = 0.12  # 连通域边长 / 结构元素边长 的下限
    se_size_max_ratio: float = 1.15  # 上限
    # --- 分水岭 ---
    ws_min_distance: int = 4       # 距离变换峰值之间的最小间距（越小越容易过度分割）
    ws_threshold_abs: float = 1.5  # 距离变换峰值的最小高度
    split_area: float = 2500.0     # 连通域大于该面积才尝试分水岭拆分
    max_split_area: float = 12000.0  # 超过该面积基本是礁石/岸线，不拆
    max_split_peaks: int = 24      # 一个连通域最多拆成多少个目标（防止爆炸）
    # --- 输出 ---
    min_score: float = 0.02        # 低于该分数的候选直接丢弃，减少 JSON 体积
    nms_iou: float = 0.35          # 跨尺度合并时的 NMS IoU 阈值


def _ellipse(k: int) -> np.ndarray:
    """生成 k×k 的椭圆形结构元素（k 必须是正奇数）。"""
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _response_maps(gray: np.ndarray, se_sizes: Sequence[int]) -> List[Tuple[int, np.ndarray]]:
    """对每个尺度计算局部对比度响应图。

    resp = max(black-hat, top-hat)
    - black-hat 对"暗目标"响应高（深色鸟在亮沙滩上）
    - top-hat   对"亮目标"响应高（白鸟在深色水面/礁石上）
    """
    maps = []
    for k in se_sizes:
        se = _ellipse(k)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
        resp = cv2.max(blackhat, tophat)
        # 轻度高斯模糊，抑制单个像素的噪声（水面反光）
        resp = cv2.GaussianBlur(resp, (3, 3), 0)
        maps.append((k, resp))
    return maps


def _bird_score(peak: float, mean: float, area: float, fill: float,
                p: ClassicalParams) -> float:
    """给一个候选框打"像鸟"的分数，用于 mAP 排序和阈值筛选。

    分数由三部分组成（都乘在一起，范围大致 0~1）：
      1. 对比度：peak/mean 是该区域在响应图上的强度；
      2. 尺寸先验：数据集中鸟框边长中位数约 42 px。太小的多半是噪声，
         太大的多半是礁石/岸线，用对数正态形状的权重压低；
      3. 形状先验：填充率越高越像完整的一只鸟。

    注意：这里的常数（42、0.85 等）来自训练集统计，
    不是从测试集"偷看"来的。
    """
    side = float(np.sqrt(max(area, 1.0)))
    # 对数正态尺寸先验：side=42 时权重=1，偏离越远权重越小
    size_prior = float(np.exp(-0.5 * (np.log(max(side, 1.0) / 42.0) / 0.85) ** 2))
    shape_prior = 0.5 + 0.5 * min(fill / 0.7, 1.0)
    contrast = 0.75 * peak + 0.25 * mean
    return float(contrast * (0.35 + 0.65 * size_prior) * shape_prior)


def _boxes_from_binary(binary: np.ndarray, resp: np.ndarray, se_size: Optional[int],
                       p: ClassicalParams) -> List[Tuple[np.ndarray, float]]:
    """把二值 mask 的连通域转换成 (box, score) 列表。

    过滤条件：
      - 面积在 [min_area, max_area]
      - 外接矩形的填充率 >= min_fill（鸟大致是实心的，树枝/浪花往往很细）
      - 长宽比合理（浪花/岸线通常是非常扁的细长条）
      - 尺度一致性：连通域的大小要和当前结构元素大小匹配
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: List[Tuple[np.ndarray, float]] = []
    for i in range(1, num):  # 0 是背景
        x, y, w, h, area = stats[i]
        if area < p.min_area or area > p.max_area:
            continue
        if w <= 1 or h <= 1:
            continue
        fill = area / float(w * h)
        aspect = w / float(h)
        if fill < p.min_fill or not (p.min_aspect <= aspect <= p.max_aspect):
            continue
        # 尺度一致性：sqrt(area) 应该和结构元素边长处于同一数量级
        if se_size is not None:
            side_ratio = np.sqrt(area) / float(se_size)
            if not (p.se_size_min_ratio <= side_ratio <= p.se_size_max_ratio):
                continue
        comp = labels[y:y + h, x:x + w] == i
        r = resp[y:y + h, x:x + w][comp]
        if r.size == 0:
            continue
        score = _bird_score(float(r.max()) / 255.0, float(r.mean()) / 255.0,
                            area, fill, p)
        if score < p.min_score:
            continue
        out.append((np.array([x, y, x + w, y + h], dtype=np.float32), score))
    return out


def _split_component_watershed(gray: np.ndarray, resp: np.ndarray, label_mask: np.ndarray,
                               p: ClassicalParams) -> List[Tuple[np.ndarray, float]]:
    """对一个面积较大的连通域做分水岭拆分。

    思路：
      1. 只在该连通域的 ROI 内计算距离变换（离背景越远，值越大）；
      2. 找距离变换的局部峰值，每个峰当作一只鸟的"种子"；
      3. 用这些种子做 watershed，把粘连区域切开；
      4. 每个切出来的区域再生成一个框。
    如果峰值太少（比如只有 1 个），就原样返回一个框。
    """
    ys, xs = np.where(label_mask)
    if len(xs) == 0:
        return []
    area = len(xs)
    # 面积太大基本是礁石/岸线，强行拆分只会产生大量误检
    if area > p.max_split_area:
        return []
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    roi_mask = label_mask[y0:y1, x0:x1].astype(np.uint8)
    roi_resp = resp[y0:y1, x0:x1]
    fill = area / float(max((x1 - x0) * (y1 - y0), 1))
    if fill < 0.4:  # 稀疏/细长的区域不是一群鸟
        return []

    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    # 局部峰值：距离变换中"离背景最远"的点
    from skimage.feature import peak_local_max
    coords = peak_local_max(dist, min_distance=p.ws_min_distance,
                            threshold_abs=p.ws_threshold_abs, labels=roi_mask)
    if len(coords) > p.max_split_peaks:
        return []
    if len(coords) <= 1:
        bw = np.where(roi_mask > 0, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((np.array([x + x0, y + y0, x + x0 + w, y + y0 + h], np.float32),
                          float(roi_resp[roi_mask > 0].max()) / 255.0))
        return boxes

    markers = np.zeros(roi_mask.shape, dtype=np.int32)
    for idx, (yy, xx) in enumerate(coords, start=1):
        markers[yy, xx] = idx
    # watershed 需要三通道图像作为"地形"
    roi_gray = gray[y0:y1, x0:x1]
    color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    out = []
    for lab in range(1, len(coords) + 1):
        m = (markers == lab) & (roi_mask > 0)
        if m.sum() < p.min_area:
            continue
        yy, xx = np.where(m)
        bx1, bx2 = int(xx.min()) + x0, int(xx.max()) + 1 + x0
        by1, by2 = int(yy.min()) + y0, int(yy.max()) + 1 + y0
        r = roi_resp[m]
        w_box, h_box = bx2 - bx1, by2 - by1
        fill_box = m.sum() / float(max(w_box * h_box, 1))
        score = _bird_score(float(r.max()) / 255.0, float(r.mean()) / 255.0,
                            float(m.sum()), fill_box, p)
        out.append((np.array([bx1, by1, bx2, by2], np.float32), score))
    return out


def detect_image(img: np.ndarray, variant: str = "multiscale_blackhat_ws",
                 params: Optional[ClassicalParams] = None) -> np.ndarray:
    """对单张 BGR 图片做检测，返回 shape=(N,5) 的 [x1,y1,x2,y2,score]。"""
    p = params or ClassicalParams()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 统一的局部对比度图（只用于打分，不参与所有变体的二值化）
    global_resp = cv2.absdiff(gray, cv2.medianBlur(gray, 31))

    candidates: List[Tuple[np.ndarray, float]] = []

    if variant == "global_otsu":
        # 全局 Otsu：假设鸟比背景暗，所以用 THRESH_BINARY_INV
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k_open = _ellipse(p.open_ksize)
        k_close = _ellipse(p.close_ksize)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        if p.min_area > 1:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _ellipse(3))
        candidates = _boxes_from_binary(binary, global_resp, None, p)

    elif variant in ("multiscale_blackhat", "multiscale_blackhat_ws"):
        all_parts: List[Tuple[np.ndarray, float]] = []
        for k, resp in _response_maps(gray, p.se_sizes):
            # 每个尺度单独用 Otsu 选阈值；resp_floor 防止在完全平坦的区域选到 0
            otsu_thr, _ = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = max(p.resp_floor, otsu_thr * p.otsu_scale)
            binary = (resp >= thr).astype(np.uint8) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _ellipse(p.open_ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ellipse(p.close_ksize))

            if variant == "multiscale_blackhat_ws":
                # 先按普通连通域处理；对过大的连通域再拆分
                num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                for i in range(1, num):
                    x, y, w, h, area = stats[i]
                    if area < p.min_area or area > p.max_area:
                        continue
                    if area >= p.split_area:
                        comp_mask = labels == i
                        parts = _split_component_watershed(gray, resp, comp_mask, p)
                        # 拆分失败（例如只有一个峰）时退回整块
                        if len(parts) >= 2:
                            all_parts.extend(parts)
                            continue
                    # 普通小连通域：做和 _boxes_from_binary 相同的形状/尺度过滤
                    fill = area / float(max(w * h, 1))
                    aspect = w / float(h)
                    if fill < p.min_fill or not (p.min_aspect <= aspect <= p.max_aspect):
                        continue
                    side_ratio = np.sqrt(area) / float(k)
                    if not (p.se_size_min_ratio <= side_ratio <= p.se_size_max_ratio):
                        continue
                    comp = labels[y:y + h, x:x + w] == i
                    r = resp[y:y + h, x:x + w][comp]
                    if r.size == 0:
                        continue
                    score = _bird_score(float(r.max()) / 255.0, float(r.mean()) / 255.0,
                                        area, fill, p)
                    if score >= p.min_score:
                        all_parts.append((np.array([x, y, x + w, y + h], np.float32), score))
            else:
                all_parts.extend(_boxes_from_binary(binary, resp, k, p))

        # 不同尺度会对同一只鸟产生多个框，用 NMS 合并
        if all_parts:
            boxes = np.stack([b for b, _ in all_parts])
            scores = np.array([s for _, s in all_parts], dtype=np.float32)
            keep = nms(boxes, scores, p.nms_iou)
            candidates = [(boxes[i], float(scores[i])) for i in keep]
    else:
        raise ValueError(f"未知 variant: {variant}")

    if not candidates:
        return np.zeros((0, 5), dtype=np.float32)

    boxes = np.stack([b for b, _ in candidates]).astype(np.float32)
    scores = np.array([s for _, s in candidates], dtype=np.float32)
    h, w = img.shape[:2]
    boxes = clip_boxes(boxes, w, h)
    return np.concatenate([boxes, scores[:, None]], axis=1)


def run_split(split: str, variant: str, params: Optional[ClassicalParams] = None,
              limit: Optional[int] = None, save_vis: int = 0) -> Dict[str, np.ndarray]:
    """在整个 split 上运行传统方法，返回预测字典，并保存 JSON。"""
    p = params or ClassicalParams()
    records = load_split(split)
    if limit:
        records = records[:limit]
    preds: Dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, rec in enumerate(records, start=1):
        img = rec.load_image()
        det = detect_image(img, variant, p)
        preds[rec.image_id] = det
        if i % 50 == 0:
            print(f"  [{variant}/{split}] {i}/{len(records)} images, "
                  f"{sum(len(v) for v in preds.values())} boxes", flush=True)
    elapsed = time.time() - t0

    meta = {"method": f"classical_{variant}", "split": split, "variant": variant,
            "params": asdict(p), "inference_seconds": elapsed,
            "ms_per_image": 1000 * elapsed / max(len(records), 1),
            "num_images": len(records)}
    out = PRED_DIR / f"classical_{variant}_{split}.json"
    save_predictions(out, preds, meta)

    ap = evaluate_ap(preds, records, 0.5)
    thr, best = select_operating_threshold(preds, records)
    print(f"[{variant}/{split}] AP@0.5={ap:.4f}  best_F1={best['f1']:.3f} "
          f"(thr={thr:.2f}, P={best['precision']:.3f}, R={best['recall']:.3f})  "
          f"{meta['ms_per_image']:.1f} ms/img")
    with open(METRICS_DIR / f"classical_{variant}_{split}.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "AP@0.5": ap, "best": best, "best_threshold": thr},
                  f, indent=2)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--variant", default="multiscale_blackhat_ws",
                    choices=["global_otsu", "multiscale_blackhat", "multiscale_blackhat_ws"])
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 张（调试用）")
    ap.add_argument("--min-area", type=float, default=60.0)
    ap.add_argument("--max-area", type=float, default=80000.0)
    ap.add_argument("--otsu-scale", type=float, default=1.0)
    ap.add_argument("--split-area", type=float, default=2200.0)
    args = ap.parse_args()

    ensure_dirs()
    p = ClassicalParams(min_area=args.min_area, max_area=args.max_area,
                        otsu_scale=args.otsu_scale, split_area=args.split_area)
    run_split(args.split, args.variant, p, args.limit)


if __name__ == "__main__":
    main()
