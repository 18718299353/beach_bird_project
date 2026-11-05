"""
common.py —— 项目中所有方法共用的工具函数
=================================================

本文件只做"基础设施"工作，不包含任何检测算法本身：

1. 路径与数据集读取（把 YOLO 格式的 txt 标签读成 [x1, y1, x2, y2] 的绝对像素坐标）
2. 边界框（bounding box）运算：IoU、NMS、坐标裁剪
3. 统一的评估：mAP@0.5（COCO 风格）、P/R/F1、按目标尺寸分组、按鸟群密度分组、
   负样本（没有鸟的图片）误检统计、计数误差
4. 预测结果 JSON 的读写

为什么要自己写评估？
-------------------
作业要求 mAP@0.5，而不同方法（传统方法 / HOG+SVM / YOLO）输出的格式并不一样。
把评估逻辑集中在这里，可以保证**所有方法用完全相同的规则**计算指标，比较才公平。
另外，自己实现一遍 AP 的计算过程，比直接调用某个库更容易在报告中解释清楚。

术语速查（给第一次接触目标检测的同学）
--------------------------------------
- 边界框 box：用左上角 (x1, y1) 和右下角 (x2, y2) 表示的矩形。
- IoU（Intersection over Union，交并比）：两个框的交集面积 ÷ 并集面积。
  IoU = 1 表示完全重合，IoU = 0 表示完全不相交。
- TP / FP / FN：真正例 / 假正例 / 假负例。
  检测框与某个真实框 IoU ≥ 阈值，就算 TP；否则算 FP。没有被任何检测框匹配到的真实框算 FN。
- Precision（精确率）= TP / (TP + FP)：报出来的框里有多少是真的。
- Recall（召回率）   = TP / (TP + FN)：真实的鸟有多少被找到了。
- AP（Average Precision）：把置信度从高到低排序后，Precision-Recall 曲线下的面积。
  单类别时 mAP@0.5 就等于 AP@0.5（m 是 mean，对类别取平均；这里只有 bird 一类）。
- 置信度 score：模型对"这个框里是鸟"的把握，0~1 之间（传统方法用启发式分数代替）。
"""

# 此文件定义的是项目的基础通用功能
# 可以理解为一个工具包
# 具体看上面注释即可，无需过多关注
#
#

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
# PROJECT_ROOT 指向 beach_bird_project/（本文件在 src/ 下，所以 parents[1] 是项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "Beach Bird Detection Dataset v1.0"
IMAGES_DIR = DATA_ROOT / "images"
LABELS_DIR = DATA_ROOT / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

SPLITS = ("train", "val", "test")
# COCO 对"小 / 中 / 大目标"的定义（按像素面积，输入图片均为 512×512）：
#   small : area < 32²   = 1024 px²
#   medium: 32² ≤ area < 96² = 9216 px²
#   large : area ≥ 96²
SMALL_MAX_AREA = 32 * 32
MEDIUM_MAX_AREA = 96 * 96

# 鸟群密度分组（按一张图里真实框的数量）。用于分析"密集场景是否更容易漏检"。
DENSITY_BINS = [(0, 0, "0 (negative)"), (1, 5, "1-5 (sparse)"),
                (6, 20, "6-20 (moderate)"), (21, 50, "21-50 (dense)"),
                (51, 10**9, "51+ (very dense)")]


def ensure_dirs() -> None:
    """创建所有输出目录（重复调用不会报错）。"""
    for d in (RESULTS_DIR, PRED_DIR, METRICS_DIR, FIG_DIR, TABLE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """固定随机种子，让"随机采样负样本"等步骤可复现。"""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 数据集读取
# ---------------------------------------------------------------------------
@dataclass
class ImageRecord:
    """一张图片及其标注信息。

    Attributes
    ----------
    image_id : 文件名去掉扩展名，例如 "000123"。全项目用它作为唯一 ID。
    image_path / label_path : 图片和标签文件的绝对路径。
    width, height : 图片尺寸（本数据集全部是 512×512）。
    boxes : shape=(N, 4) 的 float 数组，每行是 [x1, y1, x2, y2]（绝对像素）。
            没有鸟时是 shape=(0, 4) 的空数组。
    """

    image_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    boxes: np.ndarray

    @property
    def num_objects(self) -> int:
        return int(self.boxes.shape[0])

    def load_image(self) -> np.ndarray:
        """读取 BGR 图片；用 cv2.IMREAD_COLOR 保证三通道。"""
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {self.image_path}")
        return img


def _read_image_size(path: Path) -> Tuple[int, int]:
    """只读图片头获取宽高，比完整解码快很多（遍历 1900 张图时很有用）。"""
    from PIL import Image

    with Image.open(path) as im:
        return im.width, im.height


def yolo_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """YOLO 归一化格式 -> 绝对像素 xyxy。

    YOLO 标签每行是: class_id cx cy w h
    其中 cx, cy, w, h 都是相对于图片宽高的比例（0~1）。
    这里假设只有一个类别，所以 class_id 被丢弃。
    """
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = boxes.astype(np.float32).copy()
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return np.stack([x1, y1, x2, y2], axis=1)


def load_yolo_labels(label_path: Path, width: int, height: int) -> np.ndarray:
    """读取一个 YOLO txt 标签文件，返回 (N,4) 的 xyxy 数组。空文件返回空数组。"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # parts[0] 是类别 id；本数据集只有 0=bird，不需要保留
            rows.append([float(v) for v in parts[1:5]])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return yolo_to_xyxy(np.asarray(rows, dtype=np.float32), width, height)


def load_split(split: str, with_size: bool = True) -> List[ImageRecord]:
    """读取 train / val / test 某个划分的全部图片记录。

    数据集已经是官方切分好的，我们只负责读取，**绝不重新划分**，
    以避免数据泄漏（data leakage）——这是作业明确强调的。
    """
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到 {split!r}")
    img_dir = IMAGES_DIR / split
    lbl_dir = LABELS_DIR / split
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    records: List[ImageRecord] = []
    for p in images:
        if with_size:
            w, h = _read_image_size(p)
        else:
            w = h = 512
        lbl = lbl_dir / (p.stem + ".txt")
        boxes = load_yolo_labels(lbl, w, h)
        records.append(ImageRecord(p.stem, p, lbl, w, h, boxes))
    return records


# ---------------------------------------------------------------------------
# 边界框基础运算
# ---------------------------------------------------------------------------
def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """把框裁剪到图片范围内，并保证 x2>x1, y2>y1。"""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    b = boxes.copy().astype(np.float32)
    b[:, 0::2] = np.clip(b[:, 0::2], 0, width - 1)
    b[:, 1::2] = np.clip(b[:, 1::2], 0, height - 1)
    # 防止出现宽度或高度为 0 的退化框
    b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1)
    b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1)
    return b


def box_areas(boxes: np.ndarray) -> np.ndarray:
    """每个框的面积（像素²）。"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两组框两两之间的 IoU，返回 shape=(len(a), len(b)) 的矩阵。

    向量化实现：先算交集矩形，再除以并集面积。
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    # 广播成 (N,1,2) 和 (1,M,2) 以便两两比较
    lt = np.maximum(a[:, None, :2], b[None, :, :2])   # 交集左上角
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])   # 交集右下角
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = box_areas(a)[:, None]
    area_b = box_areas(b)[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """非极大值抑制（Non-Maximum Suppression, NMS）。

    目的：同一个鸟附近往往会产生很多高分框，NMS 只保留其中最可信的一个，
    把与它 IoU 超过阈值的其他框删掉。

    参数
    ----
    boxes : (N,4) xyxy
    scores: (N,) 置信度
    iou_threshold : 越大保留得越多（越宽松）；越小删得越狠（越严格）。

    返回保留下来的下标（按分数从高到低）。
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    scores = scores.astype(np.float32)
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i][None, :], boxes[order[1:]])[0]
        # 只保留与当前框重叠很小的候选，进入下一轮
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   iou_threshold: float = 0.45) -> np.ndarray:
    """按类别分别做 NMS（本项目只有一类，保留此函数是为了代码通用性）。"""
    keep_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        keep = idx[nms(boxes[idx], scores[idx], iou_threshold)]
        keep_all.append(keep)
    return np.concatenate(keep_all) if keep_all else np.zeros((0,), dtype=np.int64)


# ---------------------------------------------------------------------------
# AP / mAP 计算（COCO 风格）
# ---------------------------------------------------------------------------
def average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    """由按置信度排序的 TP/FP 序列计算 AP。

    采用 COCO 的 101 点插值（等价于 PR 曲线的单调包络面积）：
    1. 累积得到 precision 和 recall；
    2. 从右往左取 precision 的 running maximum，消除 PR 曲线的锯齿；
    3. 对 recall 的每一小段求面积。
    """
    if num_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp).astype(np.float64)
    fp_cum = np.cumsum(fp).astype(np.float64)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # 在首尾补点，保证曲线从 recall=0 开始、到 recall=1 结束
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # precision 单调包络
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # 只在对 recall 有变化的区间积分
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def _match_image(gt: np.ndarray, det: np.ndarray, iou_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """对单张图片做贪心匹配。

    det 已按置信度从高到低排序，shape=(M,5)，最后一列是 score。
    返回两个布尔数组：det_matched (M,), gt_matched (N,)。
    """
    n_gt, n_det = gt.shape[0], det.shape[0]
    gt_matched = np.zeros(n_gt, dtype=bool)
    det_matched = np.zeros(n_det, dtype=bool)
    if n_gt == 0 or n_det == 0:
        return det_matched, gt_matched
    ious = box_iou(det[:, :4], gt)  # (M, N)
    for d in range(n_det):
        # 只在"还没被匹配"的真实框里找 IoU 最大的
        candidates = np.where(~gt_matched)[0]
        if candidates.size == 0:
            break
        best = candidates[np.argmax(ious[d, candidates])]
        if ious[d, best] >= iou_thr:
            det_matched[d] = True
            gt_matched[best] = True
    return det_matched, gt_matched


def _collect_detections(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                        gt_filter=None) -> Tuple[List[Tuple[float, str, int]], List[np.ndarray], int]:
    """把所有图片的检测框汇总成一个全局列表，按分数降序。

    返回:
      entries: [(score, image_id, det_index), ...] 已排序
      dets_by_image: {image_id: (M,5) array}
      num_gt: 参与评估的真实框总数
    """
    dets_by_image: Dict[str, np.ndarray] = {}
    entries: List[Tuple[float, str, int]] = []
    num_gt = 0
    for rec in records:
        gt = rec.boxes
        if gt_filter is not None:
            gt = gt[gt_filter(gt)]
        num_gt += gt.shape[0]
        det = preds.get(rec.image_id)
        if det is None or len(det) == 0:
            dets_by_image[rec.image_id] = np.zeros((0, 5), dtype=np.float32)
            continue
        det = np.asarray(det, dtype=np.float32).reshape(-1, 5)
        # 保证每张图内部按 score 降序（评估代码依赖这个顺序）
        det = det[np.argsort(-det[:, 4])]
        dets_by_image[rec.image_id] = det
        for j in range(det.shape[0]):
            entries.append((float(det[j, 4]), rec.image_id, j))
    entries.sort(key=lambda x: -x[0])
    return entries, dets_by_image, num_gt


def evaluate_ap(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                iou_threshold: float = 0.5, gt_filter=None) -> float:
    """计算某一 IoU 阈值下的 AP（单类别）。"""
    entries, dets_by_image, num_gt = _collect_detections(preds, records, gt_filter)
    if num_gt == 0:
        return float("nan")

    # 每张图的匹配状态；全局按分数从高到低依次"占用"真实框
    rec_by_id = {r.image_id: r for r in records}
    gt_matched_map: Dict[str, np.ndarray] = {}
    for rec in records:
        gt = rec.boxes
        if gt_filter is not None:
            gt = gt[gt_filter(gt)]
        gt_matched_map[rec.image_id] = np.zeros(gt.shape[0], dtype=bool)

    tp = np.zeros(len(entries), dtype=np.float32)
    fp = np.zeros(len(entries), dtype=np.float32)
    for k, (score, image_id, j) in enumerate(entries):
        rec = rec_by_id[image_id]
        gt = rec.boxes
        if gt_filter is not None:
            gt = gt[gt_filter(gt)]
        matched = gt_matched_map[image_id]
        if gt.shape[0] == 0:
            fp[k] = 1
            continue
        ious = box_iou(dets_by_image[image_id][j, :4][None, :], gt)[0]
        candidates = np.where(~matched)[0]
        if candidates.size and ious[candidates].max() >= iou_threshold:
            best = candidates[np.argmax(ious[candidates])]
            matched[best] = True
            tp[k] = 1
        else:
            fp[k] = 1
    return average_precision(tp, fp, num_gt)


def pr_curve(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
             iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """返回整条 Precision-Recall 曲线（用于画 PR 图）。

    与 evaluate_ap 使用完全相同的匹配规则，只是把曲线上的点返回出来。
    """
    entries, dets_by_image, num_gt = _collect_detections(preds, records)
    rec_by_id = {r.image_id: r for r in records}
    gt_matched_map = {r.image_id: np.zeros(r.boxes.shape[0], dtype=bool) for r in records}
    tp = np.zeros(len(entries), dtype=np.float32)
    fp = np.zeros(len(entries), dtype=np.float32)
    for k, (score, image_id, j) in enumerate(entries):
        rec = rec_by_id[image_id]
        gt = rec.boxes
        matched = gt_matched_map[image_id]
        if gt.shape[0] == 0:
            fp[k] = 1
            continue
        ious = box_iou(dets_by_image[image_id][j, :4][None, :], gt)[0]
        candidates = np.where(~matched)[0]
        if candidates.size and ious[candidates].max() >= iou_threshold:
            best = candidates[np.argmax(ious[candidates])]
            matched[best] = True
            tp[k] = 1
        else:
            fp[k] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return recall, precision


def evaluate_ap_range(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                      iou_thresholds: Iterable[float] = (0.5,), gt_filter=None) -> Dict[str, float]:
    """在多个 IoU 阈值上计算 AP，并返回 AP@0.5、AP@0.75、AP@[.5:.95]。"""
    thrs = list(iou_thresholds)
    aps = [evaluate_ap(preds, records, t, gt_filter) for t in thrs]
    out = {}
    for t, ap in zip(thrs, aps):
        out[f"AP@{t:.2f}"] = ap
    if thrs:
        out["AP@0.5"] = aps[0]
        # 0.75 不一定在列表里，单独算一次最稳妥
        out["AP@0.75"] = evaluate_ap(preds, records, 0.75, gt_filter)
        out["AP@0.5:0.95"] = float(np.nanmean(aps))
    return out


def prf_at_threshold(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                     score_threshold: float, iou_threshold: float = 0.5) -> Dict[str, float]:
    """在固定置信度阈值下计算 Precision / Recall / F1 / TP / FP / FN。"""
    entries, dets_by_image, num_gt = _collect_detections(preds, records)
    rec_by_id = {r.image_id: r for r in records}
    gt_matched_map = {rec.image_id: np.zeros(rec.boxes.shape[0], dtype=bool) for rec in records}
    tp = fp = fn = 0
    for score, image_id, j in entries:
        if score < score_threshold:
            continue
        rec = rec_by_id[image_id]
        gt = rec.boxes
        matched = gt_matched_map[image_id]
        if gt.shape[0] == 0:
            fp += 1
            continue
        ious = box_iou(dets_by_image[image_id][j, :4][None, :], gt)[0]
        candidates = np.where(~matched)[0]
        if candidates.size and ious[candidates].max() >= iou_threshold:
            best = candidates[np.argmax(ious[candidates])]
            matched[best] = True
            tp += 1
        else:
            fp += 1
    fn = int(sum(int((~gt_matched_map[r.image_id]).sum()) for r in records))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"threshold": float(score_threshold), "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "num_gt": int(num_gt)}


def select_operating_threshold(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                               iou_threshold: float = 0.5) -> Tuple[float, Dict[str, float]]:
    """在验证集上选择使 F1 最大的置信度阈值。

    注意：这个阈值只能在 val 上选，选定后原封不动地用于 test，
    绝不能在 test 上反复调参（否则测试结果会偏乐观）。
    """
    best_t, best = 0.5, None
    for t in np.linspace(0.01, 0.99, 99):
        m = prf_at_threshold(preds, records, float(t), iou_threshold)
        if best is None or m["f1"] > best["f1"]:
            best_t, best = float(t), m
    return best_t, best


def count_metrics(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                  score_threshold: float) -> Dict[str, float]:
    """鸟类计数（counting）指标：预测数量与真实数量的误差。

    - MAE：平均每张图数错几只（越小越好）
    - bias：平均有符号误差，正数表示整体多报了，负数表示漏报了
    - rel_mae：MAE / 平均真实数量
    """
    gt_counts, pred_counts = [], []
    for rec in records:
        det = preds.get(rec.image_id, np.zeros((0, 5)))
        pred_counts.append(int((det[:, 4] >= score_threshold).sum()) if len(det) else 0)
        gt_counts.append(rec.num_objects)
    gt = np.asarray(gt_counts, dtype=np.float32)
    pr = np.asarray(pred_counts, dtype=np.float32)
    err = pr - gt
    return {"count_mae": float(np.mean(np.abs(err))),
            "count_bias": float(np.mean(err)),
            "count_rel_mae": float(np.mean(np.abs(err)) / max(np.mean(gt), 1e-9)),
            "mean_gt_count": float(np.mean(gt)),
            "mean_pred_count": float(np.mean(pr))}


def negative_scene_fp(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                      score_threshold: float) -> Dict[str, float]:
    """统计"没有鸟的负样本图片"上的误检（false positive）。

    这是检测器在实际生态监测中非常关键的指标：把石头、浪花、垃圾认成鸟，
    会让后续的鸟群计数严重偏高。
    """
    neg = [r for r in records if r.num_objects == 0]
    if not neg:
        return {"neg_images": 0, "neg_fp_total": 0, "neg_fp_per_image": 0.0,
                "neg_images_with_fp": 0, "neg_fp_rate": 0.0}
    fps = []
    for r in neg:
        det = preds.get(r.image_id, np.zeros((0, 5)))
        n = int((det[:, 4] >= score_threshold).sum()) if len(det) else 0
        fps.append(n)
    fps_arr = np.asarray(fps, dtype=np.float32)
    return {"neg_images": len(neg), "neg_fp_total": int(fps_arr.sum()),
            "neg_fp_per_image": float(fps_arr.mean()),
            "neg_images_with_fp": int((fps_arr > 0).sum()),
            "neg_fp_rate": float((fps_arr > 0).mean())}


def density_breakdown(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                      score_threshold: float, iou_threshold: float = 0.5) -> List[Dict]:
    """按真实框数量把测试图片分组，分别计算 AP / P / R / F1 / 计数误差。

    回答作业里的问题："哪种方法在 Dense Scenes 中更容易漏检？"
    """
    rows = []
    for lo, hi, name in DENSITY_BINS:
        subset = [r for r in records if lo <= r.num_objects <= hi]
        if not subset:
            continue
        ap = evaluate_ap(preds, subset, iou_threshold)
        m = prf_at_threshold(preds, subset, score_threshold, iou_threshold)
        cm = count_metrics(preds, subset, score_threshold)
        rows.append({"group": name, "num_images": len(subset),
                     "num_gt": int(sum(r.num_objects for r in subset)),
                     "AP@0.5": ap, "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "count_mae": cm["count_mae"],
                     "count_bias": cm["count_bias"]})
    return rows


def size_breakdown(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                   score_threshold: float, iou_threshold: float = 0.5) -> List[Dict]:
    """按真实框面积分组（COCO 的 small / medium / large）计算 AP 与召回率。"""
    defs = [("small (<32² px)", lambda b: box_areas(b) < SMALL_MAX_AREA),
            ("medium (32²-96² px)", lambda b: (box_areas(b) >= SMALL_MAX_AREA) &
                                             (box_areas(b) < MEDIUM_MAX_AREA)),
            ("large (≥96² px)", lambda b: box_areas(b) >= MEDIUM_MAX_AREA)]
    rows = []
    for name, filt in defs:
        n_gt = int(sum(int(filt(r.boxes).sum()) for r in records))
        if n_gt == 0:
            continue
        ap = evaluate_ap(preds, records, iou_threshold, gt_filter=filt)
        # 该尺寸组在固定阈值下的召回率：统计被 TP 覆盖的该尺寸 GT 数
        tp = fn = 0
        for rec in records:
            gt_mask = filt(rec.boxes)
            gt = rec.boxes[gt_mask]
            if gt.shape[0] == 0:
                continue
            det = preds.get(rec.image_id, np.zeros((0, 5)))
            if len(det):
                det = det[det[:, 4] >= score_threshold]
            matched = np.zeros(gt.shape[0], dtype=bool)
            if len(det):
                ious = box_iou(det[:, :4], gt)
                # 按分数从高到低贪心匹配
                for d in range(len(det)):
                    cand = np.where(~matched)[0]
                    if cand.size and ious[d, cand].max() >= iou_threshold:
                        matched[cand[np.argmax(ious[d, cand])]] = True
            tp += int(matched.sum())
            fn += int((~matched).sum())
        rows.append({"group": name, "num_gt": n_gt, "AP@0.5": ap,
                     "recall@thr": tp / max(tp + fn, 1)})
    return rows


# ---------------------------------------------------------------------------
# 预测结果持久化
# ---------------------------------------------------------------------------
def save_predictions(path: Path, preds: Dict[str, np.ndarray], meta: Optional[Dict] = None) -> None:
    """把预测结果保存成 JSON，方便不同脚本之间传递。

    结构：
    {
      "meta": {...},                       # 方法名、运行时间等
      "predictions": {
         "image_id": [[x1,y1,x2,y2,score], ...],
         ...
      }
    }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "predictions": {k: np.asarray(v, dtype=float).round(2).tolist()
                        for k, v in preds.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_predictions(path: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    """读取 save_predictions 保存的 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    preds = {k: np.asarray(v, dtype=np.float32).reshape(-1, 5)
             for k, v in payload.get("predictions", {}).items()}
    return preds, payload.get("meta", {})


def load_image_bgr(rec: ImageRecord) -> np.ndarray:
    """便捷函数：按记录读取图片。"""
    return rec.load_image()


def draw_boxes(image: np.ndarray, boxes: np.ndarray, color=(0, 255, 0),
               thickness: int = 2, scores: Optional[np.ndarray] = None) -> np.ndarray:
    """在图片副本上画框，返回新图片（不修改原图）。"""
    out = image.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if scores is not None:
            s = float(scores[i])
            cv2.putText(out, f"{s:.2f}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def summarize_predictions(preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    """输出预测框数量的简单统计，用于日志。"""
    counts = [len(v) for v in preds.values()]
    return {"num_images": len(counts), "total_boxes": int(sum(counts)),
            "mean_boxes": float(np.mean(counts)) if counts else 0.0}
