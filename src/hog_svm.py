"""
hog_svm.py —— 机器学习方法：HOG / LBP 手工特征 + 滑窗 + SVM/RandomForest
=========================================================================

这是作业提示里给出的经典流程：
    输入图片 → 多尺度滑动窗口 → HOG 特征 → SVM 分类 → 候选框 → NMS

与深度学习方法不同，这里**没有端到端学习"哪里是鸟"**：
- "哪里可能有鸟"由滑窗穷举决定（每 16 像素放一个 64×64 的窗口）；
- 分类器只回答"这个固定大小的窗口里是不是鸟"。

核心组件
--------
1. 特征（feature）
   - HOG（Histogram of Oriented Gradients，方向梯度直方图）：
     把窗口切成 8×8 的小格子（cell），统计每个格子里梯度的方向分布，
     再把相邻 2×2 个格子组成 block 做归一化。它对光照变化比较鲁棒，
     是行人检测（Dalal & Triggs, 2005）的经典特征。
   - LBP（Local Binary Pattern，局部二值模式）：
     比较每个像素与周围 8 个像素的亮度，得到一个 8 位二进制数，
     再统计直方图。它描述的是"纹理"，对单调光照变化不敏感。
2. 分类器（classifier）
   - LinearSVC：寻找一个线性超平面，把正/负样本分开，速度快。
   - RandomForest：许多决策树投票，能拟合非线性边界，但模型更大、更慢。
3. 滑窗（sliding window）
   在图像的多个位置、多个尺度上裁剪固定大小的窗口。
   窗口越大，能检测的鸟越大；步长（stride）越小，定位越准但计算越慢。
4. NMS
   同一个鸟会被相邻窗口重复检测，用 NMS 合并。

为什么这个方法值得做？
----------------------
它让我们能定量回答作业里的问题：
- 窗口大小 / 步长如何影响精度和运行时间？
- HOG 与 HOG+LBP 哪个特征更好？
- 传统特征能否区分鸟和背景？
并且它和 YOLO 形成鲜明对比：一个靠"穷举 + 手工特征"，一个靠"学习候选区域"。
"""

# Hog + svm分类方法
# 详情看上面注释
#
#

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import box_iou, clip_boxes, nms


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
@dataclass
class HogParams:
    """滑窗检测的全部参数。

    新手最常调的几个：
      window_sizes : 窗口边长列表（像素）。越大越能检测大鸟，但小目标会被漏掉。
      stride       : 滑窗步长（像素）。16 表示每隔 16 像素取一个窗口。
                     调小 → 更密 → 召回率上升、速度变慢；调大则相反。
      nms_iou      : NMS 阈值。调大保留更多重叠框（密集场景有利），但重复框变多。
      score_threshold : 推理时低于该分数的框直接丢掉（在 val 上选）。
    """
    window_sizes: Tuple[int, ...] = (32, 48, 64, 96, 128)
    stride: int = 16
    patch_size: int = 64          # 所有窗口统一缩放到 64×64 再提特征
    context: float = 1.3          # 训练正样本时，在 GT 框外扩多少倍（模拟窗口里的上下文）
    nms_iou: float = 0.40
    score_threshold: float = -0.6  # LinearSVC 的 decision_function 阈值（负值也可能保留）
    max_detections: int = 300
    max_candidates: int = 1200     # NMS 之前最多保留多少个高分候选（加速）
    # 不同窗口大小的分类器分数尺度不一样：大窗口看到的上下文更多，容易给出高分。
    # 用 score - penalty*log2(size/64) 做尺度校准，可以抑制"大框吞小目标"。
    size_score_penalty: float = 0.0
    containment_prune: bool = True   # 去掉"包含了一个可信小框"的膨胀大框
    # 训练正样本时 GT 框只占 patch 的 1/context（默认 1/1.3 ≈ 0.77）。
    # 推理时窗口就是 patch，所以输出框要按同样比例向中心收缩，
    # 否则框会系统性偏大 ~30%，IoU 很难达到 0.5。
    box_shrink: float = 0.77
    # 特征开关
    use_hog: bool = True
    use_lbp: bool = False
    # 训练参数
    neg_per_image: int = 16
    jitter_per_box: int = 1
    hard_negatives: int = 0       # 由训练脚本动态设置


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """把一张 BGR patch 转成特征向量。

    注意：HOGDescriptor 的构造比较耗时，所以只在初始化时建一次。
    """

    def __init__(self, patch_size: int = 64, use_hog: bool = True, use_lbp: bool = False):
        self.patch_size = patch_size
        self.use_hog = use_hog
        self.use_lbp = use_lbp
        if use_hog:
            # winSize=64×64, blockSize=16×16, blockStride=8×8, cellSize=8×8, nbins=9
            # 输出维度 = 7×7 blocks × 2×2 cells × 9 bins = 1764
            self.hog = cv2.HOGDescriptor(
                (patch_size, patch_size), (16, 16), (8, 8), (8, 8), 9)
        if use_lbp:
            from skimage.feature import local_binary_pattern  # noqa: F401
            self._lbp = local_binary_pattern

    @property
    def dim(self) -> int:
        d = 0
        if self.use_hog:
            d += 1764
        if self.use_lbp:
            d += 4 * 4 * 10  # 4×4 网格，每格 10 维 uniform LBP 直方图
        return d

    def __call__(self, patch_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY) if patch_bgr.ndim == 3 else patch_bgr
        gray = cv2.resize(gray, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        feats = []
        if self.use_hog:
            feats.append(self.hog.compute(gray).ravel().astype(np.float32))
        if self.use_lbp:
            lbp = self._lbp(gray, P=8, R=1, method="uniform")
            # 4×4 网格分别统计直方图，保留一定的空间信息
            h, w = lbp.shape
            gx, gy = w // 4, h // 4
            hist = []
            for i in range(4):
                for j in range(4):
                    cell = lbp[i * gy:(i + 1) * gy, j * gx:(j + 1) * gx]
                    hh, _ = np.histogram(cell, bins=np.arange(0, 11), density=True)
                    hist.append(hh)
            feats.append(np.concatenate(hist).astype(np.float32))
        return np.concatenate(feats)

    def batch(self, patches: Sequence[np.ndarray]) -> np.ndarray:
        if not patches:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self(p) for p in patches]).astype(np.float32)


# ---------------------------------------------------------------------------
# patch 裁剪
# ---------------------------------------------------------------------------
def crop_square(img: np.ndarray, cx: float, cy: float, side: float,
                out_size: int = 64) -> np.ndarray:
    """以 (cx,cy) 为中心裁一个边长为 side 的正方形，并缩放到 out_size。

    超出图片边界的部分用边缘像素复制（BORDER_REPLICATE），
    这样位于画面边缘的鸟也能得到完整 patch。
    """
    side = max(float(side), 4.0)
    h, w = img.shape[:2]
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if crop.shape[0] > out_size else cv2.INTER_LINEAR
    crop = cv2.resize(crop, (out_size, out_size), interpolation=interp)
    return crop


def crop_box(img: np.ndarray, box: np.ndarray, context: float = 1.3,
             out_size: int = 64) -> np.ndarray:
    """按 GT 框裁剪正样本 patch：以框中心为中心，边长为 max(w,h)*context。"""
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * context
    return crop_square(img, cx, cy, side, out_size)


def sample_negative_patch(img: np.ndarray, gt_boxes: np.ndarray, rng: np.random.Generator,
                          patch_size: int = 64) -> Optional[np.ndarray]:
    """随机采一个"背景"patch：要求它和所有 GT 框的 IoU 都很小。

    采样 30 次仍找不到合适位置就返回 None（密集图片里很常见）。
    """
    h, w = img.shape[:2]
    if gt_boxes is None:
        gt_boxes = np.zeros((0, 4), np.float32)
    for _ in range(30):
        side = float(rng.uniform(30, 140))
        cx = float(rng.uniform(0, w))
        cy = float(rng.uniform(0, h))
        box = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], np.float32)
        if gt_boxes.shape[0]:
            ious = box_iou(box[None, :], gt_boxes)[0]
            if ious.max() > 0.05:
                continue
        return crop_square(img, cx, cy, side, patch_size)
    return None


# ---------------------------------------------------------------------------
# 滑动窗口
# ---------------------------------------------------------------------------
def sliding_windows(width: int, height: int, sizes: Sequence[int], stride: int):
    """生成所有窗口的左上角坐标和边长。

    返回 (x, y, size)。窗口会覆盖到图片最右/最下边缘（即使不是 stride 的整数倍）。
    """
    for size in sizes:
        if size > width or size > height:
            continue
        xs = list(range(0, width - size + 1, stride))
        ys = list(range(0, height - size + 1, stride))
        if xs[-1] != width - size:
            xs.append(width - size)
        if ys[-1] != height - size:
            ys.append(height - size)
        for y in ys:
            for x in xs:
                yield x, y, size


@dataclass
class WindowDetector:
    """封装"滑窗 + 特征 + 分类器 + NMS"的推理器。"""

    model: object                  # sklearn Pipeline
    extractor: FeatureExtractor
    params: HogParams = field(default_factory=HogParams)

    def detect(self, img: np.ndarray) -> np.ndarray:
        """对单张图返回 (N,5) 的 [x1,y1,x2,y2,score]。"""
        h, w = img.shape[:2]
        boxes, scores = [], []
        patches = []          # 当前批次待提特征的 patch
        meta = []             # 与 patch 对应的框

        def flush():
            """把当前批次的 patch 提特征、打分，并收集候选。"""
            if not patches:
                return
            X = self.extractor.batch(patches)
            if hasattr(self.model, "decision_function"):
                s = self.model.decision_function(X)
            else:
                s = self.model.predict_proba(X)[:, 1]
            for (x, y, size), sc in zip(meta, s):
                # 尺度校准：大窗口的分数系统性地偏高，减去一个与 log(size) 成正比的量
                sc = float(sc) - self.params.size_score_penalty * np.log2(size / 64.0)
                # 先粗筛：低于阈值的窗口直接丢弃，减少后续 NMS 的量
                if sc >= self.params.score_threshold:
                    boxes.append([x, y, x + size, y + size])
                    scores.append(sc)
            patches.clear()
            meta.clear()

        for x, y, size in sliding_windows(w, h, self.params.window_sizes, self.params.stride):
            patches.append(crop_square(img, x + size / 2, y + size / 2, size,
                                       self.extractor.patch_size))
            meta.append((x, y, size))
            if len(patches) >= 512:
                flush()
        flush()

        if not boxes:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = np.asarray(boxes, np.float32)
        scores = np.asarray(scores, np.float32)

        # 关键后处理：把窗口框按 box_shrink 向中心收缩，补偿训练时的上下文外扩
        if self.params.box_shrink != 1.0:
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            half_w = (boxes[:, 2] - boxes[:, 0]) * self.params.box_shrink / 2
            half_h = (boxes[:, 3] - boxes[:, 1]) * self.params.box_shrink / 2
            boxes = np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)

        # 候选太多时只保留分数最高的一批，NMS 的复杂度是 O(K²)，这样能显著加速
        if len(scores) > self.params.max_candidates:
            top = np.argsort(-scores)[: self.params.max_candidates]
            boxes, scores = boxes[top], scores[top]

        if self.params.nms_iou >= 1.0:
            keep = np.arange(len(boxes))  # 调试用：完全关闭 NMS
        else:
            keep = nms(boxes, scores, self.params.nms_iou)
        if self.params.containment_prune and len(keep) > 1:
            keep = _prune_inflated_boxes(boxes[keep], scores[keep], keep)
        keep = keep[np.argsort(-scores[keep])][: self.params.max_detections]
        out = np.concatenate([clip_boxes(boxes[keep], w, h), scores[keep, None]], axis=1)
        return out.astype(np.float32)


def _prune_inflated_boxes(boxes: np.ndarray, scores: np.ndarray,
                          keep: np.ndarray) -> np.ndarray:
    """删除"把一个分数接近的可信小框整个包住"的膨胀大框。

    滑窗检测的一个典型失败模式：一只 40 px 的鸟，32/48/64 的窗口都能检测到，
    96/128 的大窗口也能检测到（因为鸟只是窗口里的一小部分），而且大窗口的分数
    有时更高。结果就是框得太大，IoU < 0.5，被算作 FP。
    这里用几何包含关系做后处理：如果大框 A 包含小框 B，且 B 的分数不低于 A 的
    一半，就认为 A 是"被背景撑大的框"，删掉 A。
    """
    if len(keep) <= 1:
        return keep
    b = boxes.astype(np.float32)
    area = np.maximum(b[:, 2] - b[:, 0], 1) * np.maximum(b[:, 3] - b[:, 1], 1)
    lt = np.maximum(b[:, None, :2], b[None, :, :2])
    rb = np.minimum(b[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    min_area = np.minimum(area[:, None], area[None, :])
    contain = inter / np.maximum(min_area, 1e-9)
    larger = area[:, None] > 1.8 * area[None, :]
    credible = scores[None, :] >= 0.5 * scores[:, None]
    # 对每一行 i（可能是大框），看是否存在 j 满足"被 i 包含 + j 更小 + j 分数够高"
    drop = np.any((contain > 0.8) & larger & credible, axis=1)
    return keep[~drop]


# ---------------------------------------------------------------------------
# 训练集构建
# ---------------------------------------------------------------------------
def build_training_data(records, params: HogParams, extractor: FeatureExtractor,
                        seed: int = 42, verbose: bool = True):
    """从训练集构造 (X, y)。

    正样本：每个 GT 框裁一个 patch（可加 jitter 增强），标签 1。
    负样本：每张图随机采 neg_per_image 个背景 patch，标签 0。
    还会把正样本水平翻转一份，提升对朝向的鲁棒性（只对训练集做，不泄漏）。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    n_pos = n_neg = 0

    for k, rec in enumerate(records):
        img = rec.load_image()
        # ---- 正样本 ----
        for box in rec.boxes:
            patch = crop_box(img, box, params.context, extractor.patch_size)
            X.append(extractor(patch)); y.append(1); n_pos += 1
            # jitter：随机缩放/平移，模拟滑窗没有正好套准的情况
            for _ in range(params.jitter_per_box):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                scale = rng.uniform(0.85, 1.15)
                cx = (x1 + x2) / 2 + rng.uniform(-0.06, 0.06) * bw
                cy = (y1 + y2) / 2 + rng.uniform(-0.06, 0.06) * bh
                side = max(bw, bh) * params.context * scale
                p = crop_square(img, cx, cy, side, extractor.patch_size)
                X.append(extractor(p)); y.append(1); n_pos += 1
            # 水平翻转（鸟没有固定左右朝向）
            X.append(extractor(cv2.flip(patch, 1))); y.append(1); n_pos += 1
        # ---- 负样本 ----
        for _ in range(params.neg_per_image):
            p = sample_negative_patch(img, rec.boxes, rng, extractor.patch_size)
            if p is not None:
                X.append(extractor(p)); y.append(0); n_neg += 1
                if rng.random() < 0.5:
                    X.append(extractor(cv2.flip(p, 1))); y.append(0); n_neg += 1
        if verbose and (k + 1) % 200 == 0:
            print(f"  [features] {k+1}/{len(records)} images, pos={n_pos}, neg={n_neg}", flush=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    if verbose:
        print(f"  [features] done: X={X.shape}, pos={int((y==1).sum())}, neg={int((y==0).sum())}")
    return X, y


def mine_hard_negatives(records, detector: WindowDetector, extractor: FeatureExtractor,
                        max_negatives: int = 5000, score_threshold: float = 0.2,
                        max_images: int = 200, seed: int = 42):
    """难负样本挖掘（hard negative mining）——本方法的关键技巧之一。

    流程：
      1. 用第一版分类器在训练图片上跑滑窗；
      2. 找出"分数很高、但其实不是鸟"的背景窗口（假正例）；
      3. 把这些窗口加入负样本重新训练。

    直观理解：普通随机负样本大多是"明显的背景"（纯沙滩/纯水面），
    分类器很容易学会；难负样本是"看起来像鸟的石头/浪花"，
    只有把它们喂给模型，误检才会真正下降。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(max_images, len(records)), replace=False)
    X_hard, n = [], 0
    for i, k in enumerate(idx):
        rec = records[int(k)]
        img = rec.load_image()
        det = detector.detect(img)
        if det.shape[0] == 0:
            continue
        # 按分数从高到低，把与 GT 重叠很小的检测当作难负样本
        for row in det[np.argsort(-det[:, 4])]:
            box, score = row[:4], float(row[4])
            if score < score_threshold:
                break
            if rec.boxes.shape[0]:
                if box_iou(box[None, :], rec.boxes)[0].max() > 0.25:
                    continue
            patch = crop_square(img, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                                max(box[2] - box[0], box[3] - box[1]),
                                extractor.patch_size)
            X_hard.append(extractor(patch))
            n += 1
            if n >= max_negatives:
                break
        if n >= max_negatives:
            break
        if (i + 1) % 50 == 0:
            print(f"  [hard-neg] scanned {i+1}/{len(idx)} images, collected {n}", flush=True)
    X_hard = np.asarray(X_hard, dtype=np.float32) if X_hard else np.zeros((0, extractor.dim), np.float32)
    print(f"  [hard-neg] collected {len(X_hard)} hard negatives")
    return X_hard


def make_classifier(kind: str = "linear", seed: int = 42):
    """构造分类器 Pipeline（标准化 + 分类器）。"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "linear": # construct liner classifier 线性svm分类器
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000, random_state=seed)

    elif kind == "rf": # random forest 随机森林
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed)
    elif kind == "rbf":
        from sklearn.svm import SVC
        clf = SVC(C=2.0, gamma="scale", class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"未知分类器: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
