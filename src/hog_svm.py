"""Bird detection using HOG or HOG+LBP features and an SVM or random forest.

A multiscale sliding window supplies candidate locations. Resize each
crop, extract its features, and score it with the trained classifier.
NMS removes overlapping detections. Window size and stride control how
much of the image is covered and how many patches need to be scored.

HOG uses local gradient histograms with block normalization. LBP adds
texture histograms. The linear SVM and random forest provide two
classifiers for the same patch features.
"""

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
# Detector parameters.
# ---------------------------------------------------------------------------
@dataclass
class HogParams:
    """Settings for feature extraction, training, and sliding-window inference.

    window_sizes are side lengths in pixels. A smaller stride samples more
    positions and takes longer. A larger nms_iou preserves more overlapping
    detections. score_threshold is the minimum classifier score.
    """
    window_sizes: Tuple[int, ...] = (32, 48, 64, 96, 128)
    stride: int = 16
    patch_size: int = 64          # Resize every patch to this size before extracting features.
    context: float = 1.3          # Expand positive training crops by this factor to include context.
    nms_iou: float = 0.40
    score_threshold: float = -0.6  # Minimum LinearSVC decision score. Negative scores are allowed.
    max_detections: int = 300
    max_candidates: int = 1200     # Maximum number of candidates passed to NMS.
    # Larger windows often receive higher scores because they include more context.
    # Subtract penalty * log2(size / 64) to compensate.
    size_score_penalty: float = 0.0
    containment_prune: bool = True   # Suppress large boxes that enclose confident smaller detections.
    # Training crops include context around the ground-truth box.
    # Shrink detections toward the center to compensate for that expansion.
    box_shrink: float = 0.77
    # Feature settings.
    use_hog: bool = True
    use_lbp: bool = False
    # Training settings.
    neg_per_image: int = 16
    jitter_per_box: int = 1
    hard_negatives: int = 0       # Set by the training script.


# ---------------------------------------------------------------------------
# Feature extraction.
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Convert a BGR patch to a feature vector using a reusable HOG descriptor."""

    def __init__(self, patch_size: int = 64, use_hog: bool = True, use_lbp: bool = False):
        self.patch_size = patch_size
        self.use_hog = use_hog
        self.use_lbp = use_lbp
        if use_hog:
            # winSize=64×64, blockSize=16×16, blockStride=8×8, cellSize=8×8, nbins=9
            # HOG gives 7 x 7 blocks, 2 x 2 cells per block, and 9 bins (1,764 values).
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
            d += 4 * 4 * 10  # Uniform LBP uses 10 bins for each cell in a 4 x 4 grid.
        return d

    def __call__(self, patch_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY) if patch_bgr.ndim == 3 else patch_bgr
        gray = cv2.resize(gray, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        feats = []
        if self.use_hog:
            feats.append(self.hog.compute(gray).ravel().astype(np.float32))
        if self.use_lbp:
            lbp = self._lbp(gray, P=8, R=1, method="uniform")
            # Compute a histogram per cell to preserve spatial information.
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
# Patch extraction.
# ---------------------------------------------------------------------------
def crop_square(img: np.ndarray, cx: float, cy: float, side: float,
                out_size: int = 64) -> np.ndarray:
    """Crop a square of side length side around (cx, cy) and resize to out_size.

    Replicate border pixels where the crop extends past the image edge.
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
    """Center the positive crop on the ground-truth box and expand its largest side by context."""
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * context
    return crop_square(img, cx, cy, side, out_size)


def sample_negative_patch(img: np.ndarray, gt_boxes: np.ndarray, rng: np.random.Generator,
                          patch_size: int = 64) -> Optional[np.ndarray]:
    """Sample a background patch with low IoU against all ground-truth boxes.

    Return None if none of the 30 attempts is suitable.
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
# Sliding windows.
# ---------------------------------------------------------------------------
def sliding_windows(width: int, height: int, sizes: Sequence[int], stride: int):
    """Yield the top-left corner and side length of each window.

    Include positions against the right and bottom edges when the stride
    does not land on those edges.
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
    """Run sliding-window feature extraction, classification, and NMS."""

    model: object                  # sklearn Pipeline
    extractor: FeatureExtractor
    params: HogParams = field(default_factory=HogParams)

    def detect(self, img: np.ndarray) -> np.ndarray:
        """Return [x1, y1, x2, y2, score] rows for one image."""
        h, w = img.shape[:2]
        boxes, scores = [], []
        patches = []          # Patches for the current batch.
        meta = []             # Boxes for the current batch.

        def flush():
            """Extract and score features for the current batch and collect candidates."""
            if not patches:
                return
            X = self.extractor.batch(patches)
            if hasattr(self.model, "decision_function"):
                s = self.model.decision_function(X)
            else:
                s = self.model.predict_proba(X)[:, 1]
            for (x, y, size), sc in zip(meta, s):
                # Penalize larger windows to account for their higher scores.
                sc = float(sc) - self.params.size_score_penalty * np.log2(size / 64.0)
                # Discard low scores before NMS.
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

        # Shrink boxes toward their centers to offset the training crop context.
        if self.params.box_shrink != 1.0:
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            half_w = (boxes[:, 2] - boxes[:, 0]) * self.params.box_shrink / 2
            half_h = (boxes[:, 3] - boxes[:, 1]) * self.params.box_shrink / 2
            boxes = np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)

        # Keep only the top candidates to bound the cost of NMS.
        if len(scores) > self.params.max_candidates:
            top = np.argsort(-scores)[: self.params.max_candidates]
            boxes, scores = boxes[top], scores[top]

        if self.params.nms_iou >= 1.0:
            keep = np.arange(len(boxes))  # Allow NMS to be disabled for debugging.
        else:
            keep = nms(boxes, scores, self.params.nms_iou)
        if self.params.containment_prune and len(keep) > 1:
            keep = _prune_inflated_boxes(boxes[keep], scores[keep], keep)
        keep = keep[np.argsort(-scores[keep])][: self.params.max_detections]
        out = np.concatenate([clip_boxes(boxes[keep], w, h), scores[keep, None]], axis=1)
        return out.astype(np.float32)


def _prune_inflated_boxes(boxes: np.ndarray, scores: np.ndarray,
                          keep: np.ndarray) -> np.ndarray:
    """Suppress oversized detections that enclose smaller high-scoring boxes.

    Sliding windows at several sizes can score well on the same bird.
    If a large box contains a smaller box with a score at least half as
    high, prefer the tighter box.
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
    # Look for sufficiently high-scoring smaller boxes contained in each candidate.
    drop = np.any((contain > 0.8) & larger & credible, axis=1)
    return keep[~drop]


# ---------------------------------------------------------------------------
# Training data.
# ---------------------------------------------------------------------------
def build_training_data(records, params: HogParams, extractor: FeatureExtractor,
                        seed: int = 42, verbose: bool = True):
    """Build the feature matrix and labels from the training images.

    Use crops around ground-truth birds as positive samples. Jitter and
    horizontal flips add variation. Sample background patches as negatives
    according to neg_per_image.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    n_pos = n_neg = 0

    for k, rec in enumerate(records):
        img = rec.load_image()
        # Positive patches.
        for box in rec.boxes:
            patch = crop_box(img, box, params.context, extractor.patch_size)
            X.append(extractor(patch)); y.append(1); n_pos += 1
            # Jitter scale and position to mimic imperfectly aligned windows.
            for _ in range(params.jitter_per_box):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                scale = rng.uniform(0.85, 1.15)
                cx = (x1 + x2) / 2 + rng.uniform(-0.06, 0.06) * bw
                cy = (y1 + y2) / 2 + rng.uniform(-0.06, 0.06) * bh
                side = max(bw, bh) * params.context * scale
                p = crop_square(img, cx, cy, side, extractor.patch_size)
                X.append(extractor(p)); y.append(1); n_pos += 1
            # Flip bird crops horizontally.
            X.append(extractor(cv2.flip(patch, 1))); y.append(1); n_pos += 1
        # Negative patches.
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
    """Mine hard negatives from false positives on the training set.

    Run sliding-window detection with the initial classifier and collect
    high-scoring patches with low overlap with any bird. Add those patches
    as negative examples when refitting the classifier.
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
        # Select high-scoring detections with low overlap with ground truth.
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
    """Build the feature-scaling and classifier pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "linear": # Linear SVM.
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000, random_state=seed)

    elif kind == "rf": # Random forest.
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
