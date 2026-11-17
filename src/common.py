"""Shared dataset, bounding-box, and evaluation utilities.

The loaders convert normalized YOLO labels to pixel-space xyxy boxes.
Detection methods share the IoU, NMS, clipping, and prediction JSON helpers.
The evaluator uses the same matching rules for AP, precision, recall, F1,
and the size and density breakdowns. AP is reported for a single class
(bird), so single-class mAP and AP are equal at the same IoU threshold.
"""

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
# Project paths.
# ---------------------------------------------------------------------------
# Resolve the project root from this module's src/ directory.
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
# COCO size groups use box area in square pixels.
#   small : area < 32²   = 1024 px²
#   medium: 32² ≤ area < 96² = 9216 px²
#   large : area ≥ 96²
SMALL_MAX_AREA = 32 * 32
MEDIUM_MAX_AREA = 96 * 96

# Group images by their ground-truth bird count.
DENSITY_BINS = [(0, 0, "0 (negative)"), (1, 5, "1-5 (sparse)"),
                (6, 20, "6-20 (moderate)"), (21, 50, "21-50 (dense)"),
                (51, 10**9, "51+ (very dense)")]


def ensure_dirs() -> None:
    """Create the output directories if needed."""
    for d in (RESULTS_DIR, PRED_DIR, METRICS_DIR, FIG_DIR, TABLE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Seed the random generators used by operations such as negative sampling."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Dataset loading.
# ---------------------------------------------------------------------------
@dataclass
class ImageRecord:
    """An image and its annotations.

    image_id is the filename stem. image_path and label_path point to the
    source files. width and height are measured in pixels. boxes is an
    (N, 4) array of xyxy coordinates, with shape (0, 4) for empty images.
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
        """Read the image as three-channel BGR."""
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {self.image_path}")
        return img


def _read_image_size(path: Path) -> Tuple[int, int]:
    """Read the image dimensions from its header without decoding the pixels."""
    from PIL import Image

    with Image.open(path) as im:
        return im.width, im.height


def yolo_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized YOLO coordinates to pixel-space xyxy boxes.

    Each label row gives class_id, cx, cy, w, and h. This dataset has one
    class, so the loader uses only the coordinates.
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
    """Read YOLO labels as an (N, 4) array of xyxy boxes."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # The only class ID is 0 (bird).
            rows.append([float(v) for v in parts[1:5]])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return yolo_to_xyxy(np.asarray(rows, dtype=np.float32), width, height)


def load_split(split: str, with_size: bool = True) -> List[ImageRecord]:
    """Load all image records from the supplied train, val, or test split."""
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
# Bounding-box operations.
# ---------------------------------------------------------------------------
def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip boxes to the image and keep their width and height positive."""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    b = boxes.copy().astype(np.float32)
    b[:, 0::2] = np.clip(b[:, 0::2], 0, width - 1)
    b[:, 1::2] = np.clip(b[:, 1::2], 0, height - 1)
    # Avoid zero-width and zero-height boxes after clipping.
    b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1)
    b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1)
    return b


def box_areas(boxes: np.ndarray) -> np.ndarray:
    """Return each box area in square pixels."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU and return an array of shape (len(a), len(b)).

    Calculate each intersection and divide by the corresponding union area.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    # Broadcast the corner coordinates for pairwise comparisons.
    lt = np.maximum(a[:, None, :2], b[None, :, :2])   # Top-left corners of the intersections.
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])   # Bottom-right corners of the intersections.
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = box_areas(a)[:, None]
    area_b = box_areas(b)[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Apply non-maximum suppression to scored xyxy boxes.

    Keep the highest-scored box at each step and suppress remaining boxes
    with IoU above the threshold. A higher threshold keeps more overlapping
    boxes. Return the retained indices in descending score order.
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
        # Keep only the candidates below the overlap threshold for the next step.
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   iou_threshold: float = 0.45) -> np.ndarray:
    """Apply NMS separately to each class."""
    keep_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        keep = idx[nms(boxes[idx], scores[idx], iou_threshold)]
        keep_all.append(keep)
    return np.concatenate(keep_all) if keep_all else np.zeros((0,), dtype=np.int64)


# ---------------------------------------------------------------------------
# Average precision and detection metrics.
# ---------------------------------------------------------------------------
def average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    """Compute AP from detections ordered by confidence.

    Accumulate precision and recall, make the precision envelope
    monotonically decreasing, and integrate over changes in recall.
    """
    if num_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp).astype(np.float64)
    fp_cum = np.cumsum(fp).astype(np.float64)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Add endpoints at recall 0 and 1.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Build the precision envelope.
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Integrate only where recall increases.
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def _match_image(gt: np.ndarray, det: np.ndarray, iou_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Match detections to ground truth greedily within one image.

    Detections have shape (M, 5) and must be sorted by descending score.
    Return Boolean arrays for matched detections and matched ground truth.
    """
    n_gt, n_det = gt.shape[0], det.shape[0]
    gt_matched = np.zeros(n_gt, dtype=bool)
    det_matched = np.zeros(n_det, dtype=bool)
    if n_gt == 0 or n_det == 0:
        return det_matched, gt_matched
    ious = box_iou(det[:, :4], gt)  # (M, N)
    for d in range(n_det):
        # Find the highest-IoU ground-truth box among the unmatched ones.
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
    """Collect detections across images and sort by descending score.

    Return (score, image_id, det_index) tuples, the per-image detections,
    and the total number of ground-truth boxes.
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
        # Keep each image's detections in descending score order.
        det = det[np.argsort(-det[:, 4])]
        dets_by_image[rec.image_id] = det
        for j in range(det.shape[0]):
            entries.append((float(det[j, 4]), rec.image_id, j))
    entries.sort(key=lambda x: -x[0])
    return entries, dets_by_image, num_gt


def evaluate_ap(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                iou_threshold: float = 0.5, gt_filter=None) -> float:
    """Compute single-class AP at a given IoU threshold."""
    entries, dets_by_image, num_gt = _collect_detections(preds, records, gt_filter)
    if num_gt == 0:
        return float("nan")

    # Match detections by descending score and track matches per image.
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
    """Return the precision-recall points using the same matching rules as evaluate_ap."""
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
    """Return AP at IoU 0.5 and 0.75 and the mean AP across the IoU thresholds."""
    thrs = list(iou_thresholds)
    aps = [evaluate_ap(preds, records, t, gt_filter) for t in thrs]
    out = {}
    for t, ap in zip(thrs, aps):
        out[f"AP@{t:.2f}"] = ap
    if thrs:
        out["AP@0.5"] = aps[0]
        # Compute AP at 0.75 separately when the grid omits it.
        out["AP@0.75"] = evaluate_ap(preds, records, 0.75, gt_filter)
        out["AP@0.5:0.95"] = float(np.nanmean(aps))
    return out


def prf_at_threshold(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                     score_threshold: float, iou_threshold: float = 0.5) -> Dict[str, float]:
    """Return precision, recall, F1, TP, FP, and FN at a fixed confidence threshold."""
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
    """Choose the confidence threshold that maximizes validation F1 for later test evaluation."""
    best_t, best = 0.5, None
    for t in np.linspace(0.01, 0.99, 99):
        m = prf_at_threshold(preds, records, float(t), iou_threshold)
        if best is None or m["f1"] > best["f1"]:
            best_t, best = float(t), m
    return best_t, best


def count_metrics(preds: Dict[str, np.ndarray], records: Sequence[ImageRecord],
                  score_threshold: float) -> Dict[str, float]:
    """Measure the error in the predicted bird count per image.

    MAE is the mean absolute count error. Bias is the mean signed error
    (predicted minus actual). Relative MAE divides by the mean true count.
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
    """Measure false positives on images that contain no birds."""
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
    """Group images by ground-truth count and compute detection and counting metrics."""
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
    """Compute AP and recall by COCO ground-truth box size."""
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
        # Compute fixed-threshold recall for the ground-truth boxes in this size group.
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
                # Match detections greedily in descending score order.
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
# Prediction JSON files.
# ---------------------------------------------------------------------------
def save_predictions(path: Path, preds: Dict[str, np.ndarray], meta: Optional[Dict] = None) -> None:
    """Write predictions and metadata to JSON for use by other scripts.

    The top-level meta field holds run metadata. predictions maps each image
    ID to rows of [x1, y1, x2, y2, score].
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
    """Read the JSON format written by save_predictions."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    preds = {k: np.asarray(v, dtype=np.float32).reshape(-1, 5)
             for k, v in payload.get("predictions", {}).items()}
    return preds, payload.get("meta", {})


def load_image_bgr(rec: ImageRecord) -> np.ndarray:
    """Load the BGR image for a dataset record."""
    return rec.load_image()


def draw_boxes(image: np.ndarray, boxes: np.ndarray, color=(0, 255, 0),
               thickness: int = 2, scores: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw the boxes on a copy of the image."""
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
    """Return summary count statistics for logging."""
    counts = [len(v) for v in preds.values()]
    return {"num_images": len(counts), "total_boxes": int(sum(counts)),
            "mean_boxes": float(np.mean(counts)) if counts else 0.0}
