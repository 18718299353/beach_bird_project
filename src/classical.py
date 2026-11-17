"""Classical bird detectors based on thresholding and morphology.

global_otsu thresholds the grayscale image for dark objects, cleans the
mask, and turns connected components into boxes. Dark rocks and shadows
can produce false positives, and light birds can be missed.

multiscale_blackhat computes local contrast at several kernel sizes.
Black-hat (closing minus image) highlights dark objects. Top-hat (image
minus opening) highlights bright objects. Otsu thresholding is applied
to the stronger response at each scale.

multiscale_blackhat_ws also uses watershed to split large connected
components that may contain several birds. These methods provide a
baseline for the learned detectors.
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
    """Parameters for the classical detectors, tuned on the validation split."""
    # Candidate filters.
    min_area: float = 80.0        # Minimum component area in pixels squared.
    max_area: float = 25000.0     # Maximum component area in pixels squared.
    min_fill: float = 0.30        # Minimum component area divided by bounding-box area.
    min_aspect: float = 0.22      # Minimum width-to-height ratio.
    max_aspect: float = 3.5       # Maximum width-to-height ratio.
    # Multiscale morphology.
    se_sizes: Tuple[int, ...] = (7, 11, 17, 25, 35, 51)  # Odd kernel sizes for the structuring elements.
    resp_floor: float = 8.0        # Minimum response threshold for flat regions.
    otsu_scale: float = 1.0        # Multiplier on the Otsu threshold. Values above 1 are stricter.
    open_ksize: int = 3            # Opening kernel size for removing isolated noise.
    close_ksize: int = 3           # Closing kernel size for filling gaps. Large kernels can join nearby birds.
    # Scale consistency.
    # Accept components from scales close to their size.
    # This avoids spurious small-scale responses on large rocks.
    se_size_min_ratio: float = 0.12  # Minimum component size relative to the morphology kernel.
    se_size_max_ratio: float = 1.15  # Maximum component size relative to the morphology kernel.
    # Watershed settings.
    ws_min_distance: int = 4       # Minimum distance between peaks. Smaller values can over-segment objects.
    ws_threshold_abs: float = 1.5  # Minimum height of a distance-transform peak.
    split_area: float = 2500.0     # Minimum component area for watershed splitting.
    max_split_area: float = 12000.0  # Maximum area for splitting. Larger regions are often rocks or shoreline.
    max_split_peaks: int = 24      # Maximum number of pieces from one component.
    # Output filters.
    min_score: float = 0.02        # Drop scores below this value before writing JSON.
    nms_iou: float = 0.35          # IoU threshold for NMS across scales.


def _ellipse(k: int) -> np.ndarray:
    """Create a k x k elliptical structuring element for positive odd k."""
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _response_maps(gray: np.ndarray, se_sizes: Sequence[int]) -> List[Tuple[int, np.ndarray]]:
    """Compute the stronger of the black-hat and top-hat responses at each scale.

    Black-hat responds to dark birds on light backgrounds. Top-hat responds
    to light birds against darker water or rocks.
    """
    maps = []
    for k in se_sizes:
        se = _ellipse(k)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
        resp = cv2.max(blackhat, tophat)
        # Apply mild Gaussian smoothing to suppress isolated bright pixels.
        resp = cv2.GaussianBlur(resp, (3, 3), 0)
        maps.append((k, resp))
    return maps


def _bird_score(peak: float, mean: float, area: float, fill: float,
                p: ClassicalParams) -> float:
    """Score a candidate box for AP ranking and thresholding.

    Multiply the local contrast, a size prior, and a shape factor based on
    component fill. The size prior peaks near the training-set median size
    of 42 pixels and downweights much smaller or larger components.
    """
    side = float(np.sqrt(max(area, 1.0)))
    # The log-normal size prior peaks at a side length of 42 pixels.
    size_prior = float(np.exp(-0.5 * (np.log(max(side, 1.0) / 42.0) / 0.85) ** 2))
    shape_prior = 0.5 + 0.5 * min(fill / 0.7, 1.0)
    contrast = 0.75 * peak + 0.25 * mean
    return float(contrast * (0.35 + 0.65 * size_prior) * shape_prior)


def _boxes_from_binary(binary: np.ndarray, resp: np.ndarray, se_size: Optional[int],
                       p: ClassicalParams) -> List[Tuple[np.ndarray, float]]:
    """Convert connected components in a binary mask to scored boxes.

    Filter by area, bounding-box fill, aspect ratio, and the ratio of
    component size to morphology kernel size.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: List[Tuple[np.ndarray, float]] = []
    for i in range(1, num):  # Label 0 is the background.
        x, y, w, h, area = stats[i]
        if area < p.min_area or area > p.max_area:
            continue
        if w <= 1 or h <= 1:
            continue
        fill = area / float(w * h)
        aspect = w / float(h)
        if fill < p.min_fill or not (p.min_aspect <= aspect <= p.max_aspect):
            continue
        # Compare the square root of component area with the kernel size.
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
    """Use distance-transform peaks to split a large component with watershed.

    Compute the transform inside the component, find peaks as seeds, and
    return a box for each region. With fewer than two peaks, keep the
    original component.
    """
    ys, xs = np.where(label_mask)
    if len(xs) == 0:
        return []
    area = len(xs)
    # Skip very large regions, which are often rocks or shoreline.
    if area > p.max_split_area:
        return []
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    roi_mask = label_mask[y0:y1, x0:x1].astype(np.uint8)
    roi_resp = resp[y0:y1, x0:x1]
    fill = area / float(max((x1 - x0) * (y1 - y0), 1))
    if fill < 0.4:  # Skip sparse or elongated components.
        return []

    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    # Find local maxima of the distance transform.
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
    # OpenCV watershed expects a three-channel input image.
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
    """Detect birds in one BGR image and return [x1, y1, x2, y2, score] rows."""
    p = params or ClassicalParams()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Compute a shared local-contrast map for scoring.
    global_resp = cv2.absdiff(gray, cv2.medianBlur(gray, 31))

    candidates: List[Tuple[np.ndarray, float]] = []

    if variant == "global_otsu":
        # Invert the global Otsu mask to keep pixels darker than the background.
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
            # Apply Otsu at each scale with a floor for flat response maps.
            otsu_thr, _ = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = max(p.resp_floor, otsu_thr * p.otsu_scale)
            binary = (resp >= thr).astype(np.uint8) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _ellipse(p.open_ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ellipse(p.close_ksize))

            if variant == "multiscale_blackhat_ws":
                # Extract components and split large ones where enabled.
                num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                for i in range(1, num):
                    x, y, w, h, area = stats[i]
                    if area < p.min_area or area > p.max_area:
                        continue
                    if area >= p.split_area:
                        comp_mask = labels == i
                        parts = _split_component_watershed(gray, resp, comp_mask, p)
                        # Keep the entire component if watershed cannot split it.
                        if len(parts) >= 2:
                            all_parts.extend(parts)
                            continue
                    # Apply the usual shape and scale filters to smaller components.
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

        # Use NMS to merge overlapping detections from different scales.
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
    """Run the detector on one split and save the predictions as JSON."""
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
