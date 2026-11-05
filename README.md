# Beach Bird Detection — Classical, HOG+SVM and YOLO

A reproducible benchmark of **three families of computer-vision methods** for detecting birds in
cluttered beach / wetland scenes, built for the COMP9517 (Computer Vision) group project.

The project uses the [Beach Bird Detection Dataset](https://www.kaggle.com/datasets/liugalhigh/beach-bird)
(1,900 images, 12,024 bird bounding boxes, 512×512, single class `bird`).

---

## What is compared

| Family | Method | Key idea |
|---|---|---|
| Non-learning | Global Otsu | Global intensity threshold + morphology + contours |
| Non-learning | Multi-scale black-hat / top-hat | Local contrast at 6 structuring-element scales |
| Non-learning | Black-hat + watershed | Splits merged blobs in dense flocks |
| Machine learning | HOG + Linear SVM (sliding) | Multi-scale sliding window, hard-negative mining |
| Machine learning | HOG + SVM (proposals) | Black-hat proposals + HOG/SVM scoring (cascade) |
| Machine learning | HOG+LBP + SVM (proposals) | Adds texture features |
| Machine learning | HOG + Random Forest (proposals) | Non-linear classifier |
| Deep learning | YOLOv8n, YOLOv8s, YOLO11n | One-stage detectors, controlled ablations |
| Deep learning | YOLOv8n @960 / TTA / 2×2 tiling | Resolution and inference-time improvements |

All methods are evaluated with the **same from-scratch COCO-style AP@0.5 implementation** in
`src/common.py`, using thresholds selected on the validation split and applied unchanged to the
test split.

---

## Results at a glance

<!-- RESULTS_TABLE_START -->
Test-set results, thresholds selected on validation data (see `results/tables/main_results.csv`):

| Family | Method | AP@0.5 | AP@0.75 | F1 | ms/img |
|---|---|---:|---:|---:|---:|
| Non-learning | Global Otsu | 0.016 | 0.004 | 0.071 | 9 |
| Non-learning | Multi-scale black-hat | 0.021 | 0.003 | 0.096 | 75 |
| Non-learning | Black-hat + watershed | 0.017 | 0.002 | 0.091 | 88 |
| ML | HOG + Linear SVM (sliding) | 0.011 | 0.000 | 0.058 | 458 |
| ML | HOG + SVM (black-hat proposals) | 0.015 | 0.001 | 0.074 | 72 |
| ML | HOG+LBP + SVM (proposals) | 0.021 | 0.001 | 0.079 | 86 |
| ML | HOG + Random Forest (proposals) | **0.022** | 0.001 | 0.076 | 75 |
| DL | YOLOv8n (no augmentation) | 0.750 | 0.323 | 0.725 | 27 |
| DL | YOLOv8n + augmentation | 0.871 | 0.458 | 0.835 | 23 |
| DL | YOLO11n + augmentation | **0.876** | 0.459 | 0.836 | 25 |
| DL | YOLOv8s + augmentation | 0.872 | 0.458 | 0.841 | 25 |
| DL | YOLOv8n @960 | 0.806 | 0.418 | 0.773 | 18 |
| DL (post-hoc) | YOLOv8n + TTA | 0.860 | **0.470** | 0.822 | 32 |
| DL (post-hoc) | YOLOv8n + 2×2 tiles | 0.779 | 0.382 | 0.771 | 49 |

**Key findings**

- YOLO is far ahead: **0.876 AP@0.5** vs. **0.022** for the best handcrafted method.
- Data augmentation is the single biggest YOLO gain: 0.750 → 0.871.
- The proposal-based cascade makes HOG 8× faster and 4× more accurate than dense sliding windows,
  but handcrafted features still cannot separate birds from foam/rocks.
- TTA improves localisation (AP@0.75 0.458 → 0.470) but not AP@0.5.
- Upsampling to 960 or 2×2 tiling does **not** help on this natively 512-px dataset; tiling badly
  hurts large birds (AP 0.680 → 0.493).
<!-- RESULTS_TABLE_END -->

Full tables, PR curves and qualitative examples are in `results/` and `report/`.
The compiled report (`report/COMP9517_Group_Project_Report.pdf`) is self-contained; if you open
`report/report.html` directly, first run `python scripts/07_make_figures.py` so that the figure
paths resolve.

---

## Repository layout

```
beach_bird_project/
├── README.md
├── requirements.txt
├── docs/
│   └── 使用文档_中文.md          # detailed beginner-friendly manual (Chinese)
├── src/
│   ├── common.py                # dataset, IoU, NMS, AP, metrics
│   ├── classical.py             # non-learning methods
│   └── hog_svm.py               # HOG/LBP + sliding window + SVM/RF
├── scripts/
│   ├── 00_explore_dataset.py
│   ├── 01_tune_classical.py
│   ├── 02_run_classical.py
│   ├── 03_train_hog_svm.py
│   ├── 03_run_hog_svm.py
│   ├── 04_train_yolo.py
│   ├── 05_predict_yolo.py
│   ├── 06_evaluate_all.py
│   ├── 06_model_info.py
│   └── 07_make_figures.py
├── report/
│   ├── report.html              # two-column IEEE-style report source
│   └── COMP9517_Group_Project_Report.pdf
└── data/                        # not committed — download separately
```

---

## Installation

Tested with Python 3.11, PyTorch 2.14 + CUDA 12.6, Ultralytics 8.4, OpenCV 4.12, on an
RTX 3060 Laptop (6 GB).

```bash
git clone <your-repo-url>
cd beach_bird_project
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# PyTorch: choose the build for your CUDA from https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> **Important:** OpenCV 5.x removed `cv2.HOGDescriptor`, which the HOG pipeline relies on.
> `requirements.txt` pins `opencv-python<5`; do not upgrade it.

CPU-only works too, but YOLO training is 10–30× slower. Reduce `--epochs` or use `yolov8n`.

---

## Dataset

Download from Kaggle and arrange the files exactly like this:

```
data/Beach Bird Detection Dataset v1.0/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
└── dataset.yaml
```

or, with `kagglehub`:

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('liugalhigh/beach-bird'))"
```

The dataset is already split into 1,369 / 291 / 240 images. **Do not re-split it** — the official
split is source-group-aware and prevents leakage.

---

## Quick start

```bash
# 0) dataset statistics + example figures
python scripts/00_explore_dataset.py

# 1) non-learning methods (tune on val, then run test)
python scripts/01_tune_classical.py --variant multiscale_blackhat --subset 120
python scripts/02_run_classical.py --variant all --split val  --use-best
python scripts/02_run_classical.py --variant all --split test --use-best

# 2) HOG + SVM (train the classifier)
python scripts/03_train_hog_svm.py --name hog_linear --hard-neg 6000

# 2a) dense sliding window (slow, weak localisation — kept for comparison)
python scripts/03_run_hog_svm.py --name hog_linear --split val  --stride 16 --score-threshold 0 --box-shrink 1.0 --tag hog_linear
python scripts/03_run_hog_svm.py --name hog_linear --split test --stride 16 --score-threshold 0 --box-shrink 1.0 --tag hog_linear

# 2b) recommended: black-hat proposals + HOG/SVM scoring (8x faster, 4x better AP)
python scripts/03_run_hog_proposals.py --model hog_linear --split val  --tag hog_prop
python scripts/03_run_hog_proposals.py --model hog_linear --split test --tag hog_prop

# 3) YOLO (example: one model)
python scripts/04_train_yolo.py --model yolov8n.pt --name yolov8n_aug --epochs 60 --batch 32
python scripts/05_predict_yolo.py --weights models/runs/yolov8n_aug/weights/best.pt --name yolov8n_aug --split val
python scripts/05_predict_yolo.py --weights models/runs/yolov8n_aug/weights/best.pt --name yolov8n_aug --split test

# 4) unified evaluation, tables and PR curves
python scripts/06_evaluate_all.py
python scripts/06_model_info.py
python scripts/07_make_figures.py
```

A single YOLOv8n run (60 epochs, 512 px, batch 32) takes roughly 25–40 minutes on the RTX 3060
used for the experiments. HOG training takes a few minutes on CPU.

---

## Method notes

### Non-learning baseline

For every scale `k`, the detector computes

```
black-hat = closing(I, SE_k) - I      # dark blobs (birds on bright sand/water)
top-hat   = I - opening(I, SE_k)      # bright blobs (gulls on dark water/rock)
response  = max(black-hat, top-hat)
```

Each response map is Otsu-thresholded, cleaned with morphology, and converted to boxes.
Candidates are filtered by area, aspect ratio, fill ratio, and **scale consistency** (a blob's
size must match the structuring element that produced it). A final NMS merges scales. The
watershed variant splits large connected components using distance-transform peaks.

### HOG + SVM

- Positives: GT boxes expanded by 1.3×, resized to 64×64, with jitter and horizontal flips.
- Negatives: random background windows (IoU < 0.05 with any GT) plus 6,000 hard negatives mined
  from the first-stage detector.
- Features: OpenCV HOG (8×8 cells, 2×2 blocks, 9 bins → 1,764-D); optional 4×4-grid uniform LBP
  (+160-D).
- Classifier: `LinearSVC(C=0.01, class_weight='balanced')`; optionally Random Forest.
- Inference: windows of 32/48/64/96/128 px, stride 16, score calibration
  `s - λ·log2(size/64)`, NMS, and containment pruning of inflated boxes.

### YOLO

Five controlled runs:

| Run | Model | Input | Augmentation |
|---|---|---|---|
| `yolov8n_plain` | YOLOv8n | 512 | none |
| `yolov8n_aug` | YOLOv8n | 512 | default |
| `yolo11n_aug` | YOLO11n | 512 | default |
| `yolov8s_aug` | YOLOv8s | 512 | default |
| `yolov8n_aug_960` | YOLOv8n | 960 | default |

Post-hoc inference experiments on the best YOLOv8n weights: TTA and 2×2 tiled inference.

---

## Evaluation protocol

- Metrics are computed by `src/common.py` from raw predictions, not by the training library.
- AP uses all-point interpolation (COCO-style) at IoU 0.5, 0.75, and 0.5:0.95.
- The operating confidence threshold is selected on **val** (max F1) and applied unchanged to
  **test**.
- Additional breakdowns: small/medium/large objects (COCO area thresholds), density bins
  (0, 1–5, 6–20, 21–50, 51+), negative-scene false positives, and counting MAE.
- Inference time is measured per image on the same machine, excluding model loading.

---

## Reproducing all experiments

See section 14 of [`docs/使用文档_中文.md`](docs/使用文档_中文.md) for the full command list.
In short:

```bash
python scripts/00_explore_dataset.py
python scripts/01_tune_classical.py --variant multiscale_blackhat --subset 120
python scripts/02_run_classical.py --variant all --split val  --use-best
python scripts/02_run_classical.py --variant all --split test --use-best

python scripts/03_train_hog_svm.py --name hog_linear     --hard-neg 6000
python scripts/03_train_hog_svm.py --name hog_lbp_linear --feature hog_lbp --hard-neg 6000
python scripts/03_train_hog_svm.py --name hog_rf         --classifier rf    --hard-neg 6000
# ... run 03_run_hog_svm.py for val/test ...

python scripts/04_train_yolo.py --model yolov8n.pt --name yolov8n_plain --epochs 60 --batch 32 --aug plain
python scripts/04_train_yolo.py --model yolov8n.pt --name yolov8n_aug   --epochs 60 --batch 32 --aug default
python scripts/04_train_yolo.py --model yolo11n.pt --name yolo11n_aug  --epochs 60 --batch 32 --aug default
python scripts/04_train_yolo.py --model yolov8s.pt --name yolov8s_aug  --epochs 60 --batch 16 --aug default
python scripts/04_train_yolo.py --model yolov8n.pt --name yolov8n_aug_960 --epochs 45 --batch 6 --imgsz 960 --aug default
# ... run 05_predict_yolo.py for val/test ...

python scripts/06_evaluate_all.py
python scripts/06_model_info.py
python scripts/07_make_figures.py
```

---

## Pretrained weights

Weights are **not** included in the repository (the assignment caps the code archive at 25 MB).
Train them with the commands above; YOLO weights are written to
`models/runs/<name>/weights/best.pt`, HOG models to `models/hog_svm/<name>.joblib`.

---

## Hardware

All experiments were run on a single NVIDIA RTX 3060 Laptop GPU (6 GB) and an Intel/AMD CPU with
16 GB RAM, Windows 11. YOLO training used automatic mixed precision (AMP).

---

## Acknowledgements

- Dataset: Kai Liu, *Beach Bird Detection Dataset v1.0*, Kaggle, 2026.
- Detectors: [Ultralytics YOLOv8/YOLO11](https://github.com/ultralytics/ultralytics).
- Classical features: OpenCV `HOGDescriptor`, scikit-image LBP, scikit-learn SVM/RF.
- Course: COMP9517 Computer Vision, UNSW, 2026 Term 3.

## License

Code released for coursework and research use. The dataset remains subject to its own Kaggle
license; see `data/Beach Bird Detection Dataset v1.0/LICENSE.md`.
