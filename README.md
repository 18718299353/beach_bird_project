# Beach Bird Detection

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%2011-00bfff)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5c3ee8)
![License](https://img.shields.io/badge/License-Coursework-lightgrey)

> **Just want to run it?** Jump to [Quick start](#quick-start). You need Python 3.10+, the
> Kaggle dataset, and about 30 minutes for one YOLO model.

A hands-on benchmark of **three different ways to detect birds in beach photos**:
classical image processing, HOG + SVM machine learning, and modern YOLO deep learning.

If you have ever tried to count birds on a beach, you know the problem. A single photo can
contain dozens of small, partially hidden birds that look almost identical to rocks, foam and
wave crests. Doing this by hand for thousands of images is slow and inconsistent. This project
asks a simple question:

> **How far can we get with handcrafted image processing, how far with a classical
> sliding-window classifier, and how much better is a modern detector?**

The repository contains the full pipeline: dataset loading, all detectors, training scripts,
prediction scripts, a shared evaluation implementation, result tables, and a written report.

---

## Table of contents

- [What is in this project](#what-is-in-this-project)
- [Results at a glance](#results-at-a-glance)
- [Quick start](#quick-start)
- [Dataset](#dataset)
- [Repository layout](#repository-layout)
- [Where each method lives](#where-each-method-lives)
- [Run one method end to end](#run-one-method-end-to-end)
- [Reproduce all experiments](#reproduce-all-experiments)
- [Outputs explained](#outputs-explained)
- [Troubleshooting](#troubleshooting)
- [Hardware](#hardware)
- [Acknowledgements](#acknowledgements)

---

## What is in this project

Three method families, **14 comparison settings**, and **10 trained models**: 5 HOG/SVM models
(3 in the main table, 2 used for ablations) and 5 YOLO models.

| Family | Method | Core idea |
|---|---|---|
| Non-learning | Global Otsu | Threshold the whole image, clean with morphology, take contours |
| Non-learning | Multi-scale black-hat / top-hat | Find small dark/bright blobs at several scales |
| Non-learning | Black-hat + watershed | Split touching birds in dense flocks |
| Machine learning | HOG + Linear SVM (sliding window) | Scan the image with fixed-size windows and classify each window |
| Machine learning | HOG + SVM (proposal cascade) | Use classical blobs as proposals, then let HOG/SVM score them |
| Machine learning | HOG + LBP + SVM | Add local texture features |
| Machine learning | HOG + Random Forest | Replace the linear SVM with a non-linear forest |
| Deep learning | YOLOv8n / YOLOv8s / YOLO11n | One-stage detectors with controlled ablations |
| Deep learning | YOLOv8n @960 / TTA / 2x2 tiles | Resolution and inference-time experiments |

All methods are evaluated by the **same code** in `src/common.py`, so the comparison is fair.

---

## Results at a glance

Test-set results. The confidence threshold for P/R/F1 is selected on the validation split and
applied unchanged to the test split. Higher AP/F1 is better; lower ms/img is better.

| Family | Method | AP@0.5 | AP@0.75 | F1 | ms/img |
|---|---|---:|---:|---:|---:|
| Non-learning | Global Otsu | 0.016 | 0.004 | 0.071 | 9 |
| Non-learning | Multi-scale black-hat | 0.021 | 0.003 | 0.096 | 75 |
| Non-learning | Black-hat + watershed | 0.017 | 0.002 | 0.091 | 88 |
| ML | HOG + Linear SVM (sliding) | 0.011 | 0.000 | 0.058 | 458 |
| ML | HOG + SVM (proposals) | 0.015 | 0.001 | 0.074 | 72 |
| ML | HOG+LBP + SVM (proposals) | 0.021 | 0.001 | 0.079 | 86 |
| ML | HOG + Random Forest (proposals) | **0.022** | 0.001 | 0.076 | 75 |
| DL | YOLOv8n (no augmentation) | 0.750 | 0.323 | 0.725 | 27 |
| DL | YOLOv8n + augmentation | 0.871 | 0.458 | 0.835 | 23 |
| DL | YOLO11n + augmentation | **0.876** | 0.459 | 0.836 | 25 |
| DL | YOLOv8s + augmentation | 0.872 | 0.458 | 0.841 | 25 |
| DL | YOLOv8n @960 | 0.806 | 0.418 | 0.773 | 18 |
| DL (post-hoc) | YOLOv8n + TTA | 0.860 | **0.470** | 0.822 | 32 |
| DL (post-hoc) | YOLOv8n + 2x2 tiles | 0.779 | 0.382 | 0.771 | 49 |

**What the numbers say**

- YOLO is far ahead: **0.876 AP@0.5** versus **0.022** for the best handcrafted method.
- Data augmentation is the biggest single win for YOLO: 0.750 to 0.871.
- For HOG, **where proposals come from** matters more than the classifier: replacing dense
  sliding windows with black-hat proposals is about 8x faster and raises validation AP from
  0.012 to 0.046. The test-set gain is smaller (0.011 to 0.015), but the speed advantage remains.
- TTA improves localisation (AP@0.75 0.458 to 0.470) but not AP@0.5.
- Upsampling to 960 or 2x2 tiling does **not** help on this natively 512-px dataset. Tiling
  badly hurts large birds (AP 0.680 to 0.493), which is itself a useful negative result.

The full table is written to `results/tables/main_results.csv` by
`scripts/06_evaluate_all.py`. The written report is in `report/`
(`COMP9517_Group_Project_Report.docx`, or the exported PDF).

---

## Quick start

### What you need

- Windows 10/11, Linux, or macOS
- Python **3.10 - 3.12**
- Git
- About 5 GB of free disk space, plus about 400 MB for the dataset
- Optional: an NVIDIA GPU. CPU works, but YOLO training is much slower.

> **Windows note:** the examples below use PowerShell paths
> (`.\.venv\Scripts\python.exe`). In CMD, use `.venv\Scripts\python.exe`
> (drop the leading `.\`).
>
> If typing `python` opens the Microsoft Store or prints nothing, that is the Windows Store
> alias, not a real Python. Install Python from <https://www.python.org/downloads/> and tick
> **Add python.exe to PATH**, then reopen the terminal.

### 1. Clone the repository

On the GitHub page, click the green **Code** button, copy the HTTPS URL, and run:

```bash
git clone <paste the copied URL here>
cd beach_bird_project
```

### 2. Create a virtual environment and install dependencies

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install PyTorch first. Choose the CUDA build that matches your machine.
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# No NVIDIA GPU? Use the CPU build instead:
# .\.venv\Scripts\python.exe -m pip install torch torchvision

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

Check the installation:

```powershell
.\.venv\Scripts\python.exe -c "import torch, cv2, ultralytics; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
```

> **Do not upgrade OpenCV to 5.x.** OpenCV 5 removed `cv2.HOGDescriptor`, which the HOG
> pipeline uses. `requirements.txt` pins `opencv-python<5`.

### 3. Download the dataset

Download the
[Beach Bird Detection Dataset](https://www.kaggle.com/datasets/liugalhigh/beach-bird)
from Kaggle and arrange it like this:

```text
beach_bird_project/
└── data/
    └── Beach Bird Detection Dataset v1.0/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── dataset.yaml
```

Or let `kagglehub` download it:

```powershell
.\.venv\Scripts\python.exe -m pip install kagglehub
.\.venv\Scripts\python.exe -c "import kagglehub; print(kagglehub.dataset_download('liugalhigh/beach-bird'))"
```

Move the downloaded folder so that the path above exists. If your folder name is different,
change one line at the top of `src/common.py`:

```python
DATA_ROOT = PROJECT_ROOT / "data" / "Beach Bird Detection Dataset v1.0"
```

The dataset is already split into **1,369 train / 291 validation / 240 test** images. Do not
re-split it; the official split is source-group-aware and prevents leakage.

### 4. Sanity check (about 10 seconds)

```powershell
.\.venv\Scripts\python.exe scripts\00_explore_dataset.py
```

This prints dataset statistics and writes:

```text
results/metrics/dataset_stats.json
results/figures/fig_dataset_examples.png
results/figures/fig_dataset_stats.png
```

If this works, your environment and dataset path are correct.

### 5. Train a tiny YOLO model (about 2 minutes)

This is only to prove that training works. Its accuracy will be poor.

```powershell
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolov8n.pt --name smoke --epochs 2 --batch 8
```

When it finishes you should see:

```text
Results saved to models/runs/smoke
```

For real results, continue with the commands in
[Run one method end to end](#run-one-method-end-to-end) or
[Reproduce all experiments](#reproduce-all-experiments).

---

## Dataset

| Split | Images | Images with birds | Images without birds | Bird boxes |
|---|---:|---:|---:|---:|
| Train | 1,369 | 1,215 | 154 | 8,663 |
| Validation | 291 | 258 | 33 | 1,843 |
| Test | 240 | 213 | 27 | 1,518 |

Key properties:

- All images are **512 x 512**.
- One class: `bird`.
- **34.7%** of boxes are smaller than 32 x 32 pixels.
- Some images contain more than 100 birds.
- 214 images contain no birds at all; these are used to measure false alarms.

Labels use the standard YOLO format:

```text
class_id x_center y_center width height
```

All coordinates are normalised to 0-1. An empty `.txt` file means "this image has no birds".

---

## Repository layout

```text
beach_bird_project/
├── README.md
├── requirements.txt
│
├── src/                         # reusable algorithms and metrics
│   ├── common.py                # dataset reader, IoU, NMS, AP, size/density breakdowns
│   ├── classical.py             # non-learning detectors
│   └── hog_svm.py               # HOG/LBP features, sliding window, SVM/RF training helpers
│
├── scripts/                     # command-line entry points
│   ├── 00_explore_dataset.py
│   ├── 01_tune_classical.py
│   ├── 02_run_classical.py
│   ├── 03_train_hog_svm.py
│   ├── 03_run_hog_svm.py
│   ├── 03_run_hog_proposals.py
│   ├── 03_make_hog_table.py
│   ├── 04_train_yolo.py
│   ├── 05_predict_yolo.py
│   ├── 06_evaluate_all.py
│   ├── 06_model_info.py
│   └── 07_make_figures.py
│
├── report/                      # written report
│   ├── report_template.html
│   ├── report.css
│   ├── build_report.py
│   └── COMP9517_Group_Project_Report.docx
│
├── data/                        # download separately; not committed
├── models/                      # trained weights; created automatically; not committed
└── results/                     # predictions, metrics, tables and figures; created automatically
```

**`src/` contains the algorithms. `scripts/` contains the commands you run.**  
The two are separated so that the same evaluation code is used for every method.

---

## Where each method lives

This is the most important section if you want to modify or extend the project.

### 1. Non-learning methods

| Item | File |
|---|---|
| Implementation | `src/classical.py` |
| Parameter class | `ClassicalParams` in `src/classical.py` |
| Single-image API | `detect_image(image, variant, params)` |
| Tuning on validation data | `scripts/01_tune_classical.py` |
| Run on train/val/test | `scripts/02_run_classical.py` |
| Trained model? | **No training at all** |
| Raw predictions | `results/predictions/classical_<variant>_<split>.json` |
| Per-method metrics | `results/metrics/classical_<variant>_<split>.json` |

The three variants are:

```text
global_otsu
multiscale_blackhat
multiscale_blackhat_ws
```

Run all three:

```powershell
.\.venv\Scripts\python.exe scripts\01_tune_classical.py --variant multiscale_blackhat --subset 120

.\.venv\Scripts\python.exe scripts\02_run_classical.py --variant all --split val  --use-best
.\.venv\Scripts\python.exe scripts\02_run_classical.py --variant all --split test --use-best
```

Use it from Python:

```python
import cv2
from src.classical import ClassicalParams, detect_image

image = cv2.imread("some_image.jpg")
boxes = detect_image(image, "multiscale_blackhat_ws", ClassicalParams())
# boxes: (N, 5) array -> [x1, y1, x2, y2, score]
```

### 2. HOG / LBP + SVM (machine learning)

| Item | File |
|---|---|
| Implementation | `src/hog_svm.py` |
| Feature extraction | `FeatureExtractor` in `src/hog_svm.py` |
| Sliding-window detector | `WindowDetector` in `src/hog_svm.py` |
| Training data + hard negatives | `build_training_data`, `mine_hard_negatives` in `src/hog_svm.py` |
| Train the classifier | `scripts/03_train_hog_svm.py` |
| Predict with dense sliding windows | `scripts/03_run_hog_svm.py` |
| Predict with classical proposals (recommended) | `scripts/03_run_hog_proposals.py` |
| Build the HOG ablation table | `scripts/03_make_hog_table.py` |
| Trained model files | `models/hog_svm/<name>.joblib` |
| Raw predictions | `results/predictions/hog_<tag>_<split>.json` |
| Per-method metrics | `results/metrics/hog_<tag>_<split>.json` |

Train the three main variants:

```powershell
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_linear     --feature hog     --classifier linear --hard-neg 6000
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_lbp_linear --feature hog_lbp --classifier linear --hard-neg 6000
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_rf         --feature hog     --classifier rf     --hard-neg 6000
```

This writes:

```text
models/hog_svm/hog_linear.joblib
models/hog_svm/hog_lbp_linear.joblib
models/hog_svm/hog_rf.joblib
```

Predict with the recommended proposal cascade:

```powershell
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_linear     --split val  --tag hog_prop
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_linear     --split test --tag hog_prop

.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_lbp_linear --split val  --tag hog_lbp_prop
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_lbp_linear --split test --tag hog_lbp_prop

.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_rf         --split val  --tag hog_rf_prop
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_rf         --split test --tag hog_rf_prop
```

The pure sliding-window version is kept for comparison:

```powershell
.\.venv\Scripts\python.exe scripts\03_run_hog_svm.py --name hog_linear --split val  --stride 16 --score-threshold 0 --box-shrink 1.0 --tag hog_linear
.\.venv\Scripts\python.exe scripts\03_run_hog_svm.py --name hog_linear --split test --stride 16 --score-threshold 0 --box-shrink 1.0 --tag hog_linear
```

Use it from Python:

```python
import joblib
from src.hog_svm import FeatureExtractor, HogParams, WindowDetector

checkpoint = joblib.load("models/hog_svm/hog_linear.joblib")
extractor = FeatureExtractor(64, checkpoint["feature"]["use_hog"], checkpoint["feature"]["use_lbp"])
detector = WindowDetector(checkpoint["model"], extractor, HogParams())
boxes = detector.detect(image)   # (N, 5): [x1, y1, x2, y2, score]
```

### 3. YOLO (deep learning)

YOLO itself is provided by the
[Ultralytics](https://github.com/ultralytics/ultralytics) package. This repository contains
the training and prediction wrappers.

| Item | File |
|---|---|
| Training wrapper | `scripts/04_train_yolo.py` |
| Prediction wrapper | `scripts/05_predict_yolo.py` |
| Training configuration | command-line flags (`--model`, `--imgsz`, `--epochs`, `--batch`, `--aug`) |
| Trained weights | `models/runs/<name>/weights/best.pt` |
| Training logs and curves | `models/runs/<name>/` |
| Raw predictions | `results/predictions/yolo_<name>_<split>.json` |
| Per-run runtime info | `results/metrics/yolo_<name>_<split>_runtime.json` |

The five trained runs:

| Name | Model | Input | Augmentation |
|---|---|---|---|
| `yolov8n_plain` | YOLOv8n | 512 | none |
| `yolov8n_aug` | YOLOv8n | 512 | default |
| `yolo11n_aug` | YOLO11n | 512 | default |
| `yolov8s_aug` | YOLOv8s | 512 | default |
| `yolov8n_aug_960` | YOLOv8n | 960 | default |

Two extra inference experiments reuse the `yolov8n_aug` weights:

| Name | What it does | Extra flag |
|---|---|---|
| `yolov8n_aug_tta` | Test-time augmentation | `--tta` |
| `yolov8n_aug_tile2` | 2x2 tiled inference | `--tiles 2 --overlap 96 --imgsz 640` |

Train one model:

```powershell
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolov8n.pt --name yolov8n_aug --epochs 60 --batch 32
```

Predict on validation and test:

```powershell
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug --split val
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug --split test
```

TTA and tiled inference:

```powershell
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug_tta --split test --tta

.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug_tile2 --split test --tiles 2 --overlap 96 --imgsz 640
```

Use it directly from Python:

```python
from ultralytics import YOLO

model = YOLO("models/runs/yolov8n_aug/weights/best.pt")
results = model.predict("some_image.jpg", imgsz=512, conf=0.25, iou=0.7)[0]
boxes = results.boxes.xyxy.cpu().numpy()
scores = results.boxes.conf.cpu().numpy()
```

### 4. Evaluation and result generation

| Item | File |
|---|---|
| Metric implementation | `src/common.py` (`evaluate_ap`, `nms`, `box_iou`, ...) |
| Main evaluation | `scripts/06_evaluate_all.py` |
| Model size / parameter count | `scripts/06_model_info.py` |
| Figures | `scripts/07_make_figures.py` |
| Main results table | `results/tables/main_results.csv` |
| Size breakdown | `results/tables/size_breakdown.csv` |
| Density breakdown | `results/tables/density_breakdown.csv` |
| HOG ablation | `results/tables/hog_ablation.csv` |
| Full JSON summary | `results/metrics/summary_test.json` |
| Per-image TP/FP/FN | `results/metrics/per_image_<method>_test.json` |

`06_evaluate_all.py` does not read model weights. It reads the prediction JSON files, selects
a confidence threshold on **validation**, applies it unchanged to **test**, and writes all
tables. This is why every method must first produce both `*_val.json` and `*_test.json`.

---

## Run one method end to end

This is the shortest path to a real result. It trains YOLOv8n with augmentation, predicts on
validation and test, and evaluates it.

```powershell
# 1. Look at the data
.\.venv\Scripts\python.exe scripts\00_explore_dataset.py

# 2. Train (about 25-40 minutes on an RTX 3060 laptop)
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolov8n.pt --name yolov8n_aug --epochs 60 --batch 32

# 3. Predict
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug --split val
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug --split test

# 4. Evaluate and generate figures
.\.venv\Scripts\python.exe scripts\06_evaluate_all.py
.\.venv\Scripts\python.exe scripts\06_model_info.py
.\.venv\Scripts\python.exe scripts\07_make_figures.py
```

Open:

```text
results/tables/main_results.csv
results/figures/fig_pr_curves.png
```

You should see `YOLOv8n + aug` with AP@0.5 around **0.87**.

> `06_evaluate_all.py` uses a fixed list of method names at the top of the file. If you use a
> new `--name` or `--tag`, add it to that list, otherwise the method is skipped.

---

## Reproduce all experiments

The full benchmark is five YOLO models, three HOG classifiers, and three classical variants.
It takes roughly 1.5-2.5 hours on the RTX 3060 used here.

### Step 1: dataset statistics

```powershell
.\.venv\Scripts\python.exe scripts\00_explore_dataset.py
```

### Step 2: classical methods

```powershell
.\.venv\Scripts\python.exe scripts\01_tune_classical.py --variant multiscale_blackhat --subset 120

.\.venv\Scripts\python.exe scripts\02_run_classical.py --variant all --split val  --use-best
.\.venv\Scripts\python.exe scripts\02_run_classical.py --variant all --split test --use-best
```

### Step 3: HOG models

```powershell
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_linear     --feature hog     --classifier linear --hard-neg 6000
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_lbp_linear --feature hog_lbp --classifier linear --hard-neg 6000
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_rf         --feature hog     --classifier rf     --hard-neg 6000

# Ablation-only models:
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_no_hard --feature hog --classifier linear --hard-neg 0
.\.venv\Scripts\python.exe scripts\03_train_hog_svm.py --name hog_linear2 --feature hog --classifier linear --hard-neg 20000 --hard-images 500 --neg-per-image 24

.\.venv\Scripts\python.exe scripts\03_run_hog_svm.py --name hog_linear --split val  --stride 16 --score-threshold 0 --box-shrink 1.0 --tag hog_linear
.\.venv\Scripts\python.exe scripts\03_run_hog_svm.py --name hog_linear --split test --stride 16 --score-threshold 0 --box-shrink 1.0 --tag hog_linear

.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_linear     --split val  --tag hog_prop
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_linear     --split test --tag hog_prop

.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_lbp_linear --split val  --tag hog_lbp_prop
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_lbp_linear --split test --tag hog_lbp_prop

.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_rf         --split val  --tag hog_rf_prop
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_rf         --split test --tag hog_rf_prop

# Validation run used by the HOG ablation table
.\.venv\Scripts\python.exe scripts\03_run_hog_proposals.py --model hog_no_hard --split val --tag hog_nohard_prop

.\.venv\Scripts\python.exe scripts\03_make_hog_table.py
```

### Step 4: YOLO models

```powershell
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolov8n.pt --name yolov8n_plain  --epochs 60 --batch 32 --aug plain
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolov8n.pt --name yolov8n_aug    --epochs 60 --batch 32 --aug default
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolo11n.pt --name yolo11n_aug    --epochs 60 --batch 32 --aug default
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolov8s.pt --name yolov8s_aug    --epochs 60 --batch 16 --aug default
.\.venv\Scripts\python.exe scripts\04_train_yolo.py --model yolov8n.pt --name yolov8n_aug_960 --epochs 40 --batch 4 --imgsz 960 --aug default
```

Predict for all five:

```powershell
$models = @("yolov8n_plain", "yolov8n_aug", "yolo11n_aug", "yolov8s_aug", "yolov8n_aug_960")

foreach ($m in $models) {
    .\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights "models\runs\$m\weights\best.pt" --name $m --split val
    .\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights "models\runs\$m\weights\best.pt" --name $m --split test
}
```

The two post-hoc experiments:

```powershell
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug_tta --split val --tta
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug_tta --split test --tta

.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug_tile2 --split val --tiles 2 --overlap 96 --imgsz 640
.\.venv\Scripts\python.exe scripts\05_predict_yolo.py --weights models\runs\yolov8n_aug\weights\best.pt --name yolov8n_aug_tile2 --split test --tiles 2 --overlap 96 --imgsz 640
```

### Step 5: evaluate everything

```powershell
.\.venv\Scripts\python.exe scripts\06_evaluate_all.py
.\.venv\Scripts\python.exe scripts\06_model_info.py
.\.venv\Scripts\python.exe scripts\07_make_figures.py
```

---

## Outputs explained

### Prediction JSON

Every detector writes the same format, so one evaluator can read all of them:

```json
{
  "meta": {
    "method": "yolo_yolov8n_aug",
    "split": "test",
    "ms_per_image": 23.0
  },
  "predictions": {
    "beach_bird_0001": [
      [x1, y1, x2, y2, score],
      [x1, y1, x2, y2, score]
    ],
    "beach_bird_0002": []
  }
}
```

- Keys are image IDs (file name without extension).
- Coordinates are absolute pixels in the original 512 x 512 image.
- `score` is the confidence.
- Images with no detections keep an empty list `[]`.

These files are **raw predictions**, not final answers. The confidence threshold is applied
later by `06_evaluate_all.py` on validation data.

### Main CSV

`results/tables/main_results.csv` columns:

| Column | Meaning |
|---|---|
| `AP@0.5` | AP at IoU 0.5 (the main metric) |
| `AP@0.75` | AP at the stricter IoU 0.75 |
| `AP@0.5:0.95` | COCO-style AP averaged over 10 IoU thresholds |
| `precision`, `recall`, `f1` | Values at the validation-selected threshold |
| `count_mae` | Mean absolute error in bird counts per image |
| `neg_fp_per_image` | False positives per bird-free image |
| `ms_per_image` | Inference time per image |
| `train_minutes` | Training time |

### Figures

| File | What it shows |
|---|---|
| `fig_dataset_examples.png` | Example images at five density levels |
| `fig_dataset_stats.png` | Box-size, density and brightness distributions |
| `fig_pr_curves.png` | Precision-recall curves for all methods |
| `fig_main_ap.png` | AP@0.5 bar chart |
| `fig_ap_vs_time.png` | Accuracy versus inference time |
| `fig_size_ap.png` | AP by small / medium / large objects |
| `fig_density_recall.png` | Recall versus flock density |
| `fig_qualitative.png` | Ground truth versus YOLO predictions |
| `fig_failures.png` | False positives and missed birds |
| `fig_yolo_curves.png` | Validation mAP during YOLO training |
| `fig_hog_ablation.png` | HOG feature / classifier / proposal ablations |
| `fig_tta_tile.png` | Baseline versus TTA versus tiling |

---

## Troubleshooting

### `python` prints nothing or opens the Microsoft Store

This is the Windows Store alias, not a real Python. Use the virtual-environment interpreter
directly:

```powershell
.\.venv\Scripts\python.exe --version
```

If `.venv` does not exist yet, install Python from <https://www.python.org/downloads/> and
tick **Add python.exe to PATH** during installation.

### `ModuleNotFoundError: No module named 'src'`

Run the command from the repository root (`beach_bird_project`), not from inside `scripts/`.

### `cv2 has no attribute 'HOGDescriptor'`

OpenCV 5 was installed. Reinstall OpenCV 4:

```powershell
.\.venv\Scripts\python.exe -m pip install "opencv-python==4.12.0.88"
```

### `CUDA out of memory`

Reduce the batch size, then the image size:

```text
--batch 32 -> 16 -> 8 -> 4
--imgsz 960 -> 640 -> 512
```

### Dataset not found

Check that this exact folder exists:

```text
data/Beach Bird Detection Dataset v1.0/images/train
```

If your folder has a different name, edit `DATA_ROOT` in `src/common.py`.

### `06_evaluate_all.py` prints `[skip]` for a method

The prediction file name does not match the `prefix` in the `METHODS` list at the top of
`06_evaluate_all.py`. Use the default `--name` / `--tag` from this README, or add your own
entry to that list.

### Training is very slow

- Check that CUDA is available:
  ```powershell
  .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
  ```
- On CPU, reduce `--epochs`, use `yolov8n`, and use a smaller `--imgsz`.

---

## Hardware

The reported numbers were produced on:

- NVIDIA RTX 3060 Laptop GPU, 6 GB
- 16 GB RAM
- Windows 11
- Python 3.11, PyTorch 2.14 + CUDA 12.6, Ultralytics 8.4, OpenCV 4.12

YOLO training used automatic mixed precision. HOG and classical methods run on the CPU.

Approximate runtimes:

| Step | Time |
|---|---|
| Dataset statistics | 10 seconds |
| Classical tuning + val/test | 1-3 minutes |
| One HOG model (train + validate) | 3-6 minutes |
| YOLOv8n, 60 epochs, 512 px | 25-40 minutes |
| All five YOLO models | 1.5-2.5 hours |
| Full evaluation and figures | 1-3 minutes |

---

## Pretrained weights

Model weights are **not committed** to this repository, because they are large and the
assignment limits the code archive size.

- YOLO weights are written to `models/runs/<name>/weights/best.pt`.
- HOG models are written to `models/hog_svm/<name>.joblib`.
- Classical methods have no model file.

Train them with the commands above. Everything in `models/`, `data/` and
`results/predictions/` is excluded by `.gitignore`.

---

## Acknowledgements

- Dataset: Kai Liu, *Beach Bird Detection Dataset v1.0*, Kaggle, 2026.
- Detectors: [Ultralytics YOLOv8 / YOLO11](https://github.com/ultralytics/ultralytics).
- Classical features: OpenCV `HOGDescriptor`, scikit-image LBP, scikit-learn SVM/RF.
- Course: COMP9517 Computer Vision, UNSW, 2026 Term 3.

## License

The code is released for coursework and research use. The dataset remains subject to its own
Kaggle license; see `data/Beach Bird Detection Dataset v1.0/LICENSE.md`.
