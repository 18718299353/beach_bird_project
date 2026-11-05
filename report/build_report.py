"""
build_report.py —— 把 report_template.html + 实验结果 渲染成最终 PDF
====================================================================

用法：
    python report/build_report.py

流程：
1. 读取 results/tables/*.csv、results/metrics/*.json；
2. 生成 HTML 表格，替换模板里的 {{TOKEN}}；
3. 写出 report/report.html；
4. 调用 Microsoft Edge headless 打印成 A4 两栏 PDF；
5. 用 PyMuPDF 输出页数（作业限制正文 ≤10 页）。

没有安装 Edge 时，可以手动用浏览器打开 report.html 并"打印为 PDF"。
"""

from __future__ import annotations

import csv
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
TABLE_DIR = ROOT / "results" / "tables"
METRIC_DIR = ROOT / "results" / "metrics"


def read_csv(name: str) -> List[Dict]:
    p = TABLE_DIR / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fnum(x, nd=3, default="—"):
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return default


def ms(x):
    try:
        return f"{float(x):.0f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def table_dataset() -> str:
    d = read_json(METRIC_DIR / "dataset_stats.json", {})
    if not d:
        return "<p><i>Dataset statistics not found.</i></p>"
    rows = []
    for split in ("train", "val", "test"):
        s = d.get(split, {})
        rows.append(
            f"<tr><td class='left'>{split.capitalize()}</td><td>{s.get('images','—')}</td>"
            f"<td>{s.get('positive_images','—')}</td><td>{s.get('negative_images','—')}</td>"
            f"<td>{s.get('objects','—')}</td><td>{s.get('boxes_per_image_max','—')}</td>"
            f"<td>{float(s.get('box_width_px',{}).get('p50',0)):.1f}</td></tr>")
    total = d.get("all", {})
    return f"""
<table>
<thead><tr><th class='left'>Split</th><th>Images</th><th>Pos.</th><th>Neg.</th>
<th>Boxes</th><th>Max/img</th><th>Median side (px)</th></tr></thead>
<tbody>{''.join(rows)}
<tr class='rule'><td class='left'>All</td><td>{total.get('images','—')}</td><td>—</td><td>—</td>
<td>{total.get('objects','—')}</td><td>130</td><td>{float(total.get('box_area_px2',{}).get('p50',0))**0.5:.1f}</td></tr>
</tbody></table>
<p class="note">Table I. Dataset composition. “Pos.” and “Neg.” count images with at least one bird and with no birds, respectively.</p>
"""


# 主表里展示的方法及短名
MAIN_ORDER = [
    ("Global Otsu", "Global Otsu"),
    ("Multi-scale Black-hat", "Black-hat"),
    ("Black-hat + Watershed", "Black-hat+WS"),
    ("HOG + Linear SVM (sliding)", "HOG slide"),
    ("HOG + SVM (proposals)", "HOG prop"),
    ("HOG+LBP + SVM (proposals)", "HOG+LBP prop"),
    ("HOG + RF (proposals)", "HOG+RF prop"),
    ("YOLOv8n (no aug)", "YOLOv8n plain"),
    ("YOLOv8n + aug", "YOLOv8n"),
    ("YOLO11n + aug", "YOLO11n"),
    ("YOLOv8s + aug", "YOLOv8s"),
    ("YOLOv8n + aug @960", "YOLOv8n@960"),
    ("YOLOv8n + aug + TTA", "YOLOv8n + TTA"),
    ("YOLOv8n + aug + 2×2 tiles", "YOLOv8n + tiles"),
]


def table_main() -> str:
    rows = read_csv("main_results.csv")
    if not rows:
        return "<p><i>Main results not found — run scripts/06_evaluate_all.py.</i></p>"
    by_name = {r["method"]: r for r in rows}
    body = []
    prev_family = None
    for full, short in MAIN_ORDER:
        r = by_name.get(full)
        if not r:
            continue
        fam = r["family"]
        rule = ""
        if prev_family is not None and fam != prev_family:
            rule = " class='rule'"
        prev_family = fam
        body.append(
            f"<tr{rule}><td class='left'>{html.escape(short)}</td>"
            f"<td>{fnum(r['AP@0.5'])}</td><td>{fnum(r['AP@0.75'])}</td>"
            f"<td>{fnum(r['precision'])}</td><td>{fnum(r['recall'])}</td>"
            f"<td>{fnum(r['f1'])}</td><td>{fnum(r['neg_fp_per_image'],2)}</td>"
            f"<td>{ms(r['ms_per_image'])}</td></tr>")
    return f"""
<table>
<thead><tr><th class='left'>Method</th><th>AP<sub>50</sub></th><th>AP<sub>75</sub></th>
<th>P</th><th>R</th><th>F1</th><th>FP<sub>neg</sub></th><th>ms</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p class="note">Table II. Test-set results. The operating threshold for P/R/F1 is selected on validation data. FP<sub>neg</sub> is the mean number of detections per bird-free image. Rows are grouped by family.</p>
"""


def table_hog() -> str:
    rows = read_csv("hog_ablation.csv")
    if not rows:
        return "<p><i>HOG ablation table not found.</i></p>"
    body = []
    for r in rows:
        body.append(
            f"<tr><td class='left'>{html.escape(r.get('factor',''))}</td>"
            f"<td class='left'>{html.escape(r.get('setting',''))}</td>"
            f"<td>{fnum(r.get('AP@0.5'))}</td>"
            f"<td>{fnum(r.get('AP_small'))}</td>"
            f"<td>{ms(r.get('ms_per_image'))}</td></tr>")
    return f"""
<table>
<thead><tr><th class='left'>Factor</th><th class='left'>Setting</th><th>AP<sub>50</sub></th>
<th>AP<sub>small</sub></th><th>ms</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p class="note">Table III. HOG ablation on the validation split. AP<sub>small</sub> is AP@0.5 restricted to boxes below 32² px.</p>
"""


def table_yolo() -> str:
    rows = read_csv("main_results.csv")
    sizes = read_csv("size_breakdown.csv")
    info = read_json(METRIC_DIR / "model_info.json", {}) or {}
    if not rows:
        return "<p><i>YOLO results not found.</i></p>"
    by_name = {r["method"]: r for r in rows}
    size_map = {}
    for s in sizes:
        if s["group"].startswith("small"):
            size_map[s["method"]] = s.get("AP@0.5")
    order = [
        ("YOLOv8n (no aug)", "YOLOv8n", "512", "none"),
        ("YOLOv8n + aug", "YOLOv8n", "512", "default"),
        ("YOLO11n + aug", "YOLO11n", "512", "default"),
        ("YOLOv8s + aug", "YOLOv8s", "512", "default"),
        ("YOLOv8n + aug @960", "YOLOv8n", "960", "default"),
    ]
    key_map = {"YOLOv8n (no aug)": "yolov8n_plain", "YOLOv8n + aug": "yolov8n_aug",
               "YOLO11n + aug": "yolo11n_aug", "YOLOv8s + aug": "yolov8s_aug",
               "YOLOv8n + aug @960": "yolov8n_aug_960"}
    body = []
    for full, model, imgsz, aug in order:
        r = by_name.get(full)
        if not r:
            continue
        k = key_map[full]
        params = info.get(k, {}).get("params_millions")
        body.append(
            f"<tr><td class='left'>{html.escape(full)}</td><td>{model}</td><td>{imgsz}</td>"
            f"<td>{aug}</td><td>{fnum(params,2)}</td><td>{fnum(r['AP@0.5'])}</td>"
            f"<td>{fnum(r['AP@0.75'])}</td><td>{fnum(size_map.get(full))}</td>"
            f"<td>{ms(r['ms_per_image'])}</td></tr>")
    return f"""
<table>
<thead><tr><th class='left'>Run</th><th>Model</th><th>px</th><th>Aug</th>
<th>Params (M)</th><th>AP<sub>50</sub></th><th>AP<sub>75</sub></th>
<th>AP<sub>small</sub></th><th>ms</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p class="note">Table IV. Controlled YOLO comparison on the test set. All runs use pretrained COCO weights and the same optimisation schedule; only the listed factor changes.</p>
"""


def table_posthoc() -> str:
    rows = read_csv("main_results.csv")
    sizes = read_csv("size_breakdown.csv")
    if not rows:
        return "<p><i>Post-hoc results not found.</i></p>"
    by_name = {r["method"]: r for r in rows}
    size_map = {}
    for s in sizes:
        if s["group"].startswith("small"):
            size_map[s["method"]] = s.get("AP@0.5")
    names = [("YOLOv8n + aug", "Baseline"), ("YOLOv8n + aug + TTA", "TTA"),
             ("YOLOv8n + aug + 2×2 tiles", "2×2 tiles")]
    body = []
    for full, short in names:
        r = by_name.get(full)
        if not r:
            continue
        body.append(
            f"<tr><td class='left'>{short}</td><td>{fnum(r['AP@0.5'])}</td>"
            f"<td>{fnum(r['AP@0.75'])}</td><td>{fnum(size_map.get(full))}</td>"
            f"<td>{ms(r['ms_per_image'])}</td></tr>")
    return f"""
<table>
<thead><tr><th class='left'>Inference</th><th>AP<sub>50</sub></th><th>AP<sub>75</sub></th>
<th>AP<sub>small</sub></th><th>ms</th></tr></thead>
<tbody>{''.join(body)}</tbody></table>
<p class="note">Table V. Inference-time improvements on the same YOLOv8n weights.</p>
"""


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
REF_MAP = {
    "REF_DATASET": "1", "REF_COCO": "2", "REF_HOG": "3", "REF_OTSU": "4",
    "REF_DPM": "5", "REF_LBP": "6", "REF_FRCNN": "7", "REF_YOLO": "8",
    "REF_YOLOV8": "9", "REF_YOLO11": "10", "REF_SAHI": "11",
    "REF_WETLAND": "12", "REF_BIGBIRD": "13", "REF_SURVEY": "14",
}


def main() -> None:
    template = (REPORT_DIR / "report_template.html").read_text(encoding="utf-8")
    rows = read_csv("main_results.csv")
    by_name = {r["method"]: r for r in rows} if rows else {}

    def get(method, key, nd=3):
        r = by_name.get(method)
        return fnum(r[key], nd) if r else "—"

    yolo_candidates = [m for m in by_name if m.startswith("YOLO")]
    yolo_ap = max([float(by_name[m]["AP@0.5"]) for m in yolo_candidates], default=float("nan"))
    hog_methods = [m for m in by_name if m.startswith("HOG")]
    hog_ap = max([float(by_name[m]["AP@0.5"]) for m in hog_methods], default=float("nan"))
    classical_methods = [m for m in by_name if "Otsu" in m or "Black-hat" in m]
    class_ap = max([float(by_name[m]["AP@0.5"]) for m in classical_methods], default=float("nan"))

    tokens = {
        "GROUP_ID": "X",
        "AUTHORS": "Member 1, Member 2, Member 3, Member 4, Member 5",
        "YOLO_AP": f"{yolo_ap:.3f}" if yolo_ap == yolo_ap else "—",
        "HOG_AP": f"{hog_ap:.3f}" if hog_ap == hog_ap else "—",
        "CLASSICAL_AP": f"{class_ap:.3f}" if class_ap == class_ap else "—",
        "BEST_AP": f"{max(yolo_ap, hog_ap, class_ap):.3f}" if rows else "—",
        "BEST_NEGFP": get("YOLOv8s + aug", "neg_fp_per_image", 2),
        "HOG_NEGFP": get("HOG+LBP + SVM (proposals)", "neg_fp_per_image", 2),
        "CLASSICAL_NEGFP": get("Multi-scale Black-hat", "neg_fp_per_image", 2),
        "TABLE_DATASET": table_dataset(),
        "TABLE_MAIN": table_main(),
        "TABLE_HOG": table_hog(),
        "TABLE_YOLO": table_yolo(),
        "TABLE_POSTHOC": table_posthoc(),
    }
    for k, v in REF_MAP.items():
        tokens[k] = v

    out = template
    for k, v in tokens.items():
        out = out.replace("{{" + k + "}}", v)

    # 如果还有未替换的 token，列出来方便排查
    import re
    left = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", out)))
    if left:
        print("[warn] unresolved tokens:", left)

    html_path = REPORT_DIR / "report.html"
    html_path.write_text(out, encoding="utf-8")
    print(f"[ok] wrote {html_path}")

    # 调用 Edge 打印 PDF
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    pdf_path = REPORT_DIR / "COMP9517_Group_Project_Report.pdf"
    if edge.exists():
        cmd = [str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[ok] wrote {pdf_path}")
    else:
        print("[warn] Edge not found; open report.html and print to PDF manually.")

    try:
        import pymupdf
        if pdf_path.exists():
            doc = pymupdf.open(pdf_path)
            print(f"[info] PDF pages: {doc.page_count} (main text should be <= 10)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
