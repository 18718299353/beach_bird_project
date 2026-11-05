"""
build_manual_pdf.py —— 把中文 Markdown 使用文档转换成 PDF
=========================================================

用法：
    python docs/build_manual_pdf.py

依赖：
    pip install markdown
    Microsoft Edge（用于 headless 打印 PDF；也可以手动用浏览器打印）

输出：
    docs/使用文档_中文.html
    docs/使用文档_中文.pdf
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MD_PATH = DOCS / "使用文档_中文.md"
HTML_PATH = DOCS / "使用文档_中文.html"
PDF_PATH = DOCS / "使用文档_中文.pdf"


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    body = markdown.Markdown(extensions=["extra", "toc", "sane_lists"]).convert(text)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Beach Bird Detection 使用文档</title>
<link rel="stylesheet" href="manual.css">
</head><body>{body}</body></html>"""
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {HTML_PATH}")

    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if edge.exists():
        subprocess.run([str(edge), "--headless=new", "--disable-gpu",
                        "--no-pdf-header-footer", f"--print-to-pdf={PDF_PATH}",
                        HTML_PATH.as_uri()], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[ok] wrote {PDF_PATH}")
    else:
        print("[warn] Edge not found; open the HTML and print to PDF manually.")

    try:
        import pymupdf
        if PDF_PATH.exists():
            doc = pymupdf.open(PDF_PATH)
            print(f"[info] PDF pages: {doc.page_count}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
