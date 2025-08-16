# figure_extractor.py
# figure_extractor.py
import os
import io
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image


def extract_figures(pdf_path: str, out_dir: str = "figures", lang: str = "eng") -> List[Dict[str, Any]]:
    """
    Detect bitmap figures in a PDF, crop them, OCR content, and read a caption band.
    Returns a list with metadata used for indexing + UI previews.
    """
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pl_doc:
        fz_doc = fitz.open(pdf_path)

        for pageno in range(len(fz_doc)):
            pl_page = pl_doc.pages[pageno]
            fz_page = fz_doc[pageno]

            images = pl_page.images or []
            if not images:
                continue

            page_h = float(pl_page.height)
            page_w = float(pl_page.width)

            for idx, im in enumerate(images, start=1):
                x0 = float(im.get("x0", 0))
                y0 = float(im.get("top", 0))
                x1 = float(im.get("x1", 0))
                y1 = float(im.get("bottom", 0))

                w = max(1.0, x1 - x0)
                h = max(1.0, y1 - y0)
                area = w * h
                page_area = page_w * page_h
                if area < 10000 or area < 0.01 * page_area:
                    continue  # skip tiny artifacts/icons

                try:
                    rect = fitz.Rect(x0, y0, x1, y1)
                    pix = fz_page.get_pixmap(clip=rect, alpha=False)
                    img_path = os.path.join(out_dir, f"page_{pageno+1}_fig_{idx}.png")
                    pix.save(img_path)
                except Exception:
                    continue

                ocr_text = ""
                try:
                    with open(img_path, "rb") as fh:
                        img = Image.open(io.BytesIO(fh.read()))
                        ocr_text = (pytesseract.image_to_string(img, lang=lang) or "").strip()
                except Exception:
                    pass

                caption = ""
                try:
                    band_top = min(y1 + 5, page_h)
                    band_bottom = min(y1 + 60, page_h)
                    if band_bottom > band_top:
                        band = pl_page.within_bbox((x0, band_top, x1, band_bottom))
                        caption = (band.extract_text() or "").strip()
                except Exception:
                    pass

                tags = ["figure"]
                low = (ocr_text + " " + caption).lower()
                if any(k in low for k in ["chart", "graph", "trend", "bar", "line", "pie"]):
                    tags.append("chart")
                if any(k in low for k in [
                    "revenue", "profit", "income", "eps", "cash flow",
                    "operating", "balance", "assets", "liabilities", "equity",
                    "ratio", "margin", "ebit", "ebitda"
                ]):
                    tags.append("finance")

                results.append({
                    "page": pageno + 1,
                    "bbox": (int(x0), int(y0), int(x1), int(y1)),
                    "path": img_path,
                    "caption": caption,
                    "ocr_text": ocr_text,
                    "tags": tags,
                })

    return results
