# doc_loader.py
import os
import io
from typing import Dict, Tuple

import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from PIL import Image


def _extract_text_digital_pdf(path: str) -> str:
    parts = []
    try:
        with fitz.open(path) as pdf:
            for page in pdf:
                txt = page.get_text("text") or ""
                if txt.strip():
                    parts.append(txt.strip())
    except Exception:
        return ""
    return "\n\n".join(parts).strip()


def _extract_text_scanned_pdf(path: str, zoom: float = 2.0, lang: str = "eng") -> str:
    out = []
    try:
        with fitz.open(path) as pdf:
            mat = fitz.Matrix(zoom, zoom)
            for page in pdf:
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang=lang) or ""
                if ocr_text.strip():
                    out.append(ocr_text.strip())
    except Exception:
        return ""
    return "\n\n".join(out).strip()


def extract_text_from_pdf(path: str, lang: str = "eng") -> str:
    digital = _extract_text_digital_pdf(path)
    if len(digital) >= 200:
        return digital
    return _extract_text_scanned_pdf(path, zoom=2.0, lang=lang)


def extract_tables_from_pdf(path: str) -> str:
    # Camelot first (lattice → stream)
    try:
        blocks = []
        for flavor in ("lattice", "stream"):
            try:
                tbs = camelot.read_pdf(path, pages="all", flavor=flavor)
                if tbs and len(tbs) > 0:
                    for i, tb in enumerate(tbs):
                        rows = [" | ".join(map(str, row)) for row in tb.df.values.tolist()]
                        blocks.append(f"Table ({flavor}) {i+1}:\n" + "\n".join(rows))
                    break
            except Exception:
                continue
        if blocks:
            return "\n\n".join(blocks).strip()
    except Exception:
        pass

    # Fallback to pdfplumber
    try:
        blocks = []
        with pdfplumber.open(path) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for t_i, table in enumerate(tables, start=1):
                    rows = [" | ".join([c if c is not None else "" for c in row]) for row in table]
                    blocks.append(f"Table (plumber) p.{pageno} #{t_i}:\n" + "\n".join(rows))
        return "\n\n".join(blocks).strip()
    except Exception:
        return ""


def extract_layout_text_pdf(path: str) -> str:
    try:
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                if txt.strip():
                    parts.append(txt.strip())
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def extract_figures_and_captions_pdf(path: str) -> str:
    try:
        out = []
        with pdfplumber.open(path) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                images = page.images or []
                for idx, im in enumerate(images, start=1):
                    x0, y0, x1, y1 = float(im.get("x0", 0)), float(im.get("top", 0)), float(im.get("x1", 0)), float(im.get("bottom", 0))
                    ph = float(page.height)
                    band_top = min(y1 + 5, ph)
                    band_bottom = min(y1 + 60, ph)
                    cap = ""
                    try:
                        band = page.within_bbox((x0, band_top, x1, band_bottom))
                        cap = (band.extract_text() or "").strip()
                    except Exception:
                        pass
                    out.append(f"Figure p.{pageno} #{idx} bbox=({int(x0)},{int(y0)},{int(x1)},{int(y1)})\nCaption: {cap or 'N/A'}")
        return "\n\n".join(out).strip()
    except Exception:
        return ""


def extract_text_from_image(path: str, lang: str = "eng") -> str:
    try:
        img = Image.open(path)
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()
    except Exception:
        return ""


def load_document(path: str, return_sections: bool = False, lang: str = "eng"):
    ext = os.path.splitext(path)[-1].lower()
    sections: Dict[str, str] = {}

    if ext == ".pdf":
        sections["Text"] = extract_text_from_pdf(path, lang=lang)
        tbl = extract_tables_from_pdf(path)
        if tbl:
            sections["Tables"] = tbl
        layout = extract_layout_text_pdf(path)
        if layout:
            sections["Layout"] = layout
        figs = extract_figures_and_captions_pdf(path)
        if figs:
            sections["Figures"] = figs

    elif ext in [".jpg", ".jpeg", ".png"]:
        sections["OCR"] = extract_text_from_image(path, lang=lang)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    merged = "\n\n".join([v for v in sections.values() if v and v.strip()]).strip()
    if return_sections:
        return merged, sections
    return merged
