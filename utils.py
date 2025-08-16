# utils.py
from __future__ import annotations

import io
import os
import tempfile
from typing import Tuple, Optional, Union

import cv2
import numpy as np
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader

# -------------------------------
# Small helpers
# -------------------------------

def _to_path(file_or_path: Union[str, bytes, os.PathLike, io.BufferedIOBase]) -> Tuple[str, Optional[str]]:
    """
    Ensure we have a filesystem path. If a file-like is provided, write it to a temp file.
    Returns (path, tmp_path); tmp_path is None if no temp file was created.
    """
    if isinstance(file_or_path, (str, bytes, os.PathLike)):
        return str(file_or_path), None
    # file-like → persist to a temp file
    suffix = ""
    try:
        name = getattr(file_or_path, "name", "")
        if isinstance(name, str) and "." in name:
            suffix = f".{name.rsplit('.', 1)[-1]}"
    except Exception:
        pass
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        # rewind if possible
        if hasattr(file_or_path, "seek"):
            try: file_or_path.seek(0)
            except Exception: pass
        tmp.write(file_or_path.read())
    finally:
        tmp.flush()
        tmp.close()
    return tmp.name, tmp.name


def _cleanup_tmp(tmp_path: Optional[str]) -> None:
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# -------------------------------
# PDF Text Extraction (PyPDF2 fast path)
# -------------------------------

def extract_text_from_pdf(file_or_path) -> str:
    """
    Extract plain text from a (digital) PDF using PyPDF2.
    Silent fallback (returns "") on failure to avoid polluting embeddings.
    """
    path, tmp = _to_path(file_or_path)
    try:
        reader = PdfReader(path)
        pages_text = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            except Exception:
                continue
        return "\n".join(pages_text).strip()
    except Exception:
        return ""
    finally:
        _cleanup_tmp(tmp)


# -------------------------------
# Image OCR
# -------------------------------

def extract_text_from_image(file_or_path, lang: str = "eng") -> str:
    """
    Basic OCR on an image (jpg/png). Handles file path or file-like.
    """
    path, tmp = _to_path(file_or_path)
    try:
        img = Image.open(path)
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()
    except Exception:
        return ""
    finally:
        _cleanup_tmp(tmp)


# -------------------------------
# Tables from PDF (Camelot stream)
# -------------------------------

def extract_tables_from_pdf(file_path: str) -> str:
    """
    Extract tables using Camelot (stream flavor).
    Returns a single plain text block.
    """
    try:
        import camelot
        tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
        extracted = []
        for i, tb in enumerate(tables):
            rows = [" | ".join(map(str, row)) for row in tb.df.values.tolist()]
            extracted.append(f"Table {i+1}:\n" + "\n".join(rows))
        return "\n\n".join(extracted).strip() if extracted else ""
    except Exception:
        return ""


# -------------------------------
# Layout-aware text (pdfplumber)
# -------------------------------

def extract_layout_text(file_path: str) -> str:
    """
    Preserve headings/paragraphs via pdfplumber tolerances.
    """
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                if txt.strip():
                    parts.append(txt.strip())
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


# -------------------------------
# Chart / Graph OCR (OpenCV + Tesseract)
# -------------------------------

def extract_chart_text(image_input: Union[str, np.ndarray], lang: str = "eng") -> str:
    """
    Extract textual info from charts/graphs using a robust preprocessing pipeline:
      - grayscale
      - morphological tophat (remove background)
      - adaptive threshold (handles light/dark themes)
      - median denoise
      - OCR with conservative psm
    Accepts a file path or a BGR numpy array (OpenCV).
    """
    try:
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input
        if img is None:
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # background suppression (tophat)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # adaptive threshold (robust to varying backgrounds)
        thr = cv2.adaptiveThreshold(
            tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        )

        # slight opening to remove specks, then median blur
        opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        denoised = cv2.medianBlur(opened, 3)

        # Some charts have light text on dark bg → try inverted too and pick longer text
        inverted = cv2.bitwise_not(denoised)

        cfg = "--oem 3 --psm 6"  # assume a block of text
        txt1 = pytesseract.image_to_string(denoised, lang=lang, config=config_str(cfg))
        txt2 = pytesseract.image_to_string(inverted, lang=lang, config=config_str(cfg))

        text = (txt1 or "")
        if len((txt2 or "").strip()) > len(text.strip()):
            text = txt2

        return text.strip()
    except Exception:
        return ""


def config_str(base: str) -> str:
    """
    Helper to make it obvious where to append tesseract configs later.
    """
    return base


# -------------------------------
# Image-Text correlation helper
# -------------------------------

def merge_image_with_caption(image_text: str, caption: str) -> str:
    """
    Combine OCR text + caption into a single blob for embedding.
    """
    image_text = (image_text or "").strip() or "No visible text"
    caption = (caption or "").strip() or "No caption"
    return f"Image Content: {image_text}\nCaption: {caption}"
