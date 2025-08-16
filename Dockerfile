# ---- Base image
FROM python:3.10-slim

# ---- System deps for OCR/PDF/CPU OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev poppler-utils \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---- Workdir & copy
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# ---- Python deps
RUN pip install --no-cache-dir -r requirements.txt

# ---- App files
COPY . /app

# ---- Streamlit must bind to the port Spaces gives via $PORT
ENV PORT=7860
EXPOSE 7860

# ---- Ensure tesseract is discoverable (matches your app.py)
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"

# ---- Run streamlit
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
