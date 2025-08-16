# app.py
import os
import json
import time
import tempfile
import streamlit as st
# --- Hugging Face Spaces / Linux runtime helpers ---
import shutil
import pytesseract

# Ensure Tesseract and Poppler are discoverable in the Space
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"
os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/bin:/usr/local/bin"

# Prefer a public, lightweight embedding model to avoid auth/gated downloads
# (If your rag_pipeline.py already sets this, you can remove the next 4 lines.)
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    os.environ.setdefault("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    pass



from doc_loader import load_document
from figure_extractor import extract_figures
from rag_pipeline import (
    build_rag_pipeline,
    query_rag_full,
    evaluate_rag,
)

st.set_page_config(page_title="📑 Visual Document RAG", layout="wide")
st.title("📑 Visual Document RAG ")

domain = st.selectbox(
    "Domain focus",
    ["Finance", "Healthcare", "Law", "Education", "Multimodal"],
    index=0,
    help="Used to lightly steer retrieval/answers",
)

uploaded_file = st.file_uploader("📂 Upload a PDF or Image", type=["pdf", "png", "jpg"])
query = st.text_input("🔎 Ask a question about the document:")
summarize = st.checkbox("📝 Summarize the whole document")

if uploaded_file:
    # persist to temp path for libs
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document..."):
        docs_text, sections = load_document(tmp_path, return_sections=True)

    # Optional figure extraction (PDF only)
    figures_meta = []
    extra_docs = []
    if uploaded_file.type == "application/pdf":
        try:
            figures_meta = extract_figures(tmp_path, out_dir="figures", lang="eng") or []
            if figures_meta:
                sections["Figures (OCR+captions)"] = "\n\n".join(
                    [
                        f"p.{f['page']} — {f.get('caption') or '(no caption)'}\n"
                        f"{(f.get('ocr_text') or '')[:200]}"
                        for f in figures_meta
                    ]
                )
                # vectorizable figure docs
                for f in figures_meta:
                    content = (
                        f"FIGURE p.{f['page']}: {f.get('caption') or ''}\n"
                        f"OCR: {f.get('ocr_text') or ''}\n"
                        f"TAGS: {' '.join(f.get('tags', []))}"
                    ).strip()
                    metadata = {
                        "type": "figure",
                        "page": f.get("page"),
                        "path": f.get("path"),
                        "caption": f.get("caption"),
                        "tags": f.get("tags", []),
                    }
                    extra_docs.append({"content": content, "metadata": metadata})
        except Exception as e:
            st.warning(f"Figure extraction skipped: {e}")

    # Build vector index (now indexes figures too)
    if docs_text.strip():
        db = build_rag_pipeline(docs_text, extra_docs=extra_docs or None)
        st.success(f"✅ Document indexed! (Domain: {domain})")

        # Show extracted sections
        with st.expander("📂 Extracted Document Content"):
            tab_names = list(sections.keys())
            tabs = st.tabs(tab_names)
            for i, name in enumerate(tab_names):
                with tabs[i]:
                    st.text_area(f"{name}", sections[name], height=230)

        # Actions (Summarize or Q&A)
        answer_text, retrieved_docs, latency = None, None, None
        colA, colB = st.columns([1, 1])

        with colA:
            if summarize:
                if st.button("📝 Summarize Document"):
                    start = time.time()
                    q = f"[Domain: {domain}] Summarize this document briefly with key points and numbers."
                    answer_text, _, retrieved_docs = query_rag_full(
                        db, q, domain=domain
                    )
                    latency = round(time.time() - start, 3)
            else:
                if query and st.button("💡 Get Answer"):
                    start = time.time()
                    q = f"[Domain: {domain}] {query}"
                    answer_text, _, retrieved_docs = query_rag_full(
                        db, q, domain=domain
                    )
                    latency = round(time.time() - start, 3)

        # Show results
        if answer_text is not None:
            st.subheader("💡 Answer")
            st.write(answer_text)
            st.caption(f"⏱️ Latency: {latency}s")

            with st.expander("🔍 Retrieved Contexts"):
                if retrieved_docs:
                    for i, d in enumerate(retrieved_docs, 1):
                        meta = getattr(d, "metadata", {}) or {}
                        if meta.get("type") == "figure" and meta.get("path"):
                            st.write(
                                f"Figure (page {meta.get('page')}): "
                                f"{meta.get('caption') or '(no caption)'}"
                            )
                            st.image(meta["path"], use_container_width=True)
                        else:
                            st.info(f"Chunk {i}:\n\n{d.page_content}")

            # Evaluation
            st.markdown("---")
            st.subheader("📊 Evaluation")
            if st.button("Evaluate (LLM-based: Faithfulness & Relevancy)"):
                try:
                    raw = evaluate_rag(
                        answer_text,
                        [d.page_content for d in (retrieved_docs or [])],
                        f"[Domain: {domain}] {query or 'Summary'}",
                    )
                    try:
                        payload = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception:
                        payload = {"raw": raw}
                    st.json(payload)
                    st.download_button(
                        "⬇️ Download Evaluation JSON",
                        data=json.dumps(payload, indent=2),
                        file_name="evaluation.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.warning(f"Evaluation unavailable: {e}")
    else:
        st.error("❌ No text could be extracted from the uploaded file.")
