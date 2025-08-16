# rag_pipeline.py
import os
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_community.vectorstores import DocArrayInMemorySearch


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Prefer FAISS if available; otherwise use a pure-Python fallback (no native build)
try:
    from langchain_community.vectorstores import FAISS
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from langchain_community.vectorstores import DocArrayInMemorySearch

# Embeddings + LLM
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _HAS_HF = True
except Exception:
    _HAS_HF = False

# Optional cross-encoder re-ranker
try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANK = True
except Exception:
    _HAS_RERANK = False


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _hf_offline() -> bool:
    # Either var set to "1"/"true" or network really blocked—honor explicit offline flags.
    return str(os.getenv("HF_HUB_OFFLINE", "")).strip() not in ("", "0", "false") or \
           str(os.getenv("TRANSFORMERS_OFFLINE", "")).strip() not in ("", "0", "false")


def _resolve_local_dir(env_var: str, default_subdir: str) -> Optional[str]:
    """
    Return an absolute path to a local model dir if it exists, else None.
    env_var (e.g., EMB_LOCAL_DIR / RERANK_LOCAL_DIR) takes priority.
    default_subdir is relative to this file's directory (e.g., 'models/<name>').
    """
    # 1) explicit env
    p = os.getenv(env_var)
    if p and Path(p).is_dir():
        return str(Path(p).resolve())

    # 2) project-relative
    here = Path(__file__).parent
    candidate = here / default_subdir
    if candidate.is_dir():
        return str(candidate.resolve())
    return None


def _get_embeddings():
    """
    Prefer OpenAI embeddings if available; otherwise use HuggingFace with a local directory if present.
    If offline and no local dir, raise a friendly error.
    """
    if _HAS_OPENAI and _has_openai_key():
        return OpenAIEmbeddings(model="text-embedding-3-small")

    if not _HAS_HF:
        raise RuntimeError(
            "No embeddings backend available. Install `langchain-openai` (and set OPENAI_API_KEY) "
            "or install `langchain_community` + `sentence-transformers`."
        )

    # Try local first
    local_dir = _resolve_local_dir(
        env_var="EMB_LOCAL_DIR",
        default_subdir="models/paraphrase-MiniLM-L6-v2",
    )
    if local_dir:
        return HuggingFaceEmbeddings(model_name=local_dir)

    # No local dir—fall back to hub name only if not offline
    if _hf_offline():
        raise RuntimeError(
            "HF offline mode is enabled and no local embedding model was found.\n"
            "Set EMB_LOCAL_DIR to a downloaded folder, or place the model at "
            "<repo_root>/models/paraphrase-MiniLM-L6-v2.\n"
            "Example download (online machine):\n"
            "  hf download sentence-transformers/paraphrase-MiniLM-L6-v2 "
            "--local-dir models/paraphrase-MiniLM-L6-v2"
        )
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")


def _get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0):
    if _HAS_OPENAI and _has_openai_key():
        return ChatOpenAI(model=model_name, temperature=temperature)
    return None


def build_rag_pipeline(
    docs: str,
    extra_docs: Optional[List[dict]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
):
    """
    Build a vector index with metadata-aware Documents.
    Falls back to DocArrayInMemorySearch when FAISS isn't available.
    - docs: merged plain text from loader
    - extra_docs: list of {"content": str, "metadata": dict} for figures, etc.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = splitter.split_text(docs) if docs else []

    documents: List[Document] = []
    for i, ch in enumerate(text_chunks):
        documents.append(
            Document(page_content=ch, metadata={"type": "text", "chunk_id": i})
        )

    if extra_docs:
        for ed in extra_docs:
            documents.append(
                Document(
                    page_content=(ed.get("content") or "").strip(),
                    metadata={**(ed.get("metadata") or {}), "type": (ed.get("metadata") or {}).get("type", "extra")}
                )
            )

    if not documents:
        raise ValueError("No content to index.")

    embeddings = _get_embeddings()
    return DocArrayInMemorySearch.from_documents(documents, embeddings)



def _domain_prompt(domain: str) -> str:
    base = (
        "You are an AI assistant specialized in {domain}. "
        "Answer strictly using the provided context (including tables/figures). "
        "Provide clear numbers and cite section/table/figure if possible. "
        'If the answer is not in the context, reply exactly: "I don\'t have enough information from the document."'
    )
    return base.format(domain=domain)


def query_rag_full(
    db,
    query: str,
    top_k: int = 12,
    rerank_keep: int = 5,
    domain: str = "Finance",
) -> Tuple[str, List[str], List[Document]]:
    """
    Returns (answer_text, retrieved_texts, retrieved_docs)
    - Retrieves Documents with metadata
    - Optional cross-encoder re-ranking (local-only if offline)
    - LLM synthesis if available, else stitched fallback
    """
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs: List[Document] = retriever.get_relevant_documents(query) or []

    # Optional re-rank
    top_docs = retrieved_docs
    if _HAS_RERANK and retrieved_docs:
        rerank_local = _resolve_local_dir("RERANK_LOCAL_DIR", "models/msmarco-MiniLM-L-6-v2")
        try:
            if rerank_local:
                model = CrossEncoder(rerank_local)
            elif not _hf_offline():
                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            else:
                model = None  # offline + no local reranker → skip
        except Exception:
            model = None

        if model is not None:
            pairs = [[query, d.page_content] for d in retrieved_docs]
            scores = model.predict(pairs)
            idx_sorted = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
            keep = max(1, min(rerank_keep, len(idx_sorted)))
            top_docs = [retrieved_docs[i] for i in idx_sorted[:keep]]

    retrieved_texts = [d.page_content for d in top_docs]
    if not retrieved_texts:
        return "I couldn't find anything relevant in the document.", [], []

    llm = _get_llm()
    sys = _domain_prompt(domain)
    if llm:
        context = "\n\n".join([f"[{i+1}] {d.page_content[:4000]}" for i, d in enumerate(top_docs)])
        cite_hints = []
        for i, d in enumerate(top_docs):
            m = d.metadata or {}
            if m.get("type") == "figure" and m.get("page"):
                cite_hints.append(f"[{i+1}] Figure p.{m.get('page')}")
            elif m.get("type") == "text":
                cite_hints.append(f"[{i+1}] Text chunk {m.get('chunk_id')}")
        hints = "; ".join(cite_hints)

        prompt = f"""{sys}

Context:
{context}

Hints for citations: {hints}

Question: {query}

Answer (include brief citations like [1] or 'Figure p.X' when appropriate):"""
        answer = llm.invoke(prompt).content.strip()
        return answer, retrieved_texts, top_docs

    # Offline fallback
    stitched = " ".join(retrieved_texts)[:1500]
    answer = f"Answer (from retrieved context): {stitched}"
    return answer, retrieved_texts, top_docs


def query_rag(db, query: str, top_k: int = 4) -> Tuple[str, List[str]]:
    ans, texts, _docs = query_rag_full(db, query, top_k=top_k)
    return ans, texts


def evaluate_rag(answer: str, retrieved_docs: List[str], query: str):
    llm = _get_llm(model_name="gpt-4o-mini", temperature=0)
    if not llm:
        return {
            "faithfulness": None,
            "relevancy": None,
            "explanation": (
                "Evaluation requires an LLM (OpenAI). Set OPENAI_API_KEY and install `langchain-openai`."
            ),
            "mode": "offline-fallback",
        }
    docs_text = "\n".join(retrieved_docs)
    eval_prompt = f"""
You are an impartial evaluator. Given a question, an assistant's answer, and the retrieved context,
score the response on:

1) Faithfulness (0-5): Is every claim supported by the retrieved context?
2) Relevancy (0-5): Do the retrieved docs directly address the question?

Return STRICT JSON ONLY:
{{
  "faithfulness": <0-5 integer>,
  "relevancy": <0-5 integer>,
  "explanation": "<one-sentence reason>"
}}

---
Question: {query}

Retrieved Context:
{docs_text}

Answer:
{answer}
"""
    raw = llm.invoke(eval_prompt).content.strip()
    return raw
