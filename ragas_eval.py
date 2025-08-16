# ragas_eval.py
# Convenience alias to run a single sample quickly.
import os, json, time
from datetime import datetime
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_recall

from doc_loader import load_document
from rag_pipeline import build_rag_pipeline, query_rag_full

DOC_PATH = "samples/finance_report.pdf"
OUTPUT_DIR = "eval_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    docs, sections = load_document(DOC_PATH, return_sections=True)
    db = build_rag_pipeline(docs)

    questions = [
        "What was the company’s net profit in 2022?",
        "What is the EPS reported for Q3 2023?",
        "Summarize the auditor’s opinion in one sentence.",
    ]
    gold = ["", "", ""]  # fill if known

    answers, contexts, latencies = [], [], []
    for q in questions:
        t0 = time.time()
        ans, ctxs, _ = query_rag_full(db, q, domain="Finance")
        lat = round(time.time() - t0, 3)
        answers.append(ans)
        contexts.append(ctxs)
        latencies.append(lat)

    ds = Dataset.from_dict({
        "question": questions,
        "contexts": [list(c) for c in contexts],
        "answer": answers,
        "ground_truth": gold,
    })
    scores = evaluate(ds, metrics=[faithfulness, answer_relevance, context_recall])

    # persist
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(OUTPUT_DIR, f"ragas_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump({
            "doc_path": DOC_PATH,
            "questions": questions,
            "answers": answers,
            "latencies": latencies,
            "scores": getattr(scores, "to_dict", lambda: str(scores))(),
        }, f, indent=2)

    print(f"[OK] Saved → {out_json}")

if __name__ == "__main__":
    run()
