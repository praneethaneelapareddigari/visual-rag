# dataset_eval.py
import os, json, glob, time
from datetime import datetime
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_recall

from doc_loader import load_document
from rag_pipeline import build_rag_pipeline, query_rag_full

DATA_DIR = "datasets/finance"
OUTPUT = "eval_runs"
os.makedirs(OUTPUT, exist_ok=True)

def load_questions(q_path):
    with open(q_path, "r") as f:
        data = json.load(f)
    return data["questions"], data.get("ground_truth", [""] * len(data["questions"]))

def eval_file(pdf_path, q_path, domain="Finance"):
    docs, _ = load_document(pdf_path, return_sections=True)
    db = build_rag_pipeline(docs)
    questions, golds = load_questions(q_path)

    answers, contexts, latencies = [], [], []
    for q in questions:
        t0 = time.time()
        ans, ctxs, _docs = query_rag_full(db, q, domain=domain)
        latencies.append(round(time.time() - t0, 3))
        answers.append(ans)
        contexts.append([c for c in ctxs])  # ragas expects List[List[str]]

    ds = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": golds,
    })
    scores = evaluate(ds, metrics=[faithfulness, answer_relevance, context_recall])
    try:
        recs = scores.to_pandas().to_dict(orient="records")
    except Exception:
        recs = [scores] if isinstance(scores, dict) else [{"scores_raw": str(scores)}]

    for i, r in enumerate(recs):
        r["latency_s"] = latencies[i] if i < len(latencies) else None

    return questions, answers, golds, recs

def main():
    rows = []
    for pdf_path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        q_path = os.path.join(DATA_DIR, f"{base}.questions.json")
        if not os.path.exists(q_path):
            print(f"[WARN] Missing questions for {pdf_path} (expected {q_path}) — skipping.")
            continue
        print(f"[INFO] Evaluating {base} ...")
        qs, ans, gold, recs = eval_file(pdf_path, q_path)
        for i, r in enumerate(recs):
            r.update({
                "file": base,
                "question": qs[i],
                "answer": ans[i],
                "ground_truth": gold[i],
            })
            rows.append(r)

    if not rows:
        print("[INFO] No evaluations produced.")
        return

    df = pd.DataFrame.from_records(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUTPUT, f"aggregate_{stamp}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved aggregate CSV → {out_csv}")

if __name__ == "__main__":
    main()
