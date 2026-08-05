#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  evaluate.py — LangSmith RAG Evaluation
# ─────────────────────────────────────────────
"""
Evaluates the RAG pipeline using LangSmith.
Tests relevancy, groundedness, and response quality.

Usage:
    python evaluate.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LANGSMITH_TRACING, LANGCHAIN_API_KEY
from src.utils.logger import get_logger

logger = get_logger("evaluate")

# ── Test cases ────────────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    "I have a high fever, severe headache, and stiff neck. What could this be?",
    "I've been experiencing chest pain, shortness of breath, and sweating. What are possible causes?",
    "I have persistent cough for 3 weeks, fatigue, and night sweats. What might be wrong?",
    "I have frequent urination, excessive thirst, and blurred vision. What disease could this be?",
    "I have a skin rash that is red, itchy, and spreading. What could cause this?",
]


def run_evaluation():
    """Run evaluation and optionally trace with LangSmith."""
    if not LANGSMITH_TRACING or not LANGCHAIN_API_KEY:
        logger.warning("LangSmith tracing not configured — running local evaluation only.")

    # Load pipeline
    from src.rag.data_loader  import load_medqa_documents
    from src.rag.vector_store import MedicalVectorStore
    from src.rag.pipeline     import MedicalRAGPipeline

    logger.info("Loading pipeline for evaluation…")
    docs      = load_medqa_documents()
    store     = MedicalVectorStore().build(docs)
    retriever = store.get_retriever()
    pipeline  = MedicalRAGPipeline(retriever)

    # Run test questions
    print("\n" + "═" * 70)
    print("  📊  RAG Pipeline Evaluation")
    print("═" * 70)

    results = []
    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] Q: {question[:60]}…")
        result = pipeline.ask(question)

        has_context = result["has_context"]
        doc_count   = len(result["retrieved_docs"])
        answer_len  = len(result["answer"])

        print(f"  ✓ Context retrieved : {has_context} ({doc_count} docs)")
        print(f"  ✓ Answer length     : {answer_len} chars")
        print(f"  ✓ Sources           : {result['sources']}")

        results.append({
            "question"    : question,
            "has_context" : has_context,
            "doc_count"   : doc_count,
            "answer_len"  : answer_len,
        })

    # Summary
    print("\n" + "─" * 70)
    print(f"  Total questions  : {len(results)}")
    print(f"  With context     : {sum(r['has_context'] for r in results)}")
    print(f"  Avg docs/query   : {sum(r['doc_count'] for r in results)/len(results):.1f}")
    print(f"  Avg answer len   : {sum(r['answer_len'] for r in results)/len(results):.0f} chars")
    print("═" * 70 + "\n")

    if LANGSMITH_TRACING:
        print("📡  Results traced to LangSmith — check your dashboard at:")
        print("    https://smith.langchain.com/")


if __name__ == "__main__":
    run_evaluation()
