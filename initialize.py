#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  initialize.py — One-time Setup Script
# ─────────────────────────────────────────────
"""
Run this ONCE before starting the Streamlit app.
It will:
  1. Download MedQA dataset (5 000 records) from HuggingFace
  2. Split into chunks
  3. Embed with sentence-transformers (local, free)
  4. Persist ChromaDB vector store to disk

Usage:
    python initialize.py
    python initialize.py --force   # force rebuild even if store exists
"""
import argparse
import sys
import time

from config import GOOGLE_API_KEY, VECTORSTORE_PATH
from src.rag.data_loader import load_medqa_documents
from src.rag.vector_store import MedicalVectorStore
from src.utils.logger import get_logger

logger = get_logger("initialize")


def main(force_rebuild: bool = False) -> None:
    print("\n" + "═" * 60)
    print("  🏥  Medical Chatbot — Initialization")
    print("═" * 60 + "\n")

    # ── Validate API Key ───────────────────────────────────────────────────────
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        print("❌  GOOGLE_API_KEY is not set!")
        print("    1. Copy .env.example → .env")
        print("    2. Add your Gemini API key (free at https://aistudio.google.com/app/apikey)")
        sys.exit(1)
    print("✅  Gemini API key found.\n")

    # ── Load Dataset ───────────────────────────────────────────────────────────
    print("📥  Step 1/2 — Loading MedQA dataset (up to 5 000 records)…")
    t0 = time.time()
    docs = load_medqa_documents()
    print(f"    Loaded {len(docs)} documents in {time.time()-t0:.1f}s\n")

    # ── Build Vector Store ─────────────────────────────────────────────────────
    print("🔢  Step 2/2 — Building ChromaDB vector store…")
    print(f"    (stored at: {VECTORSTORE_PATH})")
    t0 = time.time()
    store = MedicalVectorStore()
    store.build(docs, force_rebuild=force_rebuild)
    print(f"    Vector store ready in {time.time()-t0:.1f}s\n")

    print("═" * 60)
    print("  ✅  Initialization complete!")
    print("  🚀  Run the app with:  streamlit run app.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Medical Chatbot")
    parser.add_argument("--force", action="store_true", help="Force rebuild vector store")
    args = parser.parse_args()
    main(force_rebuild=args.force)
