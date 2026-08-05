# ─────────────────────────────────────────────
#  src/rag/hybrid_retriever.py
#  Hybrid Retrieval: BM25 (keyword) + ChromaDB (semantic) + MMR reranking
# ─────────────────────────────────────────────
"""
Combines:
  1. BM25Retriever      — keyword/lexical matching (good for exact symptom names)
  2. ChromaDB retriever — dense semantic similarity (good for conceptual matches)
  3. EnsembleRetriever  — weighted fusion of both
  4. MMR reranking      — removes redundant docs, maximises coverage
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_hybrid_retriever(
    chroma_store,           # MedicalVectorStore instance (already built)
    all_docs: List[Document],
    k: int = RETRIEVER_K,
) -> BaseRetriever:
    """
    Build a hybrid BM25 + semantic retriever with MMR reranking.

    Parameters
    ----------
    chroma_store : MedicalVectorStore  — already built vector store
    all_docs     : List[Document]      — original (unsplit) documents for BM25
    k            : int                 — number of final docs to return

    Returns
    -------
    EnsembleRetriever that combines BM25 + ChromaDB with MMR reranking
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    # ── 1. BM25 Retriever (keyword-based) ─────────────────────────────────────
    logger.info(f"Building BM25 retriever over {len(all_docs)} documents…")
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k * 2   # fetch more, then rerank

    # ── 2. Semantic Retriever (MMR for diversity) ──────────────────────────────
    logger.info("Building MMR semantic retriever…")
    mmr_retriever = chroma_store._store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k"               : k * 2,    # fetch more candidates
            "fetch_k"         : k * 4,    # MMR candidate pool
            "lambda_mult"     : 0.6,      # 0=max diversity, 1=max relevance
        },
    )

    # ── 3. Ensemble: 40% BM25 + 60% Semantic ──────────────────────────────────
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, mmr_retriever],
        weights=[0.4, 0.6],
    )

    logger.info(f"Hybrid retriever ready (BM25 40% + MMR 60%, k={k}).")
    return ensemble


class HybridRetrieverWrapper(BaseRetriever):
    """
    Wraps EnsembleRetriever and applies final top-K selection
    with deduplication by page_content hash.
    """
    ensemble: object
    k: int = RETRIEVER_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        raw_docs = self.ensemble.invoke(query)

        # Deduplicate by content
        seen = set()
        unique_docs = []
        for doc in raw_docs:
            key = hash(doc.page_content[:200])
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
            if len(unique_docs) >= self.k:
                break

        logger.info(f"Hybrid retrieval: {len(raw_docs)} raw → {len(unique_docs)} unique docs")
        return unique_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)