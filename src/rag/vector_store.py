# ─────────────────────────────────────────────
#  src/rag/vector_store.py — ChromaDB Vector Store
# ─────────────────────────────────────────────
"""
Builds and persists a ChromaDB vector store from MedQA documents
using HuggingFace sentence-transformers embeddings (free, local).
Exposes a retriever for similarity search.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    VECTORSTORE_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
    RETRIEVER_SCORE_THRESHOLD,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Text Splitter ──────────────────────────────────────────────────────────────

def _split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better retrieval granularity."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} docs → {len(chunks)} chunks.")
    return chunks


# ── Embeddings ─────────────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model (runs locally, free)."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Vector Store ───────────────────────────────────────────────────────────────

class MedicalVectorStore:
    """Wrapper around ChromaDB for medical document retrieval."""

    def __init__(self) -> None:
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._store: Optional[Chroma] = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = _get_embeddings()
        return self._embeddings

    def _is_store_populated(self) -> bool:
        """Check if the persisted ChromaDB collection already has documents."""
        try:
            client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
            collection = client.get_or_create_collection(VECTORSTORE_COLLECTION)
            count = collection.count()
            logger.info(f"Existing vector store has {count} vectors.")
            return count > 0
        except Exception as e:
            logger.warning(f"Could not check vector store: {e}")
            return False

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self, docs: list[Document], force_rebuild: bool = False) -> "MedicalVectorStore":
        """
        Build the vector store from documents.
        Skips rebuilding if the store is already populated (unless force_rebuild=True).
        """
        if not force_rebuild and self._is_store_populated():
            logger.info("Vector store already exists — loading from disk.")
            self._store = Chroma(
                collection_name=VECTORSTORE_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=VECTORSTORE_PATH,
            )
            return self

        logger.info("Building vector store from documents…")
        chunks = _split_documents(docs)

        # Batch ingestion to avoid memory spikes (ChromaDB limit: 5461/batch)
        batch_size = 2000
        if len(chunks) > batch_size:
            logger.info(f"Ingesting in batches of {batch_size}…")
            self._store = Chroma.from_documents(
                documents=chunks[:batch_size],
                embedding=self.embeddings,
                collection_name=VECTORSTORE_COLLECTION,
                persist_directory=VECTORSTORE_PATH,
            )
            for i in range(batch_size, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                self._store.add_documents(batch)
                logger.info(f"  Added batch ending at {i + len(batch)}")
        else:
            self._store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=VECTORSTORE_COLLECTION,
                persist_directory=VECTORSTORE_PATH,
            )

        logger.info("Vector store built and persisted.")
        return self

    def load(self) -> "MedicalVectorStore":
        """Load an existing persisted vector store."""
        logger.info("Loading vector store from disk…")
        self._store = Chroma(
            collection_name=VECTORSTORE_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=VECTORSTORE_PATH,
        )
        return self

    def get_retriever(self) -> BaseRetriever:
        """Return a LangChain retriever with similarity score threshold."""
        if self._store is None:
            raise RuntimeError("Vector store not initialized. Call build() or load() first.")

        retriever = self._store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": RETRIEVER_K,
                "score_threshold": RETRIEVER_SCORE_THRESHOLD,
            },
        )
        logger.info(f"Retriever ready (k={RETRIEVER_K}, threshold={RETRIEVER_SCORE_THRESHOLD}).")
        return retriever

    def similarity_search(self, query: str, k: int = RETRIEVER_K) -> list[Document]:
        """Direct similarity search for inspection / debugging."""
        if self._store is None:
            raise RuntimeError("Vector store not initialized.")
        return self._store.similarity_search(query, k=k)
