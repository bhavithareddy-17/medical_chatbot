# ─────────────────────────────────────────────
#  src/rag/data_loader.py — MedQA Dataset Loader
# ─────────────────────────────────────────────
"""
Loads the MedQA dataset from HuggingFace (bigbio/med_qa),
formats records into rich medical text documents, and caches
them locally to avoid repeated downloads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain.schema import Document

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DATASET_NAME, DATASET_SPLIT, MAX_RECORDS, DATASET_CACHE_FILE
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_record(record: dict, idx: int) -> Document:
    """Convert a single MedQA record to a LangChain Document."""
    question  = record.get("question", "").strip()
    answer_key = record.get("answer_idx", "")
    options   = record.get("options", {})

    # Build the correct answer text
    answer_text = ""
    if isinstance(options, dict):
        answer_text = options.get(str(answer_key), "")
    elif isinstance(options, list):
        # Some splits store options as a list of dicts {"key": ..., "value": ...}
        for opt in options:
            if isinstance(opt, dict) and opt.get("key") == answer_key:
                answer_text = opt.get("value", "")
                break

    # Build all options text for richer context
    all_options = ""
    if isinstance(options, dict):
        all_options = " | ".join(f"{k}: {v}" for k, v in options.items())
    elif isinstance(options, list):
        all_options = " | ".join(
            f"{o.get('key','')}: {o.get('value','')}"
            for o in options if isinstance(o, dict)
        )

    page_content = (
        f"Medical Question: {question}\n"
        f"Correct Answer: {answer_text}\n"
        f"All Options: {all_options}"
    )

    metadata = {
        "source"    : "MedQA",
        "record_id" : idx,
        "answer_key": str(answer_key),
    }
    return Document(page_content=page_content, metadata=metadata)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_medqa_documents() -> List[Document]:
    """
    Load MedQA documents — from local cache if available,
    otherwise stream from HuggingFace and cache to disk.

    Returns
    -------
    List[Document]
        Up to MAX_RECORDS LangChain Documents.
    """
    cache_path = Path(DATASET_CACHE_FILE)

    # ── Try cache first ────────────────────────────────────────────────────────
    if cache_path.exists():
        logger.info(f"Loading MedQA from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        docs = [
            Document(page_content=r["page_content"], metadata=r["metadata"])
            for r in raw
        ]
        logger.info(f"Loaded {len(docs)} documents from cache.")
        return docs

    # ── Download from HuggingFace ──────────────────────────────────────────────
    logger.info(f"Downloading {DATASET_NAME} ({DATASET_SPLIT} split)…")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            DATASET_NAME,
            name="med_qa_en_bigbio_qa",
            split=DATASET_SPLIT,
            streaming=False,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    docs: List[Document] = []
    for idx, record in enumerate(dataset):
        if idx >= MAX_RECORDS:
            break
        try:
            docs.append(_format_record(record, idx))
        except Exception as ex:
            logger.warning(f"Skipping record {idx}: {ex}")

    # ── Persist to cache ───────────────────────────────────────────────────────
    logger.info(f"Caching {len(docs)} documents to {cache_path}")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"MedQA dataset loaded: {len(docs)} records.")
    return docs
