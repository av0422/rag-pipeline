"""
vector_store.py — FAISS vector store: build, persist, and load.

FAISS (Facebook AI Similarity Search) stores dense vectors on disk and
supports sub-linear nearest-neighbor search via IndexFlatL2 (exact L2
distance) by default. For very large corpora, switch to IndexIVFFlat.
"""

import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def build_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_path: str = "faiss_index",
) -> FAISS:
    """Embed all chunks and write the FAISS index to disk."""
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(persist_path)
    return store


def load_vector_store(
    persist_path: str,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    # allow_dangerous_deserialization is required by LangChain when loading
    # pickle-serialised FAISS data that was written by trusted local code.
    return FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def store_exists(persist_path: str) -> bool:
    """Return True if a saved FAISS index directory exists."""
    return os.path.isdir(persist_path)
