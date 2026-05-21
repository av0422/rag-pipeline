"""
embeddings.py — Sentence-transformer embedding model wrapper.

Uses all-MiniLM-L6-v2: 384-dim, fast, good quality for semantic search.
normalize_embeddings=True makes cosine similarity equivalent to dot product,
which is what FAISS IndexFlatL2 expects after normalization.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Load the sentence-transformer model as a LangChain embeddings object."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
