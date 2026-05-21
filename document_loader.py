"""
document_loader.py — PDF loading and text chunking.

RecursiveCharacterTextSplitter tries progressively smaller separators
so chunks stay semantically coherent (paragraph → sentence → word).
"""

from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(pdf_path: str) -> List[Document]:
    """Load a PDF file and return one Document per page."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    Overlap preserves context across chunk boundaries so the retriever
    doesn't miss answers that straddle two chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Preference order: paragraph → newline → sentence → word → char
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
