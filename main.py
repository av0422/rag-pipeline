"""
main.py — CLI entry point for the RAG pipeline.

Two sub-commands:
  ingest  — load a PDF, chunk it, embed it, and save the FAISS index.
  query   — start an interactive Q&A session against a saved index.

Usage:
  python main.py ingest path/to/doc.pdf [--chunk-size N] [--chunk-overlap N]
  python main.py query [--k N] [--model MODEL]
"""

import argparse
import os
import sys

import mlflow

from document_loader import load_pdf, chunk_documents
from embeddings import get_embedding_model
from vector_store import build_vector_store, load_vector_store, store_exists
from pipeline import run_query
from tracking import start_experiment, log_params, log_metrics


FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 4
DEFAULT_MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Load a PDF, chunk it, embed it, and persist the FAISS index."""
    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        print(f"Error: file not found — {pdf_path}")
        sys.exit(1)

    log_params(
        {
            "pdf": os.path.basename(pdf_path),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "embedding_model": EMBEDDING_MODEL,
        }
    )

    print(f"\nLoading '{pdf_path}' ...")
    pages = load_pdf(pdf_path)
    print(f"  {len(pages)} pages loaded.")

    print(f"Chunking  (size={args.chunk_size}, overlap={args.chunk_overlap}) ...")
    chunks = chunk_documents(pages, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"  {len(chunks)} chunks created.")
    log_metrics({"num_chunks": len(chunks)})

    print(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    embeddings = get_embedding_model(EMBEDDING_MODEL)

    print("Building FAISS index ...")
    build_vector_store(chunks, embeddings, persist_path=FAISS_INDEX_PATH)
    print(f"  Index saved to '{FAISS_INDEX_PATH}/'.\n")
    print("Ingestion complete. Run 'python main.py query' to ask questions.")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def cmd_query(args: argparse.Namespace) -> None:
    """Load a saved FAISS index and start an interactive Q&A loop."""
    if not store_exists(FAISS_INDEX_PATH):
        print(f"Error: no index found at '{FAISS_INDEX_PATH}/'. Run 'ingest' first.")
        sys.exit(1)

    log_params(
        {
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_k": args.k,
            "llm_model": args.model,
        }
    )

    print(f"\nLoading embedding model '{EMBEDDING_MODEL}' ...")
    embeddings = get_embedding_model(EMBEDDING_MODEL)

    print("Loading FAISS index ...")
    vector_store = load_vector_store(FAISS_INDEX_PATH, embeddings)

    print(f"\nReady. Using top-{args.k} chunks and model '{args.model}'.")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    query_count = 0

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        print("Thinking ...\n")
        answer, sources, latency = run_query(
            vector_store, question, k=args.k, model=args.model
        )
        query_count += 1

        print(f"Answer: {answer}")
        print(f"\n[{len(sources)} chunks retrieved in {latency:.2f}s]")

        show = input("Show source chunks? [y/N]: ").strip().lower()
        if show == "y":
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                print(f"\n--- Chunk {i}  (page {page}) ---")
                # Truncate long chunks for readability in the terminal
                print(doc.page_content[:500].strip())
        print()

    log_metrics({"total_queries": query_count})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG pipeline: index a PDF and ask questions about it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- ingest --
    p_ingest = sub.add_parser("ingest", help="Load and index a PDF document.")
    p_ingest.add_argument("pdf", help="Path to the PDF file.")
    p_ingest.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Token/character target per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    p_ingest.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between adjacent chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )

    # -- query --
    p_query = sub.add_parser("query", help="Ask questions about the indexed document.")
    p_query.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of chunks to retrieve per query (default: {DEFAULT_K}).",
    )
    p_query.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use for generation (default: {DEFAULT_MODEL}).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_experiment("rag_pipeline")

    with mlflow.start_run():
        if args.command == "ingest":
            cmd_ingest(args)
        elif args.command == "query":
            cmd_query(args)


if __name__ == "__main__":
    main()
