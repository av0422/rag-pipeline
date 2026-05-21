"""
pipeline.py — RAG query pipeline using LangChain Expression Language (LCEL).

Flow per query:
  1. Retrieve top-k chunks from FAISS (with similarity scores for tracking).
  2. Format chunks into a context string.
  3. Run context + question through a ChatPromptTemplate → Ollama → StrOutputParser.
  4. Log retrieval/generation latency and similarity scores to MLflow.
"""

from typing import List, Tuple

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from tracking import Timer, log_metrics


PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use only the context below to answer the question.
If the answer is not contained in the context, respond with:
"I don't have enough information in the document to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


def _build_chain(model: str = "llama3.2"):
    """Return a compiled LCEL chain: prompt → LLM → string output."""
    llm = ChatOllama(model=model, temperature=0)
    return PROMPT | llm | StrOutputParser()


def run_query(
    vector_store: FAISS,
    question: str,
    k: int = 4,
    model: str = "llama3.2",
) -> Tuple[str, List, float]:
    """
    Execute one RAG query and return (answer, source_docs, total_latency).

    Similarity scores from FAISS are L2 distances — lower is more similar.
    We log their mean as a proxy for retrieval quality across experiments.
    """
    # --- Retrieval ---
    with Timer() as retrieval_timer:
        docs_with_scores = vector_store.similarity_search_with_score(question, k=k)

    source_docs = [doc for doc, _ in docs_with_scores]
    scores = [float(score) for _, score in docs_with_scores]
    context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

    # --- Generation ---
    chain = _build_chain(model)
    with Timer() as generation_timer:
        answer = chain.invoke({"context": context, "question": question})

    total_latency = retrieval_timer.elapsed + generation_timer.elapsed

    log_metrics(
        {
            "retrieval_latency_s": retrieval_timer.elapsed,
            "generation_latency_s": generation_timer.elapsed,
            "total_latency_s": total_latency,
            "num_retrieved_chunks": len(source_docs),
            # Mean L2 distance: lower → chunks are closer to the query vector
            "mean_similarity_score": sum(scores) / len(scores) if scores else 0.0,
        }
    )

    return answer, source_docs, total_latency
