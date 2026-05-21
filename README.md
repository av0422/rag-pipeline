# RAG Pipeline

A local Retrieval-Augmented Generation (RAG) system that lets you ask natural-language questions about any PDF document.

---

## What it does

```
PDF → chunk → embed → FAISS index
                               ↓
            question → retrieve top-k chunks → LLM → answer
```

1. **Ingest** — load a PDF, split it into overlapping text chunks, generate embeddings with a sentence-transformer model, and persist a FAISS vector index to disk.
2. **Query** — embed a question, retrieve the most relevant chunks from FAISS, pass them as context to the LLM, and return a grounded answer.
3. **Track** — every run is logged to MLflow: chunk parameters, retrieval latency, generation latency, similarity scores, and more.

---

## Technologies

| Component | Library / Model |
|---|---|
| PDF loading | `langchain-community` → `PyPDFLoader` (via `pypdf`) |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector store | `FAISS` (Facebook AI Similarity Search, CPU build) |
| LLM orchestration | LangChain Expression Language (LCEL) |
| LLM | Ollama — `llama3.2` (local, no API key needed) |
| Experiment tracking | MLflow |
| CLI | Python `argparse` |

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Download Ollama from [ollama.com](https://ollama.com), then pull the default model:

```bash
ollama pull llama3.2
```

Ollama runs locally — no API key or account required.

---

## Usage

### Ingest a PDF

```bash
python main.py ingest path/to/document.pdf
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--chunk-size` | `500` | Target characters per chunk |
| `--chunk-overlap` | `50` | Overlap between consecutive chunks |

Example with custom chunking:
```bash
python main.py ingest report.pdf --chunk-size 800 --chunk-overlap 100
```

The FAISS index is saved to `faiss_index/` in the current directory.

---

### Ask questions

```bash
python main.py query
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--k` | `4` | Number of chunks to retrieve per question |
| `--model` | `llama3.2` | Ollama model for answer generation |

Example:
```bash
python main.py query --k 6 --model mistral
```

Type your question at the prompt. Press `Enter`. Optionally view the source chunks used to generate the answer. Type `quit` to exit.

---

## Viewing MLflow experiments

Every ingest and query session is tracked as an MLflow run under the `rag_pipeline` experiment.

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser. You can compare runs with different `chunk_size`, `k`, or model settings.

### Logged parameters

| Parameter | Description |
|---|---|
| `chunk_size` | Target chunk size used during ingestion |
| `chunk_overlap` | Chunk overlap used during ingestion |
| `embedding_model` | Sentence-transformer model name |
| `retrieval_k` | Number of retrieved chunks per query |
| `llm_model` | Ollama model used for generation |

### Logged metrics

| Metric | Description |
|---|---|
| `num_chunks` | Total chunks created during ingestion |
| `retrieval_latency_s` | Seconds spent on FAISS lookup |
| `generation_latency_s` | Seconds spent on LLM generation |
| `total_latency_s` | End-to-end query latency |
| `num_retrieved_chunks` | Chunks returned per query (= k) |
| `mean_similarity_score` | Mean L2 distance of retrieved chunks (lower = more similar) |
| `total_queries` | Total questions answered in a session |

---

## Project structure

```
rag/
├── main.py              # CLI entry point (ingest / query sub-commands)
├── document_loader.py   # PDF loading and recursive text chunking
├── embeddings.py        # HuggingFace sentence-transformer wrapper
├── vector_store.py      # FAISS index: build, save, load
├── pipeline.py          # LCEL chain + query execution + MLflow metrics
├── tracking.py          # MLflow helpers and Timer utility
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Tuning tips

- **Chunk size** — smaller chunks (200–400) improve precision; larger chunks (600–1000) preserve more context per retrieval.
- **Chunk overlap** — 10–20 % of chunk size is a good starting point to avoid cutting sentences mid-thought.
- **k** — increase if answers seem incomplete; decrease to reduce noise and latency.
- Use MLflow to compare `mean_similarity_score` across chunk configurations — lower scores on your real queries indicate better retrieval alignment.
