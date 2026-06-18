# LangChain RAG with PDF Documents

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://github.com/langchain-ai/langchain)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange.svg)](https://github.com/chroma-core/chroma)
[![Ollama](https://img.shields.io/badge/Ollama-local-purple.svg)](https://ollama.com)

A fully local RAG (Retrieval-Augmented Generation) pipeline that ingests PDF files, stores embeddings in a ChromaDB vector store, and answers questions using a local Ollama LLM — no API keys, no cloud, fully private.

## How it works

```
PDF files → PyPDF loader → text splitter → Ollama embeddings → ChromaDB
                                                                    ↓
question → retriever → context + question → Ollama LLM → answer
```

| Component | Default | Description |
|---|---|---|
| Embeddings | `embeddinggemma:latest` | Local Ollama embedding model |
| Vector store | ChromaDB | Persisted to `vector_store/` |
| LLM | `gemma4:latest` | Local Ollama chat model |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

Pull the required models:

```bash
ollama pull embeddinggemma:latest
ollama pull gemma4:latest
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: override model names or Ollama base URL
cp .env.example .env
```

## Usage

### 1. Ingest PDFs

Place PDF files in `data/pdfs/`, then run:

```bash
python main.py ingest
```

Or point to a specific file or directory:

```bash
python main.py ingest --path /path/to/document.pdf
python main.py ingest --path /path/to/pdf/folder/
```

### 2. Ask a single question

```bash
python main.py query "What is the main topic of the document?"
```

### 3. Interactive chat

```bash
python main.py chat
```

Type `exit` or press `Ctrl+C` to quit.

## Configuration

Settings are in [src/config.py](src/config.py) and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:latest` | Chat model |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |

## Project structure

```
├── data/pdfs/          # Drop your PDF files here
├── src/
│   ├── config.py       # Settings (models, chunking, paths)
│   ├── ingest.py       # PDF loading, splitting, and embedding
│   └── retrieval.py    # Vector store querying and LLM chain
├── vector_store/       # Persisted ChromaDB index (auto-created)
├── main.py             # CLI entry point
└── requirements.txt
```

## Dependencies

- [LangChain](https://github.com/langchain-ai/langchain) — orchestration
- [langchain-ollama](https://github.com/langchain-ai/langchain/tree/master/libs/partners/ollama) — Ollama integration
- [ChromaDB](https://github.com/chroma-core/chroma) — vector store
- [PyPDF](https://github.com/py-pdf/pypdf) — PDF parsing
