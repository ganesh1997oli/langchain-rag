# LangChain RAG with PDF Documents

A fully local RAG (Retrieval-Augmented Generation) pipeline that ingests PDF files, stores them in a ChromaDB vector store, and answers questions using a local Ollama LLM.

## Architecture

```
PDF files → PyPDF loader → text splitter → Ollama embeddings → ChromaDB
                                                                    ↓
question → retriever → context + question → Ollama LLM → answer
```

- **Embeddings**: `embeddinggemma:latest` via Ollama (local)
- **Vector store**: ChromaDB (persisted to `vector_store/`)
- **LLM**: `gemma4:latest` via Ollama (local)

## Prerequisites

Install and start [Ollama](https://ollama.com), then pull the required models:

```bash
ollama pull embeddinggemma:lates
ollama pull gemma4:latest
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: copy .env.example to override default model names or base URL
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

Edit `src/config.py` to adjust:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MODEL_NAME` | `gemma4:latest` | Ollama LLM model |
| `EMBEDDING_MODEL` | `embeddinggemma:latest` | Ollama embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
