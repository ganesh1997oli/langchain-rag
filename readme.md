# Local RAG with Files, Websites, and Databases

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://github.com/langchain-ai/langchain)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange.svg)](https://github.com/chroma-core/chroma)
[![Ollama](https://img.shields.io/badge/Ollama-local-purple.svg)](https://ollama.com)

A RAG (Retrieval-Augmented Generation) application that reads documents, websites, and database query results, stores their text and embeddings in Chroma, and answers questions using a local Ollama model. Embeddings, the index, and answer generation run locally. URL ingestion fetches the selected website; database ingestion connects to the database you specify.

## How it works

```
Files / URL / SELECT query → source reader → text splitter → Ollama embeddings → ChromaDB
                                                                                  ↓
question → retriever → context + question → Ollama LLM → answer
```

| Component | Default | Description |
|---|---|---|
| Embeddings | `embeddinggemma:latest` | Local Ollama embedding model |
| Vector store | ChromaDB | Persisted to `vector_store/` |
| LLM | `gemma4:latest` | Local Ollama chat model |

## Supported sources

| Source | Supported input |
|---|---|
| PDF | `.pdf` containing extractable text |
| Word | `.docx`, including paragraphs and tables |
| Text and code | `.txt`, `.md`, `.rst`, `.log`, `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, and other common text-based code/config formats |
| Spreadsheets | `.csv`, `.tsv`, `.xlsx` (all worksheets) |
| Structured records | `.json` and `.jsonl` |
| HTML and websites | Local `.html`/`.htm`, a public HTTP(S) page, or a supported downloadable document |
| Email exports | `.eml` and `.mbox`, including headers and message bodies |
| Databases | SQLite, PostgreSQL, and MySQL through an explicit `SELECT` query |

Folder ingestion reads supported files recursively, including files inside `data/pdfs/`. Hidden files/folders, dependency folders, and unsupported formats are skipped. An explicitly selected unsupported file produces an error. New file formats can be added in `src/sources.py`.

CSV, TSV, and Excel inputs use the first nonempty row as column headers. Every data row retains its column labels; Excel also retains its worksheet name. Excel formulas use cached results saved by a spreadsheet application; formulas are not recalculated here. Text and code files should be UTF-8.

Web ingestion reads one page or document per command, with a 20 MiB download limit. It does not execute JavaScript or crawl linked pages. Private cloud accounts, Gmail/Outlook synchronization, automatic OCR, images/audio/video, and legacy `.doc`/`.xls` files require additional readers or integrations. Email attachments are excluded. Export cloud content to a supported format, or use an accessible public page.

## Prerequisites

- Python 3.10+ (this Mac is configured with Python 3.12)
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

An existing local chat model can be selected with `OLLAMA_MODEL` in `.env`. This Mac is configured to use `llama3:latest` for answers and `embeddinggemma:latest` for embeddings. The PostgreSQL and MySQL drivers are included in `requirements.txt`.

## Usage

### 1. Ingest local files

Place supported files anywhere under `data/`, including the existing `data/pdfs/` folder, then run:

```bash
python main.py ingest
```

Or point to a specific file or mixed directory:

```bash
python main.py ingest --path /path/to/document.pdf
python main.py ingest --path /path/to/manual.docx
python main.py ingest --path /path/to/spreadsheet.xlsx
python main.py ingest --path /path/to/code-or-documents/
```

### 2. Ingest a website or document URL

```bash
python main.py ingest --url "https://example.com/"
```

Use the URL of the specific page or supported document you want to search.

### 3. Ingest database content

Choose the rows and columns to index with a `SELECT` query. SQLite requires an existing database file:

```bash
python main.py ingest \
  --database-url "sqlite:////absolute/path/to/catalog.db" \
  --sql "SELECT name, description FROM products LIMIT 100"
```

PostgreSQL and MySQL accept SQLAlchemy connection URLs. For example, set a connection URL in your shell and select only the content you want to index:

```bash
export RAG_DATABASE_URL="postgresql+psycopg://reader:password@localhost/catalog"
python main.py ingest \
  --database-url "$RAG_DATABASE_URL" \
  --sql "SELECT title, body FROM articles LIMIT 100"
```

For MySQL, use a URL such as `mysql+pymysql://reader:password@localhost/catalog`. Database ingestion enables a read-only transaction; credentials are excluded from indexed source metadata. Query results are indexed as a snapshot. Later database changes require ingestion again.

Each ingestion command accepts one of `--path`, `--url`, or `--database-url`. Run separate commands to add different source types to the same index. Existing records are retained, and repeated ingestion can add duplicates; this application does not yet synchronize source edits or deletions.

### 4. Ask a single question

```bash
python main.py query "What is the main topic of the document?"
```

### 5. Interactive chat

```bash
python main.py chat
```

Type `exit` or press `Ctrl+C` to quit.

Ingestion must succeed before querying or chatting. An empty source folder, a PDF without extractable text, or an empty index produces a readable error. Chat answers each question independently; it does not retain conversation history.

## Configuration

Settings are in [src/config.py](src/config.py) and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:latest` | Chat model |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma:latest` | Embedding model |

`CHUNK_SIZE = 1000` and `CHUNK_OVERLAP = 200` are constants in `src/config.py`; change them there. Keep the embedding model consistent with the saved index. Changing embedding models requires rebuilding the index.

## Project structure

```
├── data/               # Local source files (excluded from Git)
│   └── pdfs/           # Existing PDF folder continues to work
├── src/
│   ├── config.py       # Settings (models, chunking, paths)
│   ├── sources.py      # File, URL, email, and database readers
│   ├── ingest.py       # Shared loading, splitting, and embedding workflow
│   └── retrieval.py    # Vector store querying and LLM chain
├── vector_store/       # Persisted ChromaDB index (auto-created)
├── main.py             # CLI entry point
└── requirements.txt
```

## Verification

Run the source reader, CLI, and empty-input regression tests without an Ollama server:

```bash
python -m unittest discover -s tests -v
```

## Dependencies

- [LangChain](https://github.com/langchain-ai/langchain) — orchestration
- [langchain-ollama](https://github.com/langchain-ai/langchain/tree/master/libs/partners/ollama) — Ollama integration
- [ChromaDB](https://github.com/chroma-core/chroma) — vector store
- [PyPDF](https://github.com/py-pdf/pypdf) — PDF parsing
- python-docx and openpyxl — Word and Excel parsing
- Beautiful Soup and requests — HTML and URL reading
- SQLAlchemy, psycopg, and PyMySQL — database ingestion
