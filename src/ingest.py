"""Load supported sources, split their text, and save embeddings in Chroma."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.sources import load_documents

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VECTOR_STORE_DIR,
)


def split_documents(documents: list) -> list:
    """Split source text into smaller documents while retaining source metadata.

    The recursive splitter tries natural text boundaries, such as paragraphs
    and spaces, to fit CHUNK_SIZE characters. CHUNK_OVERLAP repeats boundary
    text so related information can appear in neighboring chunks.

    Returns:
        A list of chunk documents suitable for embedding and retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list) -> Chroma:
    """Embed chunks through Ollama and add them to the persisted Chroma collection.

    Embeddings are numerical representations used to compare text meaning.
    Chroma stores those vectors alongside each chunk's text and metadata in
    VECTOR_STORE_DIR. Existing collection contents are retained.

    Returns:
        The Chroma store containing the indexed chunks.

    Raises:
        ValueError: No chunks were supplied.
    """
    if not chunks:
        raise ValueError("No text chunks to index. Ingest a source containing readable text first.")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
    )


def ingest(source: str | Path | None = None, *, sql: str | None = None) -> Chroma:
    """Run the source-to-index workflow and print progress for the CLI.

    Args:
        source: File, directory, web URL, or database URL. None selects DATA_DIR.
        sql: SELECT statement when source is a database URL.

    Returns:
        The persisted Chroma store, ready for questions.

    Raises:
        ValueError: The input is invalid, has no documents, or yields no text.
    """
    source = DATA_DIR if source is None else source
    print("Loading documents...")
    docs = load_documents(source, sql=sql)
    if not docs:
        label = "database query" if sql is not None else str(source)
        raise ValueError(
            f"No supported documents found in {label}. "
            "Add readable files to data/ or choose a source with --path, --url, or --database-url."
        )
    print(f"Loaded {len(docs)} documents.")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError(
            "No extractable text found in the source. "
            "Use files containing readable text; scanned PDFs may need OCR."
        )
    print(f"Created {len(chunks)} chunks.")

    print("Building vector store...")
    store = build_vector_store(chunks)
    print(f"Vector store saved to {VECTOR_STORE_DIR}")

    return store
