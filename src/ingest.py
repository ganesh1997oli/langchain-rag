from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    VECTOR_STORE_DIR,
)


def load_documents(pdf_path: str | Path) -> list:
    print(pdf_path);
    path = Path(pdf_path)
    loader = PyPDFLoader(str(path)) if path.is_file() else PyPDFDirectoryLoader(str(path))
   
    return loader.load()


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list) -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
    )


def ingest(pdf_path: str | Path | None = None) -> Chroma:
    print("Loading documents...")
    docs = load_documents(pdf_path)
    print(f"Loaded {len(docs)} pages.")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Building vector store...")
    store = build_vector_store(chunks)
    print(f"Vector store saved to {VECTOR_STORE_DIR}")

    return store
