import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:latest")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "pdf_documents"
