"""Shared paths, local model settings, and text-splitting defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read .env without replacing variables already exported in the shell.
load_dotenv()

# Anchor data and index paths to the project, regardless of the working directory.
BASE_DIR = Path(__file__).parent.parent
# Scan all data subfolders, including the existing data/pdfs folder.
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Environment variables select the Ollama server and its two different model roles.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# The chat model writes answers; the embedding model converts text into vectors.
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:latest")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
# These values count characters. Overlap repeats text across neighboring chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# A collection is the named group of indexed chunks inside Chroma.
COLLECTION_NAME = "pdf_documents"
