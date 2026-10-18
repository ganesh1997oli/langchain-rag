"""Regression checks for empty PDFs and missing indexes, without running Ollama."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

import main
from src import ingest as ingestion
from src import retrieval


class LocalEmbeddings(Embeddings):
    """Keep the real Chroma path testable without an Ollama server."""

    def embed_documents(self, texts):
        """Return one fixed two-number vector per text for deterministic tests."""
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        """Return a query vector with the same dimensions as the test documents."""
        return [1.0, 0.0]


class EmptyInputTests(unittest.TestCase):
    """Exercise validation using real PDF loading and Chroma with fake embeddings."""

    def setUp(self):
        """Give each test temporary folders, local embeddings, and automatic cleanup."""
        temporary = TemporaryDirectory(prefix="rag-validation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pdfs = self.root / "pdfs"
        self.pdfs.mkdir()
        self.index = self.root / "index"
        for module in (ingestion, retrieval):
            self.enterContext(patch.object(module, "VECTOR_STORE_DIR", self.index))
            self.enterContext(patch.object(module, "OllamaEmbeddings", return_value=LocalEmbeddings()))
        self.enterContext(redirect_stdout(StringIO()))

    def test_empty_pdf_directory_fails_before_index_creation(self):
        """An empty folder must produce guidance before creating an index."""
        with self.assertRaisesRegex(ValueError, "No supported documents found"):
            ingestion.ingest(self.pdfs)
        self.assertFalse(self.index.exists())

    def test_pdf_without_text_explains_ocr_requirement(self):
        """A real blank PDF must produce a text-extraction error and leave no index."""
        pdf = self.pdfs / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.write(pdf)
        with self.assertRaisesRegex(ValueError, "No extractable text"):
            ingestion.ingest(pdf)
        self.assertFalse(self.index.exists())

    def test_default_ingestion_uses_configured_pdf_directory(self):
        """Calling ingest without a path must use DATA_DIR."""
        with patch.object(ingestion, "DATA_DIR", self.pdfs, create=True):
            with self.assertRaisesRegex(ValueError, "No supported documents found"):
                ingestion.ingest()

    def test_empty_chunks_never_reach_chroma(self):
        """Direct callers cannot send an empty chunk list to Chroma."""
        with self.assertRaisesRegex(ValueError, "No text chunks"):
            ingestion.build_vector_store([])
        self.assertFalse(self.index.exists())

    def test_missing_index_explains_how_to_ingest(self):
        """A missing index must explain the ingestion command without creating files."""
        with self.assertRaisesRegex(ValueError, "python main.py ingest"):
            retrieval.load_vector_store()
        self.assertFalse(self.index.exists())

    def test_empty_existing_collection_is_rejected(self):
        """An existing Chroma collection with zero records must be rejected."""
        Chroma(collection_name=retrieval.COLLECTION_NAME, persist_directory=str(self.index))
        with self.assertRaisesRegex(ValueError, "python main.py ingest"):
            retrieval.load_vector_store()

    def test_populated_collection_still_loads(self):
        """Validation must allow a saved collection containing an indexed document."""
        Chroma.from_texts(
            ["The support window is 37 days."],
            embedding=LocalEmbeddings(),
            collection_name=retrieval.COLLECTION_NAME,
            persist_directory=str(self.index),
        )
        store = retrieval.load_vector_store()
        self.assertEqual(len(store.get(include=[])["ids"]), 1)

    def test_cli_reports_empty_input_without_traceback(self):
        """The CLI must report empty input with exit status 2 and a readable error."""
        stderr = StringIO()
        with patch("sys.argv", ["main.py", "ingest", "--path", str(self.pdfs)]):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                main.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("No supported documents found", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_chat_does_not_prompt_without_an_index(self):
        """Chat must validate the index before requesting the user's first question."""
        with patch("builtins.input", side_effect=AssertionError("Chat must not start")):
            with self.assertRaisesRegex(ValueError, "python main.py ingest"):
                main.cmd_chat(argparse.Namespace(k=4))


if __name__ == "__main__":
    unittest.main()
