"""Retrieve relevant source chunks and ask a local Ollama model to answer from them."""

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MODEL_NAME,
    OLLAMA_BASE_URL,
    VECTOR_STORE_DIR,
)

# The chain fills these placeholders with retrieved text and the current question.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an assistant that answers questions based on the provided context from documents and other sources. "
            "Use only the information in the context to answer. "
            "If the answer is not in the context, say so clearly.\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{question}"),
])


def _format_docs(docs: list) -> str:
    """Join retrieved chunk texts into the prompt's context, separated by blank lines.

    Only page_content is included; filename and page metadata stay in the index.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def load_vector_store() -> Chroma:
    """Open the saved index and check that it contains searchable documents.

    The embedding model converts new questions into vectors for comparison
    with saved chunk vectors. It must match the model used during ingestion.

    Returns:
        A Chroma store connected to the local persistent collection.

    Raises:
        ValueError: The index is missing or its collection contains no chunks.
    """
    empty_index_message = (
        "No indexed source text is available. Add supported files to data/ and run "
        "'python main.py ingest' before asking questions."
    )
    if not VECTOR_STORE_DIR.exists():
        raise ValueError(empty_index_message)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    # Request one record's ID to check for content without fetching text or vectors.
    if not store.get(limit=1, include=[])["ids"]:
        raise ValueError(empty_index_message)
    return store


def build_rag_chain(k: int = 4):
    """Build an executable retrieval-augmented generation (RAG) pipeline.

    Args:
        k: Maximum number of similar chunks to retrieve per question.

    Returns:
        A LangChain runnable. Calling .invoke(question) retrieves context,
        formats the prompt, calls Ollama, and returns the answer as a string.
    """
    store = load_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    # LangChain's | operator passes each step's output to the next step.
    return (
        # Send the same input to retrieval and to the unchanged question field.
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        # Convert the model's message object into plain answer text.
        | StrOutputParser()
    )


def query(question: str, k: int = 4) -> str:
    """Build a chain, answer one question using up to k chunks, and return its text.

    Interactive chat builds its chain once and reuses it across questions.
    """
    return build_rag_chain(k=k).invoke(question)
