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

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an assistant that answers questions based on the provided context from PDF documents. "
            "Use only the information in the context to answer. "
            "If the answer is not in the context, say so clearly.\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{question}"),
])


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def load_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )


def build_rag_chain(k: int = 4):
    store = load_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def query(question: str, k: int = 4) -> str:
    return build_rag_chain(k=k).invoke(question)
