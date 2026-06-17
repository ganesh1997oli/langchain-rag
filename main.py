import argparse
from pathlib import Path


def cmd_ingest(args):
    from src.ingest import ingest
    ingest(args.path)


def cmd_query(args):
    from src.retrieval import query
    print(query(args.question, k=args.k))


def cmd_chat(args):
    from src.retrieval import build_rag_chain
    print("RAG Chat — type 'exit' to quit.\n")
    chain = build_rag_chain(k=args.k)
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        print(f"\nAssistant: {chain.invoke(question)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LangChain RAG over PDF documents"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load PDFs and build the vector store")
    p_ingest.add_argument(
        "--path", type=Path, default='data/pdfs',
        help="Path to a PDF file or directory (default: data/pdfs/)"
    )

    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", help="Question to ask")
    p_query.add_argument("--k", type=int, default=4, help="Chunks to retrieve (default: 4)")

    p_chat = sub.add_parser("chat", help="Start an interactive chat session")
    p_chat.add_argument("--k", type=int, default=4, help="Chunks to retrieve (default: 4)")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "query": cmd_query, "chat": cmd_chat}[args.command](args)


if __name__ == "__main__":
    main()
