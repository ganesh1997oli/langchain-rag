"""Expose source ingestion, single questions, and interactive chat as CLI commands.

Command handlers import their own dependencies so displaying --help stays fast.
"""

import argparse
from pathlib import Path


def cmd_ingest(args):
    """Index local files, one URL, or the rows returned by a database SELECT."""
    from src.ingest import ingest
    if args.database_url and not args.sql:
        raise ValueError("--database-url requires --sql with a SELECT query.")
    if args.sql is not None and not args.database_url:
        raise ValueError("--sql requires --database-url.")
    source = args.database_url or args.url or args.path
    ingest(source, sql=args.sql)


def cmd_query(args):
    """Print one answer; args.k controls how many source chunks are retrieved."""
    from src.retrieval import query
    print(query(args.question, k=args.k))


def cmd_chat(args):
    """Read questions until exit, EOF, or Ctrl+C, reusing one retrieval chain.

    Each question is answered independently; previous turns are not stored.
    Building the chain first checks that an index exists before prompting.
    """
    from src.retrieval import build_rag_chain
    chain = build_rag_chain(k=args.k)
    print("RAG Chat — type 'exit' to quit.\n")
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
    """Parse CLI arguments, call the chosen handler, and report validation errors.

    Supported commands are ingest, query, and chat. argparse places each
    command's options in args, which is passed to the matching handler.
    """
    parser = argparse.ArgumentParser(
        description="LangChain RAG over files, websites, and database content"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load sources and build the vector store")
    sources = p_ingest.add_mutually_exclusive_group()
    sources.add_argument(
        "--path", type=Path, default=None,
        help="Path to a supported file or mixed directory (default: data/)"
    )
    sources.add_argument("--url", help="Public HTTP(S) page or supported document URL")
    sources.add_argument("--database-url", help="SQLite, PostgreSQL, or MySQL connection URL")
    p_ingest.add_argument("--sql", help="SELECT query to read from --database-url")

    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", help="Question to ask")
    p_query.add_argument("--k", type=int, default=4, help="Chunks to retrieve (default: 4)")

    p_chat = sub.add_parser("chat", help="Start an interactive chat session")
    p_chat.add_argument("--k", type=int, default=4, help="Chunks to retrieve (default: 4)")

    args = parser.parse_args()
    try:
        # Select a function by command name, then call it with the parsed options.
        {"ingest": cmd_ingest, "query": cmd_query, "chat": cmd_chat}[args.command](args)
    except ValueError as error:
        # Show an actionable CLI error and exit with status 2.
        parser.error(str(error))


if __name__ == "__main__":
    # Start the CLI when this file is executed directly.
    main()
