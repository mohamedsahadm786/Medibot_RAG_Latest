#!/usr/bin/env python
"""
MediBot v2 — PDF Ingestion CLI

Processes a medical PDF into PostgreSQL (document_chunks table) and
Pinecone (hybrid dense + sparse index).

Usage:
    # Full ingestion
    python run_ingestion.py --pdf data/medical_book.pdf

    # Quick test — first 20 pages only
    python run_ingestion.py --pdf data/medical_book.pdf --pages 20
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Silence noisy third-party loggers
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a medical PDF into MediBot v2 (PostgreSQL + Pinecone)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the medical PDF file (e.g. data/medical_book.pdf)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Limit to first N pages (useful for quick testing). Omit for full ingestion.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        print(f"\n❌  PDF not found: '{pdf_path}'", file=sys.stderr)
        print("    Place your PDF in the data/ directory and re-run.", file=sys.stderr)
        sys.exit(1)

    # Import here so .env is already loaded by the time Settings() is called
    from backend.database import AsyncSessionLocal
    from backend.services.ingestion import IngestionPipeline

    async with AsyncSessionLocal() as session:
        pipeline = IngestionPipeline(
            pdf_path=str(pdf_path),
            db_session=session,
            max_pages=args.pages,
        )
        stats = await pipeline.run()

    if stats.get("skipped"):
        print("Nothing to do — ingestion skipped.")
    else:
        print(f"All done. Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
