"""
MediBot v2 — PDF Ingestion Pipeline

Processes a medical PDF into:
  - PostgreSQL: parent + child chunks with full metadata
  - Pinecone: child chunk vectors (dense 768-dim + BM25 sparse)

Memory budget: 8 GB RAM, no GPU.
All embedding work is batched to keep peak usage bounded.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import sqlalchemy as sa
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.models.database import DocumentChunk

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
PINECONE_DIMENSION = 768

# ~1500 tokens × 4 chars/token
PARENT_MAX_CHARS = 6_000
PARENT_OVERLAP_CHARS = 800   # ~200 tokens

# Encode in mini-batches to respect 8 GB RAM ceiling
EMBED_BATCH_SIZE = 32

# Pinecone upsert limit
PINECONE_BATCH = 100

# PostgreSQL insert batch
PG_BATCH = 500


# ── LangChain-compatible wrapper ───────────────────────────────────────────────

class _STEmbeddings(Embeddings):
    """
    Thin LangChain Embeddings wrapper around a loaded SentenceTransformer,
    so we can pass one model instance to SemanticChunker instead of loading
    a second copy.
    """

    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents, batched to manage RAM."""
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            vecs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_vecs.extend(vecs.tolist())
        return all_vecs

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


# ── Main pipeline class ────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full 7-step ingestion pipeline.

    Usage:
        async with AsyncSessionLocal() as session:
            pipeline = IngestionPipeline(pdf_path="data/medical_book.pdf", db_session=session)
            stats = await pipeline.run()
    """

    def __init__(
        self,
        pdf_path: str,
        db_session: AsyncSession,
        max_pages: int | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.source_pdf = self.pdf_path.name
        self.db = db_session
        self.max_pages = max_pages   # None = process all pages

        # Loaded once and reused across steps
        self._model: SentenceTransformer | None = None
        self._bm25: BM25Encoder | None = None

    # ── Lazy model loader ──────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Load PubMedBERT once and cache it for the lifetime of this pipeline."""
        if self._model is None:
            logger.info("Loading SentenceTransformer: %s (CPU)", EMBEDDING_MODEL)
            self._model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return self._model

    # ── Step 1: PDF extraction ─────────────────────────────────────────────────

    def extract_pages(self) -> list[dict[str, Any]]:
        """
        Extract text from the PDF page by page.

        Section heading detection:
          - Scans every span in every text block.
          - A span is treated as a heading if it is bold (flags & 16) OR its
            font size is >20 % larger than the page median, AND the text is
            between 2 and 15 words (typical chapter/section title length).
          - The first such span on a page is recorded as that page's
            section_heading.
        """
        doc = fitz.open(str(self.pdf_path))
        total_pages = len(doc)
        limit = min(self.max_pages, total_pages) if self.max_pages else total_pages
        pages: list[dict] = []

        for page_idx in range(limit):
            page = doc[page_idx]
            raw = page.get_text("dict")
            blocks = raw.get("blocks", [])

            # Collect all font sizes on this page to find the median
            font_sizes: list[float] = []
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = span.get("size", 0.0)
                        if sz > 0:
                            font_sizes.append(sz)

            median_size: float = (
                sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 12.0
            )

            page_text_parts: list[str] = []
            section_heading: str | None = None

            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    span_texts: list[str] = []
                    line_is_heading = False

                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        flags = span.get("flags", 0)
                        size = span.get("size", 12.0)

                        is_bold = bool(flags & 16)
                        is_larger = size > median_size * 1.2
                        word_count = len(text.split())

                        if (is_bold or is_larger) and 2 <= word_count <= 15:
                            line_is_heading = True

                        span_texts.append(text)

                    if span_texts:
                        line_str = " ".join(span_texts)
                        page_text_parts.append(line_str)
                        if line_is_heading and section_heading is None:
                            section_heading = line_str[:200]

            pages.append(
                {
                    "page_number": page_idx + 1,
                    "text": " ".join(page_text_parts).strip(),
                    "section_heading": section_heading,
                }
            )

        doc.close()
        logger.info("Extracted %d / %d pages", len(pages), total_pages)
        return pages

    # ── Step 2: Parent chunks ──────────────────────────────────────────────────

    def create_parent_chunks(self, pages: list[dict]) -> list[dict[str, Any]]:
        """
        Group consecutive pages into parent chunks.

        Strategy:
          1. Accumulate pages into the current section until a new heading
             appears OR the accumulated text exceeds PARENT_MAX_CHARS.
          2. If the accumulated text still exceeds PARENT_MAX_CHARS after
             grouping, split it with RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_MAX_CHARS,
            chunk_overlap=PARENT_OVERLAP_CHARS,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        parents: list[dict] = []

        def _flush(heading: str | None, text: str, page_num: int) -> None:
            text = text.strip()
            if not text:
                return
            if len(text) > PARENT_MAX_CHARS:
                for sub in splitter.split_text(text):
                    sub = sub.strip()
                    if sub:
                        parents.append(
                            {
                                "id": str(uuid.uuid4()),
                                "content": sub,
                                "section_heading": heading or "",
                                "page_number": page_num,
                            }
                        )
            else:
                parents.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": text,
                        "section_heading": heading or "",
                        "page_number": page_num,
                    }
                )

        current_heading: str | None = None
        current_text: str = ""
        current_page: int = 1

        for page in pages:
            heading = page.get("section_heading")
            text = page.get("text", "")
            page_num = page["page_number"]

            new_section = heading and heading != current_heading
            overflow = len(current_text) + len(text) > PARENT_MAX_CHARS * 1.5

            if new_section or overflow:
                _flush(current_heading, current_text, current_page)
                current_heading = heading if new_section else current_heading
                current_text = text
                current_page = page_num
            else:
                current_text += " " + text

        _flush(current_heading, current_text, current_page)

        logger.info("Created %d parent chunks", len(parents))
        return parents

    # ── Step 3: Child chunks ───────────────────────────────────────────────────

    def create_child_chunks(self, parents: list[dict]) -> list[dict[str, Any]]:
        """
        Apply SemanticChunker to every parent to get fine-grained children.

        The same SentenceTransformer instance used for dense embeddings is
        reused here via the _STEmbeddings wrapper — no second model load.

        Each child stores:
          id, parent_id, content, section_heading, page_number
        """
        model = self._get_model()
        lc_embeddings = _STEmbeddings(model)

        chunker = SemanticChunker(
            embeddings=lc_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )

        children: list[dict] = []
        total = len(parents)

        for idx, parent in enumerate(parents):
            if (idx + 1) % 20 == 0 or idx == 0:
                logger.info("  Semantic chunking parent %d / %d", idx + 1, total)

            try:
                child_texts = chunker.split_text(parent["content"])
            except Exception as exc:
                logger.warning(
                    "SemanticChunker failed on parent %s (%s) — using parent as single child",
                    parent["id"],
                    exc,
                )
                child_texts = [parent["content"]]

            for child_text in child_texts:
                child_text = child_text.strip()
                if len(child_text) < 80:   # skip fragments
                    continue
                children.append(
                    {
                        "id": str(uuid.uuid4()),
                        "parent_id": parent["id"],
                        "content": child_text,
                        "section_heading": parent["section_heading"],
                        "page_number": parent["page_number"],
                    }
                )

        logger.info("Created %d child chunks from %d parents", len(children), total)
        return children

    # ── Step 4: PostgreSQL storage ─────────────────────────────────────────────

    async def store_chunks(
        self, parents: list[dict], children: list[dict]
    ) -> None:
        """
        Insert all parent rows first (needed for FK from children), then children.
        Uses PG_BATCH-sized transactions to avoid locking the table for too long.
        """
        logger.info(
            "Storing %d parents + %d children in PostgreSQL...",
            len(parents),
            len(children),
        )

        def _make_row(chunk: dict, chunk_type: str) -> DocumentChunk:
            return DocumentChunk(
                id=uuid.UUID(chunk["id"]),
                chunk_type=chunk_type,
                parent_id=uuid.UUID(chunk["parent_id"]) if chunk.get("parent_id") else None,
                content=chunk["content"],
                page_number=chunk["page_number"],
                section_heading=chunk["section_heading"],
                source_pdf=self.source_pdf,
            )

        all_rows = [_make_row(p, "parent") for p in parents] + [
            _make_row(c, "child") for c in children
        ]

        for i in range(0, len(all_rows), PG_BATCH):
            self.db.add_all(all_rows[i : i + PG_BATCH])
            await self.db.commit()
            logger.debug("  Committed rows %d–%d", i, min(i + PG_BATCH, len(all_rows)))

        logger.info("PostgreSQL: %d total rows inserted", len(all_rows))

    # ── Step 5+6: Embeddings ───────────────────────────────────────────────────

    def fit_bm25(self, texts: list[str], save_path: str = "data/bm25_encoder.json") -> BM25Encoder:
        """
        Fit BM25 vocabulary + IDF on all child texts and save to disk.
        The retriever loads this file at query time.
        """
        logger.info("Fitting BM25 on %d documents...", len(texts))
        self._bm25 = BM25Encoder()
        self._bm25.fit(texts)
        self._bm25.dump(save_path)
        logger.info("BM25 encoder saved to %s", save_path)
        return self._bm25

    # ── Step 7: Pinecone upsert ────────────────────────────────────────────────

    def _get_or_create_pinecone_index(self):
        """Return the Pinecone Index object, creating the index if needed."""
        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing_names = [idx.name for idx in pc.list_indexes()]

        if settings.pinecone_index_name not in existing_names:
            logger.info(
                "Creating Pinecone index '%s' (dim=%d, metric=dotproduct)...",
                settings.pinecone_index_name,
                PINECONE_DIMENSION,
            )
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=PINECONE_DIMENSION,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for the index to be ready
            for _ in range(30):
                status = pc.describe_index(settings.pinecone_index_name).status
                if status.get("ready"):
                    break
                logger.info("  Waiting for index to become ready...")
                time.sleep(3)

        return pc.Index(settings.pinecone_index_name)

    def upsert_to_pinecone(
        self, children: list[dict], index, bm25: BM25Encoder
    ) -> None:
        """
        Upsert child vectors in PINECONE_BATCH-sized batches.

        For each batch:
          1. Encode dense (SentenceTransformer, inner batch = EMBED_BATCH_SIZE)
          2. Encode sparse (BM25)
          3. Upsert to Pinecone
        """
        model = self._get_model()
        total = len(children)

        for start in range(0, total, PINECONE_BATCH):
            batch = children[start : start + PINECONE_BATCH]
            texts = [c["content"] for c in batch]
            end = min(start + PINECONE_BATCH, total)
            logger.info("  Upserting vectors %d–%d / %d...", start + 1, end, total)

            # Dense: encode in EMBED_BATCH_SIZE mini-batches inside ST
            dense_vecs = model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Sparse
            sparse_vecs = bm25.encode_documents(texts)

            vectors = [
                {
                    "id": child["id"],
                    "values": dense.tolist(),
                    "sparse_values": sparse,
                    "metadata": {
                        "child_id": child["id"],
                        "parent_id": child["parent_id"],
                        "page_number": child["page_number"],
                        "section_heading": child["section_heading"] or "",
                        "source_pdf": self.source_pdf,
                    },
                }
                for child, dense, sparse in zip(batch, dense_vecs, sparse_vecs)
            ]

            index.upsert(vectors=vectors, namespace=settings.pinecone_namespace)

        logger.info(
            "Pinecone: upserted %d vectors to namespace '%s'",
            total,
            settings.pinecone_namespace,
        )

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Execute all 7 steps and return summary statistics."""

        sep = "-" * 60
        print(f"\n{sep}")
        print("MediBot v2 - Ingestion Pipeline")
        print(f"PDF : {self.pdf_path}")
        if self.max_pages:
            print(f"Mode: first {self.max_pages} pages (test run)")
        print(f"{sep}\n")

        # Guard: skip if this PDF was already ingested
        existing = await self.db.scalar(
            sa.select(sa.func.count()).where(
                DocumentChunk.source_pdf == self.source_pdf
            )
        )
        if existing and existing > 0:
            print(
                f"WARNING: Found {existing} existing chunks for '{self.source_pdf}'. "
                "Delete them first or use a different namespace.\n"
            )
            return {"skipped": True, "existing_chunks": existing}

        t0 = time.perf_counter()

        print("Step 1/7 - Extracting text from PDF...")
        pages = self.extract_pages()
        print(f"  [OK] {len(pages)} pages\n")

        print("Step 2/7 - Creating parent chunks...")
        parents = self.create_parent_chunks(pages)
        print(f"  [OK] {len(parents)} parents\n")

        print("Step 3/7 - Creating child chunks (SemanticChunker + PubMedBERT)...")
        print("  Note: loads ~400 MB model; first run may be slow on CPU.")
        children = self.create_child_chunks(parents)
        print(f"  [OK] {len(children)} children\n")

        print("Step 4/7 - Storing all chunks in PostgreSQL...")
        await self.store_chunks(parents, children)
        print(f"  [OK] {len(parents) + len(children)} rows inserted\n")

        child_texts = [c["content"] for c in children]

        print("Step 5/7 - Dense embedding model ready (already loaded in step 3)...")
        print(f"  [OK] Using {EMBEDDING_MODEL}\n")

        print("Step 6/7 - Fitting BM25 sparse encoder...")
        bm25 = self.fit_bm25(child_texts)
        print("  [OK] BM25 fitted\n")

        print("Step 7/7 - Upserting to Pinecone (dense + sparse)...")
        index = self._get_or_create_pinecone_index()
        self.upsert_to_pinecone(children, index, bm25)
        print("  [OK] Done\n")

        elapsed = time.perf_counter() - t0
        stats = {
            "source_pdf": self.source_pdf,
            "pages_processed": len(pages),
            "parents": len(parents),
            "children": len(children),
            "elapsed_seconds": round(elapsed, 1),
        }

        print(sep)
        print("Ingestion complete!")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"{sep}\n")

        return stats
