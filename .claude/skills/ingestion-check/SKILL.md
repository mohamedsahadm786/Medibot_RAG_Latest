---
name: ingestion-check
description: Validate the ingestion pipeline output. Checks parent-child chunk integrity, Pinecone vector counts vs Postgres chunk counts, embedding dimensions, and BM25 encoder freshness. Use after running `run_ingestion.py` on new PDFs or after any change to `ingestion.py`.
user-invocable: true
---

# Ingestion Check Skill

Post-ingestion validation. Catches orphan chunks, dimension mismatches, stale BM25 encoders — issues that present downstream as retrieval quality problems.

## When to invoke

- After `python run_ingestion.py --pdf data/<new_pdf>`
- After changing `SemanticChunker` settings, parent chunk size, or overlap in `backend/services/ingestion.py`
- Before a production deploy if ingestion was part of the change
- When retrieval quality drops suddenly (run this first — ingestion issues often look like retrieval bugs)

## Checks performed

1. **Chunk count parity:**
   - Pinecone vector count vs `SELECT COUNT(*) FROM document_chunks WHERE chunk_type='child'`
   - Flag if mismatch > 1%
2. **Parent-child integrity:**
   - Every child has a valid `parent_id` pointing to an existing parent
   - Every parent has ≥ 1 child
   - Flag orphan children and childless parents
3. **Embedding dimensions:**
   - Pinecone index must be `dimension=768` (PubMedBERT)
   - Random sample 20 vectors, verify dimension
4. **BM25 encoder freshness:**
   - `data/bm25_encoder.json` mtime vs newest `document_chunks.created_at`
   - Encoder older than newest chunk = stale (hybrid search will use outdated sparse vectors)
5. **Pinecone metadata completeness:**
   - Random sample 50 vectors
   - Every one must have: `child_id`, `parent_id`, `page_number`, `section_heading`, `source_pdf`

## Commands

```bash
# Run all checks
python scripts/ingestion_check.py

# Verbose mode
python scripts/ingestion_check.py --verbose
```

## Output

`ingestion_reports/{timestamp}.json` with pass/fail per check and counts.

## Note

Script to be created under `scripts/ingestion_check.py`. ~100 lines of Python: queries Postgres for counts, queries Pinecone via `pinecone-client` for stats, compares. Use this skill as the spec.