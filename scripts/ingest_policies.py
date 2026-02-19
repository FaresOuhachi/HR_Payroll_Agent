"""
Policy Document Ingestion Script
=============================================================================
CONCEPT: Knowledge Base Bootstrapping

This script ingests all HR policy documents from the sample_data/policies/
directory into the vector store. It's the "loading dock" of the RAG system —
you run it once to populate the knowledge base, and again whenever policies
are updated.

WHAT IT DOES:
  1. Scans sample_data/policies/ for Markdown files
  2. For each file, runs the ingestion pipeline:
     a. Read the file content
     b. Split into overlapping chunks (RecursiveCharacterTextSplitter)
     c. Generate embeddings for each chunk (OpenAI text-embedding-3-small)
     d. Store chunks + embeddings in PostgreSQL (pgvector)
  3. Reports ingestion statistics

WHEN TO RUN:
  - First time setup: after running migrations and seed_data.py
  - Policy updates: whenever a policy document is modified
  - Re-ingestion is safe: existing chunks from the same source are deleted
    before new ones are inserted (upsert-like behavior)

PREREQUISITES:
  - PostgreSQL running with pgvector extension enabled
  - Database migrations applied (documents table exists)
  - OPENAI_API_KEY set in .env file
  - Policy Markdown files in sample_data/policies/

Run: python -m scripts.ingest_policies
=============================================================================
"""

import asyncio
import glob
import os
import sys
import time

# Add project root to path so we can import src modules
# CONCEPT: Python Path Management
# When running a script from the scripts/ directory, Python doesn't
# automatically know about our src/ package. We add the project root
# to sys.path so imports like "from src.rag.ingestion import ..." work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.ingestion import ingest_markdown_file
from src.rag.vectorstore import count_documents
from src.db.engine import engine


async def main():
    """
    Ingest all policy documents from sample_data/policies/.

    CONCEPT: Batch Ingestion with Progress Reporting
    For operational visibility, we report:
      - Which files are being processed
      - How many chunks each file produces
      - Total ingestion time and statistics
    This helps diagnose issues (empty files, failed embeddings) and gives
    a sense of knowledge base size.
    """
    print("=" * 70)
    print("HR Payroll Agent — Policy Document Ingestion")
    print("=" * 70)

    # Locate the policies directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policies_dir = os.path.join(project_root, "sample_data", "policies")

    if not os.path.exists(policies_dir):
        print(f"\nERROR: Policies directory not found: {policies_dir}")
        print("Make sure sample_data/policies/ exists with Markdown files.")
        sys.exit(1)

    # Find all Markdown files
    # CONCEPT: Glob Pattern Matching
    # glob.glob("*.md") finds all files ending in .md in a directory.
    # This is more reliable than manually listing files — any new policy
    # document added to the directory will be automatically discovered.
    md_files = sorted(glob.glob(os.path.join(policies_dir, "*.md")))

    if not md_files:
        print(f"\nNo Markdown files found in: {policies_dir}")
        print("Add .md policy files to sample_data/policies/ and try again.")
        sys.exit(1)

    print(f"\nFound {len(md_files)} policy document(s):")
    for f in md_files:
        size = os.path.getsize(f)
        print(f"  - {os.path.basename(f)} ({size:,} bytes)")

    print(f"\n{'─' * 70}")

    # Check current document count before ingestion
    try:
        before_count = await count_documents()
        print(f"Documents in vector store before ingestion: {before_count}")
    except Exception as e:
        print(f"Warning: Could not count existing documents: {e}")
        before_count = 0

    print(f"{'─' * 70}\n")

    # Ingest each file
    total_chunks = 0
    total_chars = 0
    results = []
    overall_start = time.time()

    for i, file_path in enumerate(md_files, 1):
        filename = os.path.basename(file_path)
        print(f"[{i}/{len(md_files)}] Ingesting: {filename}")
        print(f"  {'.' * 50}", end=" ", flush=True)

        start_time = time.time()

        try:
            result = await ingest_markdown_file(file_path)
            elapsed = time.time() - start_time

            total_chunks += result["total_chunks"]
            total_chars += result["total_characters"]
            results.append(result)

            print(f"OK ({elapsed:.1f}s)")
            print(f"  Chunks: {result['total_chunks']}, "
                  f"Characters: {result['total_characters']:,}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"  Error: {e}")
            # Continue with next file instead of stopping entirely
            # CONCEPT: Graceful Error Handling
            # In batch operations, one file's failure shouldn't prevent
            # other files from being processed. We log the error and continue.
            continue

        print()

    overall_elapsed = time.time() - overall_start

    # Print summary
    print(f"{'=' * 70}")
    print("INGESTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Files processed:    {len(results)}/{len(md_files)}")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  Total characters:   {total_chars:,}")
    print(f"  Total time:         {overall_elapsed:.1f}s")
    if total_chunks > 0:
        print(f"  Avg chunk size:     {total_chars // total_chunks} chars")
        print(f"  Avg time/file:      {overall_elapsed / len(results):.1f}s")

    # Verify final count
    try:
        after_count = await count_documents()
        print(f"\n  Documents in store: {after_count}")
    except Exception:
        pass

    print(f"\n{'=' * 70}")

    if len(results) == len(md_files):
        print("All policy documents ingested successfully!")
    else:
        failed = len(md_files) - len(results)
        print(f"WARNING: {failed} file(s) failed to ingest. Check logs above.")

    print("\nThe knowledge base is ready for RAG queries.")
    print("Try: POST /documents/search with body: {\"query\": \"What is the sick leave policy?\"}")
    print(f"{'=' * 70}")

    # Clean up database connections
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
