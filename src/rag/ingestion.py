"""
Document Ingestion Pipeline
=============================================================================
CONCEPT: Why Do We Need an Ingestion Pipeline?

RAG (Retrieval-Augmented Generation) requires a knowledge base of embedded
document chunks. The ingestion pipeline converts raw documents (Markdown files,
PDFs, web pages) into searchable vector embeddings. It's a multi-step process:

  Raw Document → Split into Chunks → Generate Embeddings → Store in Vector DB

CONCEPT: Why Chunk Documents?

LLMs have a limited context window (e.g., GPT-4 can handle ~128K tokens, but
shorter contexts produce better results and cost less). If we stored an entire
50-page policy document as one chunk:
  1. The embedding would be too "diluted" — it tries to capture the meaning of
     50 pages in 1536 numbers, losing fine-grained details.
  2. We'd waste context space — when the user asks about "sick leave", we'd
     retrieve the entire document instead of just the relevant paragraph.
  3. Poor relevance ranking — a chunk about "sick leave" and another about
     "tax brackets" in the same document would get the same embedding.

By splitting into smaller chunks (e.g., 800 characters each), each chunk:
  - Has a focused embedding that captures its specific content
  - Can be retrieved independently (only the relevant paragraph)
  - Uses minimal context window space, leaving room for more relevant chunks

CONCEPT: Chunk Overlap

When we split text at arbitrary boundaries, we risk splitting a sentence or
idea in the middle:

  Chunk 1: "...employees receive 10 paid sick days per year."
  Chunk 2: "Sick days do not carry over to the next year..."

Without overlap, if we search for "do sick days carry over?", the relevant
context is split across two chunks, and neither chunk alone contains the
full answer.

With overlap (200 characters), the end of Chunk 1 and the beginning of
Chunk 2 share common text, creating a "bridge":

  Chunk 1: "...employees receive 10 paid sick days per year. Sick days do
            not carry over to the next year..."  (includes start of next section)
  Chunk 2: "Sick days do not carry over to the next year. Unused sick days
            are not compensated..."  (repeats end of previous section)

Now both chunks contain "Sick days do not carry over", so a search for
"do sick days carry over?" will find at least one chunk with full context.

The trade-off: overlap creates duplicate content (more storage, more embeddings
to compute), but significantly improves retrieval quality at chunk boundaries.

CONCEPT: Chunk Size Selection
  - Too small (100 chars): Chunks lack context ("10 paid sick days" alone
    doesn't mention WHO gets them or WHEN)
  - Too large (5000 chars): Embeddings become diluted, retrieval less precise
  - Sweet spot (500-1500 chars): Each chunk contains a complete thought/paragraph
  - We use 800 chars as a good default for policy documents

CONCEPT: RecursiveCharacterTextSplitter (from LangChain)
  This splitter tries to keep semantically related text together by splitting
  at natural boundaries, in this priority order:
    1. Double newlines ("\n\n") — paragraph breaks
    2. Single newlines ("\n") — line breaks
    3. Spaces (" ") — word boundaries
    4. Empty string ("") — character level (last resort)

  It "recurses" through these separators: if splitting by paragraphs produces
  chunks that are still too large, it tries splitting by lines, then by words.
  This produces much better chunks than splitting every N characters blindly.
=============================================================================
"""

import logging
import os
import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings import embedding_service
from src.rag.vectorstore import store_document, delete_documents_by_source
from src.db.engine import async_session_maker

logger = logging.getLogger(__name__)


# =============================================================================
# Text Splitter Configuration
# =============================================================================
# CONCEPT: Splitter as a Shared Instance
# We configure the splitter once and reuse it. The parameters are tuned for
# HR policy documents, which typically have well-structured sections with
# headers, lists, and paragraphs.
#
# chunk_size=800: Target ~800 characters per chunk. Policy sections are
#   typically 200-1000 chars, so this captures 1-2 paragraphs.
#
# chunk_overlap=200: 200 characters of overlap between consecutive chunks.
#   This is ~25% of chunk size, which ensures good context continuity
#   without excessive duplication.
#
# separators: Priority list for finding split points. We prefer to split
#   at Markdown headers first (##, ###) to keep sections intact, then
#   at paragraph boundaries, then at line breaks, then at word boundaries.
# =============================================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,  # Measure chunk size by character count
    separators=[
        "\n## ",      # Markdown H2 headers (major sections)
        "\n### ",     # Markdown H3 headers (subsections)
        "\n\n",       # Paragraph breaks
        "\n",         # Line breaks
        ". ",         # Sentence boundaries
        " ",          # Word boundaries
        "",           # Character level (last resort)
    ],
)


def _extract_section_header(chunk_text: str, full_text: str) -> str:
    """
    Extract the most relevant section header for a chunk of text.

    CONCEPT: Section Metadata
    When we retrieve a chunk, it's helpful to know which section it came from.
    For example, knowing that a chunk is from "Section 2: Sick Leave" gives
    both the user and the LLM important context about what they're reading.

    We look backwards from the chunk's position in the original document to
    find the nearest Markdown header (## or ###). This becomes the chunk's
    section metadata.

    Args:
        chunk_text: The text of the chunk
        full_text:  The complete original document text

    Returns:
        The section header string, or "General" if no header is found.
    """
    # Find where this chunk starts in the full document
    chunk_start = full_text.find(chunk_text[:100])  # Use first 100 chars for matching
    if chunk_start == -1:
        return "General"

    # Look at the text before this chunk for the nearest header
    text_before = full_text[:chunk_start]

    # Find all Markdown headers in the preceding text
    # Matches lines starting with ## or ### (H2 and H3)
    headers = re.findall(r'^#{2,3}\s+(.+)$', text_before, re.MULTILINE)

    if headers:
        return headers[-1].strip()  # Return the most recent (closest) header

    # Also check if the chunk itself starts with a header
    header_match = re.match(r'^#{2,3}\s+(.+)$', chunk_text.strip(), re.MULTILINE)
    if header_match:
        return header_match.group(1).strip()

    return "General"


async def ingest_markdown_file(
    file_path: str,
    source_name: Optional[str] = None,
) -> dict:
    """
    Read a Markdown file, split it into chunks, embed each chunk, and store
    in the vector database.

    CONCEPT: End-to-End Ingestion Pipeline
    This function orchestrates the full ingestion workflow:
      1. READ: Load the raw Markdown content from disk
      2. SPLIT: Break it into overlapping chunks using RecursiveCharacterTextSplitter
      3. EMBED: Convert each chunk to a 1536-dim vector using OpenAI
      4. STORE: Insert each (chunk, embedding) pair into PostgreSQL with metadata

    The source_name is used to identify which file a chunk came from. This
    enables:
      - Re-ingestion: delete old chunks by source, then re-ingest
      - Filtered search: "search only in the leave policy"
      - Citation: "this answer comes from compensation_policy.md, Section 2"

    Args:
        file_path:   Path to the Markdown file on disk
        source_name: Human-readable name for this source. If not provided,
                     defaults to the filename (e.g., "leave_policy.md")

    Returns:
        A dict with ingestion statistics:
          - source: The source name used
          - file_path: The original file path
          - total_chunks: Number of chunks created
          - total_characters: Total character count across all chunks
          - chunk_ids: List of UUIDs for the stored chunks

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If embedding or storage fails
    """
    # Step 1: READ the document
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    source = source_name or os.path.basename(file_path)

    logger.info(f"Starting ingestion of: {file_path} (source: {source})")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        logger.warning(f"File is empty: {file_path}")
        return {
            "source": source,
            "file_path": file_path,
            "total_chunks": 0,
            "total_characters": 0,
            "chunk_ids": [],
        }

    logger.info(f"Read {len(content)} characters from {file_path}")

    # Step 2: SPLIT into chunks
    # CONCEPT: The splitter returns a list of strings, each representing one chunk.
    # Chunks may slightly exceed chunk_size if splitting at the preferred separator
    # would produce a chunk that's too small. The splitter balances chunk size
    # against semantic coherence.
    chunks = text_splitter.split_text(content)

    logger.info(
        f"Split into {len(chunks)} chunks "
        f"(avg {sum(len(c) for c in chunks) // max(len(chunks), 1)} chars/chunk)"
    )

    # Step 3: EMBED all chunks in a batch
    # CONCEPT: Batch vs Individual Embedding
    # Embedding all chunks in one batch call is dramatically faster than
    # embedding them one-by-one. For 20 chunks:
    #   - Individual: 20 API calls x ~200ms = ~4 seconds
    #   - Batch: 1 API call x ~400ms = ~0.4 seconds (10x faster!)
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = await embedding_service.embed_batch(chunks)

    # Step 4: STORE each chunk with its embedding
    # CONCEPT: Transactional Batch Insert
    # We use a single database session for all inserts. This means either
    # ALL chunks are stored (commit) or NONE are (rollback on error).
    # This prevents partial ingestion, which would leave the knowledge base
    # in an inconsistent state.
    logger.info(f"Storing {len(chunks)} chunks in vector store...")

    # First, delete any existing chunks from this source (re-ingestion support)
    await delete_documents_by_source(source)

    chunk_ids = []
    async with async_session_maker() as session:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Extract the section header for this chunk
            section = _extract_section_header(chunk, content)

            # Build metadata for this chunk
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
                "file_path": file_path,
            }

            doc_id = await store_document(
                content=chunk,
                embedding=embedding,
                source=source,
                section=section,
                metadata=metadata,
                session=session,
            )
            chunk_ids.append(doc_id)

        # Commit all inserts in one transaction
        await session.commit()

    total_chars = sum(len(c) for c in chunks)
    logger.info(
        f"Ingestion complete: {source} → {len(chunks)} chunks, "
        f"{total_chars} chars, {len(chunk_ids)} stored"
    )

    return {
        "source": source,
        "file_path": file_path,
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "chunk_ids": chunk_ids,
    }


async def ingest_text(
    content: str,
    source: str,
    section: str = "",
) -> dict:
    """
    Ingest a raw text string (not from a file) into the vector store.

    CONCEPT: API-Based Ingestion
    Not all documents come from files. This function supports ingesting text
    that comes from:
      - API requests (POST /documents/ingest)
      - Database fields
      - Scraped web content
      - User-provided knowledge

    It follows the same split → embed → store pipeline as file ingestion.

    Args:
        content: The raw text to ingest
        source:  Source identifier for these chunks
        section: Optional section label

    Returns:
        Dict with ingestion statistics (same format as ingest_markdown_file)
    """
    if not content.strip():
        return {
            "source": source,
            "total_chunks": 0,
            "total_characters": 0,
            "chunk_ids": [],
        }

    logger.info(f"Ingesting text content ({len(content)} chars) as source: {source}")

    # Split → Embed → Store (same pipeline as file ingestion)
    chunks = text_splitter.split_text(content)
    logger.info(f"Split into {len(chunks)} chunks")

    embeddings = await embedding_service.embed_batch(chunks)

    # Delete old chunks from this source before inserting new ones
    await delete_documents_by_source(source)

    chunk_ids = []
    async with async_session_maker() as session:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_section = section or _extract_section_header(chunk, content)
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
                "ingestion_type": "text",
            }

            doc_id = await store_document(
                content=chunk,
                embedding=embedding,
                source=source,
                section=chunk_section,
                metadata=metadata,
                session=session,
            )
            chunk_ids.append(doc_id)

        await session.commit()

    return {
        "source": source,
        "total_chunks": len(chunks),
        "total_characters": sum(len(c) for c in chunks),
        "chunk_ids": chunk_ids,
    }
