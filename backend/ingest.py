"""
AutoBrain Lite — PDF Ingestion Pipeline

Converts a vehicle owner manual (PDF) into a searchable FAISS vector index.

Pipeline:
    PDF → Extract text (per page) → Chunk text → Embed chunks → Save FAISS index + metadata

Usage:
    python backend/ingest.py data/manual.pdf
    python backend/ingest.py data/manual.pdf --chunk-size 400 --overlap 40
"""

import json
import os
import sys
from pathlib import Path

import click
import fitz  # pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Step 1: Extract text from PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF file.
    
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    Each page's text is cleaned of excessive whitespace.
    """
    doc = fitz.open(pdf_path)
    pages = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction

        # Clean up: collapse multiple newlines/spaces
        text = " ".join(text.split())

        if text.strip():  # skip empty pages (e.g. images-only pages)
            pages.append({
                "page": page_num + 1,  # 1-indexed for humans
                "text": text
            })

    doc.close()
    click.echo(f"  [OK] Extracted text from {len(pages)} pages (skipped {total_pages - len(pages)} empty pages)")
    return pages


# ---------------------------------------------------------------------------
# Step 2: Chunk text into overlapping segments
# ---------------------------------------------------------------------------

def chunk_text(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split page texts into overlapping chunks of roughly `chunk_size` words.
    
    Each chunk keeps track of which page(s) it came from, so we can
    cite sources later in the RAG pipeline.
    
    Returns: [{"text": "...", "page": 3, "chunk_id": 0}, ...]
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        words = page_data["text"].split()
        page_num = page_data["page"]

        # Slide a window of `chunk_size` words with `overlap` word stride
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_words) >= 20:  # skip tiny trailing chunks
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page_num,
                    "word_count": len(chunk_words),
                })
                chunk_id += 1

            # Move window forward by (chunk_size - overlap) words
            start += chunk_size - overlap

    click.echo(f"  [OK] Created {len(chunks)} chunks (avg {np.mean([c['word_count'] for c in chunks]):.0f} words each)")
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Generate embeddings and build FAISS index
# ---------------------------------------------------------------------------

def build_index(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2"
) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Embed all chunks using a sentence-transformer model and build a FAISS index.
    
    Uses Inner Product (IP) similarity with normalized vectors,
    which is equivalent to cosine similarity but faster.
    
    Returns: (faiss_index, embeddings_array)
    """
    click.echo(f"  [INFO] Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    click.echo(f"  [INFO] Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # normalize so IP = cosine similarity
        batch_size=32,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index (FlatIP = brute-force inner product, exact results)
    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    click.echo(f"  [OK] Built FAISS index: {index.ntotal} vectors, {dimension} dimensions")
    return index, embeddings


# ---------------------------------------------------------------------------
# Step 4: Save index + metadata to disk
# ---------------------------------------------------------------------------

def save_index(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    output_dir: str
) -> None:
    """
    Save the FAISS index and chunk metadata to disk.
    
    Creates two files:
      - index.faiss  — binary FAISS index for vector search
      - chunks.json  — text + metadata for each chunk (for retrieval)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, index_path)

    # Save chunk metadata (text + page numbers)
    meta_path = os.path.join(output_dir, "chunks.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    click.echo(f"  [OK] Saved index to {index_path}")
    click.echo(f"  [OK] Saved {len(chunks)} chunks to {meta_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--chunk-size", default=500, help="Words per chunk (default: 500)")
@click.option("--overlap", default=50, help="Word overlap between chunks (default: 50)")
@click.option("--output-dir", default=None, help="Output directory (default: data/<pdf_name>)")
@click.option("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
def main(pdf_path: str, chunk_size: int, overlap: int, output_dir: str, model: str):
    """
    Ingest a vehicle manual PDF into a searchable FAISS vector index.
    
    Example:
        python backend/ingest.py data/manual.pdf
    """
    pdf_name = Path(pdf_path).stem

    if output_dir is None:
        output_dir = os.path.join("data", pdf_name)

    click.echo(f"\n[AutoBrain Lite] Ingesting '{pdf_name}'")
    click.echo("=" * 50)

    # Step 1: Extract text
    click.echo("\n[Step 1/4] Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        click.echo("  [ERROR] No text found in PDF. Is it a scanned/image-only PDF?")
        sys.exit(1)

    # Step 2: Chunk text
    click.echo(f"\n[Step 2/4] Chunking text (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)

    # Step 3: Embed and build index
    click.echo(f"\n[Step 3/4] Generating embeddings with '{model}'...")
    index, embeddings = build_index(chunks, model_name=model)

    # Step 4: Save
    click.echo(f"\n[Step 4/4] Saving to '{output_dir}'...")
    save_index(index, chunks, output_dir)

    click.echo(f"\n[SUCCESS] Done! Index ready at '{output_dir}/'")
    click.echo(f"   Chunks: {len(chunks)} | Dimensions: {embeddings.shape[1]} | Index size: {index.ntotal}")


if __name__ == "__main__":
    main()
