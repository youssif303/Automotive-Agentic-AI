"""
AutoBrain Lite — Query Tool (RAG Retrieval Only)

Tests the vector search pipeline without the LLM.
Given a question, finds the most relevant chunks from the FAISS index.

Usage:
    python backend/query.py "What is the recommended tire pressure?"
    python backend/query.py "engine warning light" --top-k 3
"""

import json
import os

import click
import faiss
import numpy as np

# Try importing FastEmbed (ONNX runtime, uses ~35MB RAM, ideal for cloud)
try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class Retriever:
    """
    Loads a FAISS index + chunk metadata and performs semantic search.
    
    This class is reused later by the Agent (Day 2) and API (Day 3).
    Keeping it separate follows the Single Responsibility Principle.
    """

    def __init__(self, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load the FAISS index and chunk metadata from disk.
        
        Args:
            index_dir: Directory containing index.faiss and chunks.json
            model_name: Same embedding model used during ingestion
        """
        # Load FAISS index
        index_path = os.path.join(index_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No FAISS index found at {index_path}. Run ingest.py first.")
        self.index = faiss.read_index(index_path)

        # Load chunk metadata
        meta_path = os.path.join(index_dir, "chunks.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load lightweight embedding model (FastEmbed preferred for cloud, SentenceTransformer fallback)
        if HAS_FASTEMBED:
            print("  [INFO] Using FastEmbed (ultra-low memory ONNX Runtime)")
            self.fast_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.model = None
        elif HAS_SENTENCE_TRANSFORMERS:
            print("  [INFO] Using SentenceTransformer")
            self.model = SentenceTransformer(model_name)
            self.fast_model = None
        else:
            raise ImportError("Neither fastembed nor sentence-transformers is installed.")

        click.echo(f"  [OK] Loaded index: {self.index.ntotal} chunks, {self.index.d} dimensions")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top-k most relevant chunks for a given query.
        
        Args:
            query: Natural language question
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys: chunk_id, text, page, score
        """
        if self.fast_model is not None:
            # FastEmbed ONNX embedding (normalized 384-dim vector)
            emb = list(self.fast_model.embed([query]))[0]
            query_embedding = np.array([emb], dtype=np.float32)
        else:
            # SentenceTransformer embedding
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
            )
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search FAISS index (inner product = cosine similarity for normalized vectors)
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for "no result"
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

@click.command()
@click.argument("query")
@click.option("--index-dir", default="data", help="Directory containing the FAISS index")
@click.option("--top-k", default=5, help="Number of results to return (default: 5)")
@click.option("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
def main(query: str, index_dir: str, top_k: int, model: str):
    """
    Search the vehicle manual for relevant content.
    
    Example:
        python backend/query.py "tire pressure" --index-dir data/bmw_manual
    """
    click.echo(f"\n[Search] Searching for: \"{query}\"")
    click.echo("=" * 50)

    # Auto-detect index directory (find first subdirectory with index.faiss)
    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
        for subdir in os.listdir(index_dir):
            candidate = os.path.join(index_dir, subdir)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "index.faiss")):
                index_dir = candidate
                break

    retriever = Retriever(index_dir, model_name=model)
    results = retriever.search(query, top_k=top_k)

    click.echo(f"\n[Results] Top {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        score_bar = "=" * int(result["score"] * 20)  # visual score bar with ASCII
        click.echo(f"  [{i}] Score: {result['score']:.4f} {score_bar}")
        click.echo(f"      Page: {result['page']}")
        # Show first 200 chars of the chunk text
        preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
        click.echo(f"      Text: {preview}")
        click.echo()


if __name__ == "__main__":
    main()
