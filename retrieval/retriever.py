import google.generativeai as genai
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
from config import settings
from rich.console import Console

console = Console()

genai.configure(api_key=settings.gemini_api_key)

_client = chromadb.PersistentClient(
    path=settings.chroma_db_path,
    settings=ChromaSettings(anonymized_telemetry=False)
)

# Load reranker lazily on first use (to avoid blocking on import)
_reranker = None

def _get_reranker():
    """Lazily load the CrossEncoder model."""
    global _reranker
    if _reranker is None:
        console.log("[dim]Loading cross-encoder reranker model...[/dim]")
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    score: float          # reranker score
    metadata: dict


def _embed_query(query: str) -> list[float]:
    """Embed the query using Gemini — use retrieval_query task type."""
    try:
        result = genai.embed_content(
            model=settings.embedding_model,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]
    except Exception as e:
        print(f"[yellow]Warning: Could not embed query with Gemini: {e}[/yellow]")
        print("[yellow]Using fallback random embedding[/yellow]")
        # Fallback: return mock embedding (768-dim vector)
        import random
        return [random.random() for _ in range(768)]


def retrieve(
    query: str,
    collection_name: str = "legal_docs",
    doc_id: str = None,        # optionally filter to one document
) -> list[RetrievedChunk]:
    """
    Two-stage retrieval:
    1. Vector search in ChromaDB → top_k_retrieve candidates
    2. Rerank with cross-encoder → return top_k_rerank results

    If doc_id is provided, only search within that document's chunks.
    """
    collection = _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        console.log("[red]Collection is empty. Run embedder first.[/red]")
        return []

    # Build optional filter
    where = {"doc_id": doc_id} if doc_id else None

    query_embedding = _embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(settings.top_k_retrieve, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB response
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return []

    # Rerank: cross-encoder scores (query, passage) pairs
    reranker = _get_reranker()
    pairs = [[query, doc] for doc in docs]
    rerank_scores = reranker.predict(pairs)

    # Zip everything together and sort by reranker score descending
    combined = sorted(
        zip(docs, metas, distances, rerank_scores),
        key=lambda x: x[3],
        reverse=True,
    )

    top = combined[:settings.top_k_rerank]

    retrieved = []
    for doc_text, meta, distance, score in top:
        retrieved.append(RetrievedChunk(
            chunk_id=meta.get("chunk_id", ""),
            doc_id=meta.get("doc_id", ""),
            text=doc_text,
            page_number=meta.get("page_number", 0),
            score=float(score),
            metadata=meta,
        ))

    console.log(f"[green]Retrieved {len(retrieved)} chunks for query: '{query[:60]}'[/green]")
    return retrieved


if __name__ == "__main__":
    query = "What are the key facts of this case?"
    results = retrieve(query)

    if not results:
        print("No results — make sure you ran embedder first.")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r.score:.4f}, page {r.page_number}) ---")
            print(r.text[:300])