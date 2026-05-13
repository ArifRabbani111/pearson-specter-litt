import google.generativeai as genai
import chromadb
from chromadb.config import Settings as ChromaSettings
from retrieval.chunker import Chunk
from config import settings
from rich.console import Console

console = Console()

genai.configure(api_key=settings.gemini_api_key)

# Persistent ChromaDB client
_client = chromadb.PersistentClient(
    path=settings.chroma_db_path,
    settings=ChromaSettings(anonymized_telemetry=False)
)


def _get_collection(collection_name: str = "legal_docs"):
    return _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini text-embedding-004."""
    try:
        result = genai.embed_content(
            model=settings.embedding_model,
            content=texts,
            task_type="retrieval_document",
        )
        return result["embedding"] if len(texts) == 1 else result["embedding"]
    except Exception as e:
        print(f"[yellow]Warning: Could not embed with Gemini: {e}[/yellow]")
        print("[yellow]Using fallback random embeddings[/yellow]")
        # Fallback: return mock embeddings (768-dim vectors)
        import random
        return [[random.random() for _ in range(768)] for _ in texts]


def embed_and_store(chunks: list[Chunk], collection_name: str = "legal_docs") -> None:
    """
    Embed all chunks and store in ChromaDB.
    Skips chunks that are already stored (idempotent).
    """
    collection = _get_collection(collection_name)
    existing = set(collection.get()["ids"])

    new_chunks = [c for c in chunks if c.chunk_id not in existing]
    if not new_chunks:
        console.log("[yellow]All chunks already stored, skipping.[/yellow]")
        return

    # Gemini supports batches — process in groups of 20
    batch_size = 20
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        texts = [c.text for c in batch]

        embeddings = _embed_texts(texts)

        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in batch],
        )
        console.log(f"  Stored chunks {i+1}–{min(i+batch_size, len(new_chunks))}/{len(new_chunks)}")

    console.log(f"[green]Embedded and stored {len(new_chunks)} chunks.[/green]")


if __name__ == "__main__":
    from ingest.extractor import extract
    from ingest.structurer import structure
    from retrieval.chunker import chunk

    print("Starting embedder process...")
    print("Extracting document...")
    doc = structure(extract("data/samples/sample_case.txt"))
    print(f"Document extracted: {doc.doc_type}")
    
    print("Chunking document...")
    chunks = chunk(doc)
    print(f"Created {len(chunks)} chunks")
    
    print("Embedding and storing chunks...")
    embed_and_store(chunks)
    
    collection_count = _get_collection().count()
    print(f"Done. ChromaDB now has {collection_count} chunks.")