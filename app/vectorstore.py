import sys
from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List
from app.core.embedding import ChunkEmbedding, embed_chunk
from app.core.ingestion import IngestedDocument, extract_with_markitdown
from app.models import ChunkRetrievalResult, ChunkRetrievalResults
from app.db.connections import get_db_manager
from app.config import get_config

# --- Initialize ChromaDB ---
def get_chroma_collection():
    """Get ChromaDB collection using centralized database manager."""
    db_manager = get_db_manager()
    return db_manager.get_chroma_collection()

# --- Add Embedded Chunks ---
def add_chunk_embeddings(chunk_embs: List[ChunkEmbedding], doc_id: str, batch_size: int = None):
    """Add chunk embeddings to ChromaDB using configurable batch size."""
    config = get_config()
    if batch_size is None:
        batch_size = config.processing.batch_size * 1000  # Scale up for ChromaDB operations
    
    collection = get_chroma_collection()
    total = len(chunk_embs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunk_embs[start:end]
        ids = [f"{doc_id}_chunk{i}" for i in range(start, end)]
        embeddings = [ce.embedding for ce in batch]
        metadatas = [{"doc_id": doc_id, "chunk_index": ce.chunk_index} for ce in batch]
        documents = [ce.chunk.text for ce in batch]
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        logging.info(f"Added batch {start}-{end-1} of {total} chunks for doc_id '{doc_id}' to ChromaDB.")

# --- Query ChromaDB ---
def query_chunks(query: str, top_k: int = 3) -> ChunkRetrievalResults:
    """Query ChromaDB for relevant chunks using centralized configuration."""
    config = get_config()
    collection = get_chroma_collection()
    
    # Use configured embedding model
    query_emb = embed_chunk(query, model=config.models.embedding_model)
    print(f"[DEBUG] Query embedding dimension: {len(query_emb) if hasattr(query_emb, '__len__') else 'unknown'}")
    
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    logging.info(f"Queried ChromaDB for top {top_k} results.")
    
    # Parse ChromaDB output into Pydantic models
    # ChromaDB returns: {'ids': [[...]], 'documents': [[...]], 'embeddings': None, 'metadatas': [[...]], 'distances': [[...]]}
    chunk_results = []
    ids = results.get('ids', [[]])[0]
    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0] if 'distances' in results else [None]*len(ids)
    
    for idx, (doc, meta, score) in enumerate(zip(documents, metadatas, distances)):
        chunk_results.append(ChunkRetrievalResult(
            text=doc,
            doc_id=meta.get('doc_id', ''),
            chunk_index=meta.get('chunk_index', -1),
            score=score if score is not None else 0.0,
            metadata=meta
        ))
    
    # PATCH: Deduplicate by text, preserving order and top score
    seen = set()
    deduped_results = []
    for cr in chunk_results:
        norm_text = cr.text.strip()
        if norm_text not in seen:
            seen.add(norm_text)
            deduped_results.append(cr)
    
    if len(deduped_results) < len(chunk_results):
        logging.info(f"[VECTORSTORE] Deduplicated {len(chunk_results) - len(deduped_results)} duplicate chunks from top {top_k} results.")
    
    return ChunkRetrievalResults(results=deduped_results)

# --- CLI Test ---
if __name__ == "__main__":
    import json
    import argparse
    from app.core.serialization import load_ingested_document
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Query ChromaDB using pre-chunked documents. Ingestion/embedding must be done via ingest_parallel.")
    parser.add_argument("--chunks-file", required=True, help="Path to pre-chunked IngestedDocument JSON file (produced by ingest_parallel)")
    parser.add_argument("--query", required=True, help="Query string to search ChromaDB")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top results to return")
    args = parser.parse_args()

    doc = load_ingested_document(args.chunks_file)
    # Query ChromaDB
    results = query_chunks(args.query, top_k=args.top_k)
    print(json.dumps(results.dict(), indent=2))
