import argparse
import tempfile
import os
import concurrent.futures
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from app.core.ingestion import extract_with_markitdown
from app.core.serialization import save_ingested_document, load_ingested_document
from app.vectorstore import add_chunk_embeddings
from app.core.embedding import embed_document
from app.graph.ingestion_core import ingest_document_to_neo4j
from app.models import IngestedDocument
from app.utils.status import set_backend_status

def cleanup_temp_files(temp_dir=None):
    """Delete all temp chunk files (tmp*.txt) in the working/temp directory."""
    if temp_dir is None:
        temp_dir = os.getcwd()
    for fname in os.listdir(temp_dir):
        if fname.startswith("tmp") and fname.endswith(".txt"):
            try:
                os.remove(os.path.join(temp_dir, fname))
                logging.info(f"[CLEANUP] Deleted temp file: {fname}")
            except Exception as e:
                logging.warning(f"[CLEANUP] Failed to delete {fname}: {e}")

def run_vector_ingest(chunks_path, doc_id, embedding_model="mxbai-embed-large"):
    set_backend_status(doc_id, "vector", "processing")
    doc = load_ingested_document(chunks_path)
    total = len(doc.chunks)
    logging.info(f"[VECTOR] Starting vector embedding for {total} chunks...")
    # Use more workers and larger batch size for faster embedding
    chunk_embs = embed_document(doc, model=embedding_model, max_workers=8, batch_size=16)
    add_chunk_embeddings(chunk_embs, doc_id=doc_id)
    set_backend_status(doc_id, "vector", "ready")
    logging.info(f"[VECTOR] Vector ingestion complete for doc_id={doc_id}.")
    return "vector"

def run_graph_ingest(chunks_path, doc_id, extractor_model="gemma3:1b", entity_extractor="llama"):
    set_backend_status(doc_id, "graph", "processing")
    doc = load_ingested_document(chunks_path)
    total = len(doc.chunks)
    logging.info(f"[GRAPH] Starting graph ingestion for {total} chunks...")
    if entity_extractor == "spacy":
        from app.entity_extraction.spacy_extractor import extract_entities_and_relations_spacy
        def extractor(text):
            return extract_entities_and_relations_spacy(text)
    else:
        from app.entity_extraction import extract_entities_and_relations_llm
        def extractor(text):
            return extract_entities_and_relations_llm(text, model=extractor_model)
    for idx, chunk in enumerate(doc.chunks):
        preview = chunk.text[:80].replace('\n', ' ')
        logging.info(f"[GRAPH][CHUNK {idx+1}/{total}] Preview: '{preview}...' ")
        result = extractor(chunk.text)
        if isinstance(result, (tuple, list)) and len(result) == 3:
            entities, relations, _ = result
        elif isinstance(result, (tuple, list)) and len(result) == 2:
            entities, relations = result
            _ = None
        else:
            raise ValueError("Extractor must return 2 or 3 values (entities, relations[, summary])")
        logging.info(f"[GRAPH][CHUNK {idx+1}/{total}] Entities: {entities}")
        logging.info(f"[GRAPH][CHUNK {idx+1}/{total}] Relations: {relations}")
    ingest_document_to_neo4j(doc, extractor=extractor, repair_relations=False)
    set_backend_status(doc_id, "graph", "ready")
    logging.info(f"[GRAPH] Graph ingestion complete for doc_id={doc_id}.")
    return "graph"

def main():
    cleanup_temp_files()
    parser = argparse.ArgumentParser(description="Parallel ingestion for vector and graph DBs")
    parser.add_argument("file", type=str, help="Path to input file")
    parser.add_argument("--chunk-strategy", choices=["char", "word", "paragraph", "content-aware"], default="content-aware")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--embedding-model", type=str, default="mxbai-embed-large")
    parser.add_argument("--extractor-model", type=str, default="gemma3:1b")
    parser.add_argument("--entity-extractor", type=str, choices=["spacy", "llama"], default="llama", help="Entity extraction method for graph ingestion (spacy or llama)")
    args = parser.parse_args()

    logging.info(f"[MAIN] Starting chunking for {args.file} (strategy={args.chunk_strategy}, size={args.chunk_size}, overlap={args.overlap})")
    doc = extract_with_markitdown(Path(args.file), chunk_strategy=args.chunk_strategy, chunk_size=args.chunk_size, overlap=args.overlap)
    logging.info(f"[MAIN] Chunking complete: {len(doc.chunks)} chunks generated.")
    # Persist chunked document in a dedicated folder
    persistent_chunks_dir = Path("chunks"); persistent_chunks_dir.mkdir(exist_ok=True)
    persistent_chunks_path = persistent_chunks_dir / f"{doc.metadata.filename}.chunks.json"
    save_ingested_document(doc, str(persistent_chunks_path))
    logging.info(f"[MAIN] Chunked document persisted to {persistent_chunks_path}")
    # Use a temp file for parallel ingestion as before
    with tempfile.NamedTemporaryFile(delete=False, suffix=".chunks.json") as tf:
        save_ingested_document(doc, tf.name)
        chunks_path = tf.name
    logging.info(f"[MAIN] Chunked document saved to {chunks_path}")

    logging.info("[MAIN] Starting parallel ingestion: [vector] and [graph] ...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_vector = executor.submit(run_vector_ingest, chunks_path, doc.metadata.filename, args.embedding_model)
        future_graph = executor.submit(run_graph_ingest, chunks_path, doc.metadata.filename, args.extractor_model, args.entity_extractor)
        completed = set()
        for future in concurrent.futures.as_completed([future_vector, future_graph]):
            result = future.result()
            completed.add(result)
            logging.info(f"[MAIN] {result.capitalize()} ingestion finished.")
    os.remove(chunks_path)
    logging.info(f"[MAIN] Deleted temp chunk file {chunks_path}")
    logging.info(f"[MAIN] Persistent chunked file remains at {persistent_chunks_path}")

if __name__ == "__main__":
    main()
