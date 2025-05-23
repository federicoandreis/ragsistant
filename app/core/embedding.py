import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List
import ollama
from pydantic import BaseModel
from ..models import IngestedDocument, DocumentChunk
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config import get_config

class ChunkEmbedding(BaseModel):
    chunk: DocumentChunk
    embedding: List[float]
    chunk_index: int

# --- Utility: Preprocess chunk text for embedding ---
def preprocess_chunk_for_embedding(chunk, add_instruction=True, highlight_entities=True):
    """Preprocess chunk text for better embedding quality."""
    text = chunk.text
    
    # Entity highlighting (for spaCy-extracted entities)
    if highlight_entities:
        try:
            # Import here to avoid circular imports
            from app.entity_extraction import extract_entities_and_relations_spacy
            entities, _, _ = extract_entities_and_relations_spacy(text)
            for ent in entities:
                name = ent.get("name")
                if name:
                    # Highlight in text (bold markdown)
                    text = text.replace(name, f"**{name}**")
        except Exception as e:
            logging.warning(f"Entity highlighting failed: {e}")
    
    # Instructional prefix
    if add_instruction:
        text = f"Context for semantic search and KG extraction: {text}"
    
    return text

def embed_chunk(text: str, model: str = None, preprocess=False, chunk_obj=None) -> list:
    """
    Embed a single chunk of text using Ollama.
    
    Args:
        text: Text to embed
        model: Ollama model name (defaults to config)
        preprocess: Whether to preprocess the text
        chunk_obj: Chunk object for preprocessing
        
    Returns:
        List of embedding floats
    """
    config = get_config()
    if model is None:
        model = config.models.embedding_model
    
    if preprocess and chunk_obj is not None:
        text = preprocess_chunk_for_embedding(chunk_obj)
    
    result = ollama.embed(model=model, input=text)
    
    # Robustly handle possible Ollama response formats
    # 1. If it's a dict with 'embedding'
    if isinstance(result, dict):
        if 'embedding' in result:
            emb = result['embedding']
            # PATCH: flatten if [[...]]
            if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
                emb = emb[0]
            return emb
    
    if hasattr(result, 'embedding'):
        emb = result.embedding
        if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
            emb = emb[0]
        return emb
    
    # 2. If it's a list (legacy API)
    if isinstance(result, list):
        # PATCH: flatten if [[...]]
        if len(result) == 1 and isinstance(result[0], list):
            return result[0]
        return result
    
    # 3. Try to find embedding in attributes
    if hasattr(result, '__dict__'):
        for k, v in result.__dict__.items():
            if 'embed' in k and isinstance(v, list):
                if len(v) == 1 and isinstance(v[0], list):
                    return v[0]
                return v
    
    for attr in dir(result):
        if 'embed' in attr and isinstance(getattr(result, attr), list):
            v = getattr(result, attr)
            if len(v) == 1 and isinstance(v[0], list):
                return v[0]
            return v
    
    # 4. Log all attributes for debugging (only if error)
    logging.error(f"[embed_chunk][ERROR] Unexpected Ollama embedding result format: {result} (type: {type(result)})")
    logging.error(f"[embed_chunk][DEBUG] All attributes: {dir(result)}")
    logging.error(f"[embed_chunk][DEBUG] __dict__: {getattr(result, '__dict__', None)}")
    raise ValueError(f"Unexpected Ollama embedding result format: {result} (type: {type(result)})")

def embed_document(doc: IngestedDocument, model: str = None, show_progress: bool = True, 
                  max_workers: int = None, batch_size: int = None, preprocess=True) -> List[ChunkEmbedding]:
    """
    Embed all chunks in a document using configurable parameters.
    
    Args:
        doc: Document to embed
        model: Ollama model name (defaults to config)
        show_progress: Whether to show progress
        max_workers: Number of parallel workers (defaults to config)
        batch_size: Batch size (defaults to config)
        preprocess: Whether to preprocess chunks
        
    Returns:
        List of ChunkEmbedding objects
    """
    config = get_config()
    
    # Use config defaults if not specified
    if model is None:
        model = config.models.embedding_model
    if max_workers is None:
        max_workers = config.processing.max_workers
    if batch_size is None:
        batch_size = config.processing.batch_size
    
    total = len(doc.chunks)
    results = [None] * total
    seen_texts = set()
    
    def embed_batch(batch_start):
        batch_end = min(batch_start + batch_size, total)
        batch_chunks = doc.chunks[batch_start:batch_end]
        batch_results = []
        
        for i, chunk in enumerate(batch_chunks):
            # Deduplicate
            norm_text = chunk.text.strip()
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            
            # Preprocess and embed
            emb = embed_chunk(chunk.text, model=model, preprocess=preprocess, chunk_obj=chunk)
            idx = batch_start + i
            
            # Metadata enrichment
            meta = {
                "filename": getattr(doc.metadata, "filename", None),
                "filetype": getattr(doc.metadata, "filetype", None),
                "chunk_index": idx
            }
            batch_results.append((idx, ChunkEmbedding(chunk=chunk, embedding=emb, chunk_index=idx)))
        
        return batch_results
    
    batch_starts = list(range(0, total, batch_size))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_batch, batch_start) for batch_start in batch_starts]
        completed = 0
        
        for future in as_completed(futures):
            batch_results = future.result()
            for idx, chunk_emb in batch_results:
                results[idx] = chunk_emb
                completed += 1
                if show_progress:
                    logging.info(f"[EMBEDDING] Embedded {completed}/{total} chunks...")
    
    if show_progress:
        logging.info(f"[EMBEDDING] Done embedding {total} chunks.")
    
    # Remove None (deduped)
    results = [r for r in results if r is not None]
    return results

# --- CLI Test ---
if __name__ == "__main__":
    import sys
    import argparse
    from app.core.ingestion import extract_with_markitdown
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Test Ollama embedding performance.")
    parser.add_argument("file", type=str, help="Path to input file")
    parser.add_argument("--max-workers", type=int, help="Number of parallel batch workers (default: from config)")
    parser.add_argument("--batch-size", type=int, help="Batch size for each embedding request (default: from config)")
    parser.add_argument("--model", type=str, help="Ollama embedding model to use (default: from config)")
    args = parser.parse_args()
    
    p = Path(args.file)
    doc = extract_with_markitdown(p)
    
    # Use the model specified in the CLI or config
    config = get_config()
    model = args.model or config.models.embedding_model
    max_workers = args.max_workers or config.processing.max_workers
    batch_size = args.batch_size or config.processing.batch_size
    
    print(f"\n=== Testing embedding with model: {model} (max_workers={max_workers}, batch_size={batch_size}) ===")
    
    import time
    start = time.time()
    chunk_embs = embed_document(doc, model=model, max_workers=max_workers, batch_size=batch_size)
    end = time.time()
    
    print(f"Total embedding time for {model}: {end-start:.2f}s\n")
    
    for ce in chunk_embs[:2]:
        logging.info(f"Chunk {ce.chunk_index}: {len(ce.embedding)}-dim vector")
        logging.info(ce.chunk.text[:100].replace("\n", " ") + ("..." if len(ce.chunk.text) > 100 else ""))
        logging.info("")

    # GPU detection (Ollama)
    print("\n=== Checking Ollama GPU support ===")
    import platform
    import subprocess
    if platform.system() == "Windows":
        try:
            result = subprocess.run(["ollama", "run", "--help"], capture_output=True, text=True)
            if "--gpu" in result.stdout:
                print("Ollama CLI shows GPU flag (experimental, may require config).")
            else:
                print("Ollama CLI does not show GPU flag. Most Windows installs are CPU-only.")
        except Exception as e:
            print(f"Could not check Ollama GPU support: {e}")
    else:
        print("Non-Windows system: please check Ollama documentation for GPU support.")
