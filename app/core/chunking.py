from typing import List, Literal
import logging
import re
from app.models import DocumentChunk

ChunkStrategy = Literal["char", "word", "paragraph", "content-aware"]

try:
    import spacy
    _spacy_nlp = spacy.blank("en")
    _spacy_nlp.add_pipe("sentencizer")
except ImportError:
    _spacy_nlp = None
    logging.warning("[CHUNKING] spaCy not available, 'content-aware' chunking will not work.")


def chunk_text(
    text: str,
    strategy: ChunkStrategy = "content-aware",
    chunk_size: int = 300,  # ~2-4 sentences, ~200-400 tokens
    overlap: int = 1       # 1 sentence overlap for both vector and KG
) -> List[DocumentChunk]:
    """
    Split text into chunks according to the chosen strategy.
    - strategy: 'char', 'word', 'paragraph', or 'content-aware'
    - chunk_size: size of each chunk (chars or words or sentences)
    - overlap: overlap size (chars, words, or sentences)
    Returns a list of DocumentChunk.
    """
    chunks = []
    if strategy == "content-aware":
        if not _spacy_nlp:
            raise ImportError("spaCy is required for content-aware chunking.")
        # Sentence segmentation
        doc = _spacy_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        # Group sentences into chunks of approx chunk_size (by char length)
        current_chunk = []
        current_len = 0
        for sent in sentences:
            if current_len + len(sent) > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(text=chunk_text))
                # Overlap: keep last N sentences
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:]
                    current_len = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_len = 0
            current_chunk.append(sent)
            current_len += len(sent)
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk(text=chunk_text))
    elif strategy == "char":
        # PATCH: Avoid splitting words between chunks
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Extend end to the next whitespace unless at end of text
            if end < len(text):
                match = re.search(r"\s", text[end:])
                if match:
                    end += match.start()
            chunk_text = text[start:end]
            chunks.append(DocumentChunk(text=chunk_text))
            if end == len(text):
                break
            # For overlap, move back by overlap but not into previous word
            next_start = end - overlap
            if next_start > start:
                # Move next_start to next whitespace to avoid splitting words
                match = re.search(r"\S", text[next_start:])
                if match:
                    next_start += match.start()
                start = next_start
            else:
                start = end
    elif strategy == "word":
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(DocumentChunk(text=chunk_text))
            if end == len(words):
                break
            start += chunk_size - overlap
    elif strategy == "paragraph":
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append(DocumentChunk(text=para))
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    logging.info(f"Chunked text into {len(chunks)} chunks using strategy '{strategy}'.")
    return chunks
