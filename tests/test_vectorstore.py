import pytest
from app.vectorstore import add_chunk_embeddings, query_chunks
from app.embedding import ChunkEmbedding, embed_chunk
from app.models import ChunkRetrievalResults, DocumentChunk

class DummyChunk:
    def __init__(self, text):
        self.text = text

@pytest.fixture
def dummy_embeddings():
    # Use the real embed_chunk function to get correct dimensionality
    texts = [f"chunk {i}" for i in range(3)]
    return [
        ChunkEmbedding(chunk=DocumentChunk(text=text), embedding=embed_chunk(text), chunk_index=i)
        for i, text in enumerate(texts)
    ]

def test_add_and_query_chunks(dummy_embeddings):
    doc_id = "testdoc"
    add_chunk_embeddings(dummy_embeddings, doc_id=doc_id)
    # Query for a string similar to chunk 1
    results = query_chunks("chunk 1", top_k=2)
    assert isinstance(results, ChunkRetrievalResults)
    assert len(results.results) > 0
    # Check structure of first result
    first = results.results[0]
    assert hasattr(first, "text")
    assert hasattr(first, "doc_id")
    assert hasattr(first, "chunk_index")
    assert hasattr(first, "score")
    assert first.doc_id == doc_id
    assert first.text.startswith("chunk")
