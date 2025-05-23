import os
import pytest
from app import langchain_rag

TEST_PDF = os.path.abspath(os.path.join(os.path.dirname(__file__), '../andreis_cv_mar_25_bbc.pdf'))
TEST_TXT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../partial_promessi_sposi.txt'))

def test_ingest_and_query_pdf_and_txt():
    """Test multi-file ingestion and retrieval for PDF and TXT files."""
    query = "who worked at Nesta?"
    results = langchain_rag.ingest_and_query(
        file_path=TEST_PDF,
        chunk_strategy="char",
        chunk_size=1000,
        chunk_overlap=100,
        query=query,
        persist_dir="./test_chroma_db",
        use_markitdown=False
    )
    assert any("Nesta" in doc.page_content for doc in results), "Should find 'Nesta' in PDF results"

    results = langchain_rag.ingest_and_query(
        file_path=TEST_TXT,
        chunk_strategy="char",
        chunk_size=1000,
        chunk_overlap=100,
        query="Renzo",
        persist_dir="./test_chroma_db",
        use_markitdown=False
    )
    assert any("Renzo" in doc.page_content for doc in results), "Should find 'Renzo' in TXT results"

@pytest.mark.parametrize("file_path,query,expected", [
    (TEST_PDF, "who worked at Nesta?", "Nesta"),
    (TEST_TXT, "Renzo", "Renzo"),
])
def test_individual_file_ingestion(file_path, query, expected):
    results = langchain_rag.ingest_and_query(
        file_path=file_path,
        chunk_strategy="char",
        chunk_size=1000,
        chunk_overlap=100,
        query=query,
        persist_dir="./test_chroma_db",
        use_markitdown=False
    )
    assert any(expected in doc.page_content for doc in results), f"Should find '{expected}' in results"

if __name__ == "__main__":
    pytest.main([__file__])
