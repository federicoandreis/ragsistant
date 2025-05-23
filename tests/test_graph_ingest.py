import pytest
from pathlib import Path
from app.ingestion import extract_txt
from app.graph_ingest import ingest_document_to_neo4j
from app.models import IngestedDocument

TEST_TXT = "partial_promessi_sposi.txt"

def test_graph_ingest_txt():
    # Extract the document as a single chunk (for MVP)
    doc = extract_txt(Path(TEST_TXT))
    assert isinstance(doc, IngestedDocument)
    # Ingest into Neo4j (should not raise)
    ingest_document_to_neo4j(doc)
    # If no exception, ingestion succeeded (for MVP)
    # Further assertions can be made by querying Neo4j, but this requires a test DB
