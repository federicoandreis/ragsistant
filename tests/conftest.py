"""Pytest configuration and fixtures for RAGsistant tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    This is a sample document for testing purposes.
    It contains multiple sentences and paragraphs.
    
    The document discusses various topics including:
    - Natural language processing
    - Information retrieval
    - Knowledge graphs
    
    This text will be used to test chunking, embedding, and other functionality.
    """

@pytest.fixture
def mock_ollama():
    """Mock Ollama client for testing."""
    with patch('ollama.Client') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock embedding response
        mock_instance.embeddings.return_value = {
            'embedding': [0.1] * 768
        }
        
        # Mock chat response
        mock_instance.chat.return_value = {
            'message': {
                'content': 'This is a test response from the mocked LLM.'
            }
        }
        
        yield mock_instance

@pytest.fixture
def mock_neo4j():
    """Mock Neo4j driver for testing."""
    with patch('neo4j.GraphDatabase.driver') as mock_driver:
        mock_session = Mock()
        mock_driver.return_value.session.return_value = mock_session
        
        # Mock query results
        mock_session.run.return_value = []
        
        yield mock_driver

@pytest.fixture
def mock_chroma():
    """Mock ChromaDB client for testing."""
    with patch('chromadb.PersistentClient') as mock_client:
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Mock query results
        mock_collection.query.return_value = {
            'documents': [['Sample document content']],
            'metadatas': [[{'source': 'test.txt'}]],
            'distances': [[0.5]]
        }
        
        yield mock_client

@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        {
            'text': 'John Doe',
            'label': 'PERSON',
            'start': 0,
            'end': 8
        },
        {
            'text': 'New York',
            'label': 'GPE',
            'start': 20,
            'end': 28
        }
    ]

@pytest.fixture
def sample_relations():
    """Sample relations for testing."""
    return [
        {
            'source': 'John Doe',
            'target': 'New York',
            'relation': 'LIVES_IN',
            'confidence': 0.8
        }
    ]

@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        {
            'text': 'This is the first chunk of text.',
            'metadata': {'source': 'test.txt', 'chunk_id': 0}
        },
        {
            'text': 'This is the second chunk of text.',
            'metadata': {'source': 'test.txt', 'chunk_id': 1}
        }
    ]
