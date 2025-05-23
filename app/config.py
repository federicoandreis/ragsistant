"""
Centralized configuration for the RAG system.
All configuration constants should be defined here to avoid duplication across modules.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue with system environment variables
    pass


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    chroma_persist_dir: str = os.getenv("CHROMA_DB_PATH", "storage/chroma_db")
    chroma_collection: str = "rag_chunks"
    neo4j_data_path: str = os.getenv("NEO4J_DATA_PATH", "storage/neo4j_data")


@dataclass
class ModelConfig:
    """Model configuration for various AI components."""
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    routing_model: str = os.getenv("ROUTING_MODEL", "gemma3:1b")
    synthesis_model: str = os.getenv("SYNTHESIS_MODEL", "gemma3:1b")
    entity_extraction_model: str = os.getenv("ENTITY_MODEL", "gemma3:1b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_dimension: int = get_env_int("EMBEDDING_DIMENSION", 1024)


@dataclass
class ProcessingConfig:
    """Configuration for document processing and chunking."""
    default_chunk_size: int = get_env_int("DEFAULT_CHUNK_SIZE", 1000)
    default_chunk_overlap: int = get_env_int("DEFAULT_CHUNK_OVERLAP", 200)
    max_workers: int = get_env_int("MAX_WORKERS", 4)
    batch_size: int = get_env_int("BATCH_SIZE", 8)
    max_relationships: int = get_env_int("MAX_RELATIONSHIPS", 10)
    max_hops: int = get_env_int("MAX_HOPS", 1)


@dataclass
class PathConfig:
    """File and directory path configuration."""
    documents_path: str = os.getenv("DOCUMENTS_PATH", "data/documents")
    chunks_path: str = os.getenv("CHUNKS_PATH", "data/chunks")
    logs_path: str = os.getenv("LOGS_PATH", "logs")
    
    def __post_init__(self):
        """Ensure all paths exist."""
        for path_attr in ['documents_path', 'chunks_path', 'logs_path']:
            path = Path(getattr(self, path_attr))
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = os.getenv("APP_NAME", "RAGsistant")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    debug: bool = get_env_bool("DEBUG", False)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    database: DatabaseConfig = None
    models: ModelConfig = None
    processing: ProcessingConfig = None
    paths: PathConfig = None
    
    def __post_init__(self):
        """Initialize nested configs and validate configuration after initialization."""
        if self.database is None:
            self.database = DatabaseConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.paths is None:
            self.paths = PathConfig()
            
        # Validate configuration
        if self.processing.max_workers < 1:
            self.processing.max_workers = 1
        if self.processing.batch_size < 1:
            self.processing.batch_size = 1
            
        # Ensure database directories exist
        Path(self.database.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.database.neo4j_data_path).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
