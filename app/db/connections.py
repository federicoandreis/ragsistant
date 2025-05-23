"""
Centralized database connection management.
Provides connection pooling and consistent error handling for all database operations.
"""
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from neo4j import GraphDatabase, basic_auth, Session
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from app.config import get_config


class DatabaseManager:
    """Manages database connections and provides context managers for safe access."""
    
    def __init__(self):
        self.config = get_config()
        self._neo4j_driver: Optional[GraphDatabase.driver] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        
    def _get_neo4j_driver(self):
        """Get or create Neo4j driver with connection pooling."""
        if self._neo4j_driver is None:
            try:
                self._neo4j_driver = GraphDatabase.driver(
                    self.config.database.neo4j_uri,
                    auth=basic_auth(
                        self.config.database.neo4j_username,
                        self.config.database.neo4j_password
                    )
                )
                logging.info(f"Connected to Neo4j at {self.config.database.neo4j_uri}")
            except Exception as e:
                logging.error(f"Failed to connect to Neo4j: {e}")
                raise
        return self._neo4j_driver
    
    @contextmanager
    def neo4j_session(self) -> Generator[Session, None, None]:
        """Context manager for Neo4j sessions with automatic cleanup."""
        driver = self._get_neo4j_driver()
        session = None
        try:
            session = driver.session()
            yield session
        except Exception as e:
            logging.error(f"Neo4j session error: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def get_chroma_client(self):
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            try:
                self._chroma_client = chromadb.PersistentClient(
                    path=self.config.database.chroma_persist_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
                logging.info(f"Connected to ChromaDB at {self.config.database.chroma_persist_dir}")
            except Exception as e:
                logging.error(f"Failed to connect to ChromaDB: {e}")
                raise
        return self._chroma_client
    
    def get_chroma_collection(self):
        """Get or create the main ChromaDB collection with dimension validation."""
        client = self.get_chroma_client()
        embedding_function = OllamaEmbeddingFunction(
            model_name=self.config.models.embedding_model,
            url=f"{self.config.models.ollama_base_url}/api/embeddings"
        )
        
        try:
            collection_names = [col.name for col in client.list_collections()]
            
            if self.config.database.chroma_collection not in collection_names:
                logging.info(f"Creating new ChromaDB collection: {self.config.database.chroma_collection}")
                collection = client.create_collection(
                    name=self.config.database.chroma_collection,
                    embedding_function=embedding_function
                )
            else:
                logging.info(f"Using existing ChromaDB collection: {self.config.database.chroma_collection}")
                try:
                    collection = client.get_collection(
                        name=self.config.database.chroma_collection,
                        embedding_function=embedding_function
                    )
                    # Test the collection with a dummy embedding to check dimensions
                    self._validate_collection_dimensions(collection, embedding_function)
                except Exception as dim_error:
                    if "dimension" in str(dim_error).lower():
                        logging.warning(f"Dimension mismatch detected: {dim_error}")
                        logging.info("Recreating collection with correct dimensions...")
                        # Delete and recreate the collection
                        client.delete_collection(name=self.config.database.chroma_collection)
                        collection = client.create_collection(
                            name=self.config.database.chroma_collection,
                            embedding_function=embedding_function
                        )
                    else:
                        raise dim_error
            return collection
        except Exception as e:
            logging.error(f"Failed to get ChromaDB collection: {e}")
            # Fallback: try to get collection without checking if it exists
            try:
                collection = client.get_collection(
                    name=self.config.database.chroma_collection,
                    embedding_function=embedding_function
                )
                return collection
            except Exception as fallback_e:
                logging.error(f"Fallback also failed: {fallback_e}")
                raise e
    
    def _validate_collection_dimensions(self, collection, embedding_function):
        """Validate that the collection dimensions match the embedding function."""
        try:
            # Try a test query to see if dimensions match
            test_embedding = embedding_function._embed_with_retry(["test"])[0]
            collection.query(query_embeddings=[test_embedding], n_results=1)
            logging.debug("Collection dimension validation passed")
        except Exception as e:
            if "dimension" in str(e).lower():
                raise e
            # If it's not a dimension error, ignore it (collection might be empty)
            logging.debug(f"Collection validation skipped: {e}")
    
    def close_connections(self):
        """Close all database connections."""
        if self._neo4j_driver:
            try:
                self._neo4j_driver.close()
                logging.info("Closed Neo4j connection")
            except Exception as e:
                logging.error(f"Error closing Neo4j connection: {e}")
            finally:
                self._neo4j_driver = None
        
        # ChromaDB client doesn't need explicit closing
        self._chroma_client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_connections()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_db_manager():
    """Reset the global database manager (useful for testing)."""
    global _db_manager
    if _db_manager:
        _db_manager.close_connections()
    _db_manager = None
