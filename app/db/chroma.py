"""
ChromaDB connector for vector storage and retrieval operations.
Provides high-level interface for document storage, embedding, and similarity search.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

from app.db.connections import get_db_manager
from app.config import get_config


@dataclass
class ChromaDocument:
    """Represents a document stored in ChromaDB."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class ChromaSearchResult:
    """Represents a search result from ChromaDB."""
    document: ChromaDocument
    distance: float
    score: float  # 1 - distance for similarity score


class ChromaConnector:
    """High-level interface for ChromaDB operations."""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()
        self._collection = None
    
    @property
    def collection(self):
        """Lazy-load the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.db_manager.get_chroma_collection()
        return self._collection
    
    def add_documents(self, documents: List[ChromaDocument]) -> bool:
        """
        Add multiple documents to ChromaDB.
        
        Args:
            documents: List of ChromaDocument objects to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                logging.warning("No documents provided to add_documents")
                return True
            
            # Prepare data for ChromaDB
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to collection (embeddings will be generated automatically)
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            
            logging.info(f"Successfully added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    def add_document(self, document: ChromaDocument) -> bool:
        """
        Add a single document to ChromaDB.
        
        Args:
            document: ChromaDocument object to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.add_documents([document])
    
    def search_similar(
        self, 
        query: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[ChromaSearchResult]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Text query to search for
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            List of ChromaSearchResult objects
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to ChromaSearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    doc = ChromaDocument(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    )
                    
                    distance = results['distances'][0][i]
                    score = max(0.0, 1.0 - distance)  # Convert distance to similarity score
                    
                    search_results.append(ChromaSearchResult(
                        document=doc,
                        distance=distance,
                        score=score
                    ))
            
            logging.info(f"Found {len(search_results)} similar documents for query: '{query[:50]}...'")
            return search_results
            
        except Exception as e:
            logging.error(f"Failed to search ChromaDB: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[ChromaDocument]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            ChromaDocument if found, None otherwise
        """
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids'] and results['ids'][0]:
                return ChromaDocument(
                    id=results['ids'][0],
                    content=results['documents'][0],
                    metadata=results['metadatas'][0] or {}
                )
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to get document {doc_id} from ChromaDB: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            logging.info(f"Deleted document {doc_id} from ChromaDB")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete document {doc_id} from ChromaDB: {e}")
            return False
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete multiple documents by ID.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not doc_ids:
                return True
                
            self.collection.delete(ids=doc_ids)
            logging.info(f"Deleted {len(doc_ids)} documents from ChromaDB")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete documents from ChromaDB: {e}")
            return False
    
    def update_document(self, document: ChromaDocument) -> bool:
        """
        Update an existing document.
        
        Args:
            document: ChromaDocument with updated content/metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.update(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.metadata]
            )
            logging.info(f"Updated document {document.id} in ChromaDB")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update document {document.id} in ChromaDB: {e}")
            return False
    
    def count_documents(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            int: Number of documents
        """
        try:
            return self.collection.count()
        except Exception as e:
            logging.error(f"Failed to count documents in ChromaDB: {e}")
            return 0
    
    def list_documents(
        self, 
        limit: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[ChromaDocument]:
        """
        List documents in the collection with optional filtering.
        
        Args:
            limit: Maximum number of documents to return
            where: Metadata filter conditions
            
        Returns:
            List of ChromaDocument objects
        """
        try:
            results = self.collection.get(
                limit=limit,
                where=where,
                include=['documents', 'metadatas']
            )
            
            documents = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    documents.append(ChromaDocument(
                        id=results['ids'][i],
                        content=results['documents'][i],
                        metadata=results['metadatas'][i] or {}
                    ))
            
            return documents
            
        except Exception as e:
            logging.error(f"Failed to list documents from ChromaDB: {e}")
            return []
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all document IDs
            results = self.collection.get(include=[])
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logging.info(f"Cleared {len(results['ids'])} documents from ChromaDB collection")
            else:
                logging.info("ChromaDB collection was already empty")
            return True
            
        except Exception as e:
            logging.error(f"Failed to clear ChromaDB collection: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if ChromaDB is accessible and working.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Try to get collection info
            count = self.collection.count()
            logging.info(f"ChromaDB health check passed. Collection has {count} documents.")
            return True
            
        except Exception as e:
            logging.error(f"ChromaDB health check failed: {e}")
            return False


def create_document_id(source_file: str, chunk_index: int) -> str:
    """
    Create a consistent document ID for a chunk.
    
    Args:
        source_file: Source file path
        chunk_index: Index of the chunk within the file
        
    Returns:
        str: Unique document ID
    """
    # Create a deterministic ID based on source and chunk index
    base_id = f"{source_file}:chunk:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base_id))


def create_document_from_chunk(
    content: str,
    source_file: str,
    chunk_index: int,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> ChromaDocument:
    """
    Create a ChromaDocument from a text chunk.
    
    Args:
        content: Text content of the chunk
        source_file: Source file path
        chunk_index: Index of the chunk within the file
        additional_metadata: Additional metadata to include
        
    Returns:
        ChromaDocument object
    """
    doc_id = create_document_id(source_file, chunk_index)
    
    metadata = {
        "source_file": source_file,
        "chunk_index": chunk_index,
        "content_length": len(content),
        "doc_type": "chunk"
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return ChromaDocument(
        id=doc_id,
        content=content,
        metadata=metadata
    )
