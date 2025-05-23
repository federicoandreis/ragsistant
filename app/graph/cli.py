"""
Command-line interface for graph ingestion.

This module provides:
- CLI argument parsing
- Document loading and processing
- Extractor configuration
- Main execution entry point
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Callable
from app.config import get_config
from app.core.serialization import load_ingested_document
from app.core.ingestion import extract_with_markitdown
from app.entity_extraction import extract_entities_and_relations
from .ingestion_core import ingest_document_to_neo4j


def create_cli_parser() -> argparse.ArgumentParser:
    """
    Create and configure the CLI argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Graph Ingestion CLI - Ingest documents into Neo4j knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.txt --entity-extractor spacy
  %(prog)s document.json --entity-extractor llm --llm-model phi4
  %(prog)s document.pdf --chunk-strategy content-aware --chunk-size 1000
        """
    )
    
    # Required arguments
    parser.add_argument(
        "file", 
        type=str, 
        help="Path to input file (txt, pdf, docx, or pre-chunked JSON)"
    )
    
    # Entity extraction options
    parser.add_argument(
        "--entity-extractor", 
        choices=["spacy", "llm"], 
        default="spacy",
        help="Entity extraction method (default: spacy)"
    )
    
    parser.add_argument(
        "--llm-model", 
        type=str,
        help="Ollama LLM model for entity extraction (defaults to config)"
    )
    
    # Chunking options
    parser.add_argument(
        "--chunk-strategy", 
        type=str, 
        default="char",
        help="Chunking strategy: char, word, paragraph, content-aware (default: char)"
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=int,
        help="Chunk size in characters/words (defaults to config)"
    )
    
    parser.add_argument(
        "--overlap", 
        type=int,
        help="Chunk overlap size (defaults to config)"
    )
    
    # Processing options
    parser.add_argument(
        "--repair-relations", 
        action="store_true",
        help="Attempt to repair malformed relations using LLM"
    )
    
    parser.add_argument(
        "--skip-community-detection", 
        action="store_true",
        help="Skip community detection and analysis"
    )
    
    parser.add_argument(
        "--skip-gap-filling", 
        action="store_true",
        help="Skip gap filling for isolated nodes"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Suppress all output except errors"
    )
    
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Display detailed ingestion statistics"
    )
    
    return parser


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure logging based on verbosity settings.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        quiet: Suppress all output except errors
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def load_document(file_path: str, chunk_strategy: str, chunk_size: Optional[int], overlap: Optional[int]):
    """
    Load and process a document for ingestion.
    
    Args:
        file_path: Path to the input file
        chunk_strategy: Chunking strategy to use
        chunk_size: Size of chunks
        overlap: Overlap between chunks
        
    Returns:
        IngestedDocument ready for processing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if file_path.suffix.lower() == ".json":
        logging.info(f"Loading pre-chunked document from {file_path}")
        return load_ingested_document(str(file_path))
    else:
        logging.info(f"Processing document {file_path} with {chunk_strategy} chunking")
        config = get_config()
        chunk_size = chunk_size or config.processing.default_chunk_size
        overlap = overlap or config.processing.default_chunk_overlap
        
        return extract_with_markitdown(
            file_path, 
            chunk_strategy=chunk_strategy, 
            chunk_size=chunk_size, 
            overlap=overlap
        )


def create_extractor(extractor_type: str, llm_model: Optional[str]) -> Callable:
    """
    Create an entity extraction function based on the specified type.
    
    Args:
        extractor_type: Type of extractor ("spacy" or "llm")
        llm_model: LLM model name for LLM extractor
        
    Returns:
        Configured extractor function
    """
    config = get_config()
    
    if extractor_type == "spacy":
        def extractor(text: str):
            result = extract_entities_and_relations(text, method="spacy")
            entities = [entity.dict() for entity in result.entities]
            relations = [(rel.source, rel.type, rel.target) for rel in result.relations]
            return entities, relations, result.summary
        
        logging.info("Using spaCy entity extractor")
        return extractor
    
    elif extractor_type == "llm":
        model = llm_model or config.models.entity_extraction_model
        
        def extractor(text: str):
            result = extract_entities_and_relations(text, method="llm", model=model)
            entities = [entity.dict() for entity in result.entities]
            relations = [(rel.source, rel.type, rel.target) for rel in result.relations]
            return entities, relations, result.summary
        
        logging.info(f"Using LLM entity extractor with model: {model}")
        return extractor
    
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


def display_statistics(stats: dict) -> None:
    """
    Display ingestion statistics in a formatted way.
    
    Args:
        stats: Statistics dictionary from ingestion
    """
    print("\n" + "="*50)
    print("INGESTION STATISTICS")
    print("="*50)
    
    print(f"Document: {stats.get('document', 'Unknown')}")
    print(f"Chunks processed: {stats.get('chunks_processed', 0)}")
    print(f"Entities extracted: {stats.get('entities_extracted', 0)}")
    print(f"Relations extracted: {stats.get('relations_extracted', 0)}")
    print(f"Entities after coalescing: {stats.get('entities_after_coalescing', 0)}")
    print(f"Relations after remapping: {stats.get('relations_after_remapping', 0)}")
    print(f"Entities written to Neo4j: {stats.get('entities_written', 0)}")
    print(f"Relations written to Neo4j: {stats.get('relations_written', 0)}")
    
    # Calculate efficiency metrics
    if stats.get('entities_extracted', 0) > 0:
        entity_retention = (stats.get('entities_after_coalescing', 0) / stats.get('entities_extracted', 1)) * 100
        print(f"Entity retention rate: {entity_retention:.1f}%")
    
    if stats.get('relations_extracted', 0) > 0:
        relation_retention = (stats.get('relations_after_remapping', 0) / stats.get('relations_extracted', 1)) * 100
        print(f"Relation retention rate: {relation_retention:.1f}%")
    
    print("="*50)


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    try:
        # Load document
        doc = load_document(args.file, args.chunk_strategy, args.chunk_size, args.overlap)
        logging.info(f"Loaded document with {len(doc.chunks)} chunks")
        
        # Create extractor
        extractor = create_extractor(args.entity_extractor, args.llm_model)
        
        # Perform ingestion
        logging.info("Starting graph ingestion...")
        stats = ingest_document_to_neo4j(
            doc=doc,
            extractor=extractor,
            repair_relations=args.repair_relations
        )
        
        if not args.quiet:
            print(f"\n✅ Successfully ingested '{args.file}' into Neo4j knowledge graph")
            
            if args.stats:
                display_statistics(stats)
            else:
                print(f"📊 {stats.get('entities_written', 0)} entities, {stats.get('relations_written', 0)} relations written")
        
        return 0
        
    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit(main())
