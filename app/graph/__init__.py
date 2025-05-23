"""
Graph ingestion module for RAG system.

This module provides a modular, well-structured approach to ingesting documents
into a Neo4j knowledge graph with entity extraction, relation processing,
community detection, and gap filling.

Public API:
    ingest_document_to_neo4j: Main ingestion function
    CLI functions: Command-line interface utilities
    Statistics and validation functions
"""

# Core ingestion functionality
from .ingestion_core import (
    ingest_document_to_neo4j,
    get_ingestion_statistics,
    validate_graph_integrity
)

# CLI functionality
from .cli import main as cli_main, create_cli_parser, setup_logging

# Individual component functions (for advanced usage)
from .relation_repair import (
    sanitize_relation_type,
    is_valid_entity_id,
    repair_malformed_relations,
    process_relations_with_repair
)

from .entity_coalescing import (
    coalesce_entities,
    remap_relations_with_coalesced_entities,
    validate_entity_consistency
)

from .community_detection import (
    perform_community_detection,
    generate_community_summaries,
    get_community_statistics,
    cleanup_graph_projection
)

from .gap_filling import (
    perform_gap_filling,
    cleanup_isolated_nodes,
    get_isolation_statistics,
    analyze_connectivity_patterns
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "RAG System Team"

# Public API exports
__all__ = [
    # Core functions
    "ingest_document_to_neo4j",
    "get_ingestion_statistics", 
    "validate_graph_integrity",
    
    # CLI functions
    "cli_main",
    "create_cli_parser",
    "setup_logging",
    
    # Component functions
    "sanitize_relation_type",
    "is_valid_entity_id",
    "repair_malformed_relations",
    "process_relations_with_repair",
    "coalesce_entities",
    "remap_relations_with_coalesced_entities",
    "validate_entity_consistency",
    "perform_community_detection",
    "generate_community_summaries",
    "get_community_statistics",
    "cleanup_graph_projection",
    "perform_gap_filling",
    "cleanup_isolated_nodes",
    "get_isolation_statistics",
    "analyze_connectivity_patterns",
]


def get_module_info() -> dict:
    """
    Get information about the graph module.
    
    Returns:
        Dictionary with module metadata
    """
    return {
        "name": "graph",
        "version": __version__,
        "description": "Modular graph ingestion system for Neo4j knowledge graphs",
        "components": [
            "relation_repair",
            "entity_coalescing", 
            "community_detection",
            "gap_filling",
            "ingestion_core",
            "cli"
        ],
        "main_functions": [
            "ingest_document_to_neo4j",
            "cli_main"
        ]
    }


# Backward compatibility - maintain the original function signature
# This allows existing code to continue working without changes
def ingest_document_to_neo4j_legacy(*args, **kwargs):
    """
    Legacy wrapper for backward compatibility.
    Delegates to the new modular implementation.
    """
    return ingest_document_to_neo4j(*args, **kwargs)
