"""
Core graph ingestion orchestration module.

This module handles:
- Main document ingestion workflow
- Entity and relation processing coordination
- Neo4j database operations
- Integration of all graph processing components
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from app.models import IngestedDocument
from app.entity_extraction import extract_entities_and_relations
from ..utils.normalization import normalize_entities, normalize_relations
from app.db.connections import get_db_manager
from app.config import get_config
from .relation_repair import sanitize_relation_type
from .entity_coalescing import coalesce_entities, remap_relations_with_coalesced_entities
from .community_detection import perform_community_detection, cleanup_graph_projection
from .gap_filling import perform_gap_filling, cleanup_isolated_nodes


def ingest_document_to_neo4j(
    doc: IngestedDocument, 
    chunk_embs: Optional[List] = None, 
    extractor: Optional[Callable] = None, 
    chunk_anchor: bool = True, 
    repair_relations: bool = False
) -> Dict[str, Any]:
    """
    Ingest a document to Neo4j using centralized database connections and entity extraction.
    
    Args:
        doc: IngestedDocument to process
        chunk_embs: Optional chunk embeddings (not used in this version)
        extractor: Optional custom extractor function
        chunk_anchor: Whether to anchor chunks (not used in this version)
        repair_relations: Whether to attempt relation repair
        
    Returns:
        Dictionary with ingestion statistics and results
    """
    config = get_config()
    db_manager = get_db_manager()
    
    ingestion_stats = {
        "document": doc.metadata.filename,
        "chunks_processed": 0,
        "entities_extracted": 0,
        "relations_extracted": 0,
        "entities_after_coalescing": 0,
        "relations_after_remapping": 0,
        "entities_written": 0,
        "relations_written": 0
    }
    
    with db_manager.neo4j_session() as session:
        # Extract entities and relations from all chunks
        all_entities, all_relations = _extract_entities_and_relations_from_document(
            doc, extractor, ingestion_stats
        )
        
        if not all_entities:
            logging.warning(f"[NEO4J][WARNING] No entities extracted from document '{doc.metadata.filename}'. Skipping Neo4j ingestion.")
            return ingestion_stats
        
        if not all_relations:
            logging.warning(f"[NEO4J][WARNING] No relations extracted from document '{doc.metadata.filename}'.")
        
        logging.info(f"[NEO4J] Extracted {len(all_entities)} entities and {len(all_relations)} relations from {len(doc.chunks)} chunks.")
        
        # Coalesce entities globally
        unique_entities, id_map = coalesce_entities(all_entities)
        ingestion_stats["entities_after_coalescing"] = len(unique_entities)
        logging.info(f"[NEO4J] After coalescing: {len(unique_entities)} unique entities.")
        
        # Remap relations to use coalesced entity ids
        remapped_relations = remap_relations_with_coalesced_entities(all_relations, id_map)
        ingestion_stats["relations_after_remapping"] = len(remapped_relations)
        logging.info(f"[NEO4J] {len(remapped_relations)} relations after remapping.")
        
        # Write entities and relations to Neo4j
        entities_written = _write_entities_to_neo4j(session, unique_entities)
        relations_written = _write_relations_to_neo4j(session, remapped_relations)
        
        ingestion_stats["entities_written"] = entities_written
        ingestion_stats["relations_written"] = relations_written
        
        logging.info(f"[NEO4J] Written {entities_written} entities and {relations_written} relations to Neo4j.")
        
        # Perform advanced graph analysis
        _perform_advanced_graph_analysis(session, doc, extractor, id_map, config)
        
    return ingestion_stats


def _extract_entities_and_relations_from_document(
    doc: IngestedDocument, 
    extractor: Optional[Callable], 
    stats: Dict[str, Any]
) -> tuple:
    """
    Extract entities and relations from all chunks in a document.
    
    Args:
        doc: IngestedDocument to process
        extractor: Optional custom extractor function
        stats: Statistics dictionary to update
        
    Returns:
        Tuple of (all_entities, all_relations)
    """
    all_entities = []
    all_relations = []
    
    for idx, chunk in enumerate(doc.chunks):
        if extractor:
            result = extractor(chunk.text)
            if isinstance(result, (tuple, list)) and len(result) == 3:
                ents, rels, summary = result
            elif isinstance(result, (tuple, list)) and len(result) == 2:
                ents, rels = result
                summary = None
            else:
                raise ValueError("Extractor must return 2 or 3 values (entities, relations[, summary])")
        else:
            # Use new entity extraction module
            extraction_result = extract_entities_and_relations(chunk.text, method="spacy")
            ents = [entity.model_dump() for entity in extraction_result.entities]
            rels = [(rel.source, rel.type, rel.target) for rel in extraction_result.relations]
            summary = extraction_result.summary
        
        logging.info(f"[EXTRACTOR][CHUNK {idx}] Entities: {len(ents)}")
        logging.info(f"[EXTRACTOR][CHUNK {idx}] Relations: {len(rels)}")
        
        # Normalize entities and relations
        ents = normalize_entities(ents)
        rels = normalize_relations(rels)
        
        # Filter out generic relations if more specific ones exist
        rels = _filter_generic_relations(rels)
        
        all_entities.extend(ents)
        all_relations.extend(rels)
        
        stats["chunks_processed"] += 1
    
    stats["entities_extracted"] = len(all_entities)
    stats["relations_extracted"] = len(all_relations)
    
    return all_entities, all_relations


def _filter_generic_relations(relations: List[tuple]) -> List[tuple]:
    """
    Filter out generic 'MENTIONED_WITH' or 'RELATED_TO' if a more specific relation exists between the same pair.
    
    Args:
        relations: List of (source, type, target) relation tuples
        
    Returns:
        Filtered list of relations
    """
    specific_relations = set()
    filtered_relations = []
    
    # First pass: identify specific relations
    for (src, typ, tgt) in relations:
        if typ not in ("MENTIONED_WITH", "RELATED_TO"):
            specific_relations.add((src, tgt))
    
    # Second pass: filter out generic relations where specific ones exist
    for (src, typ, tgt) in relations:
        if typ in ("MENTIONED_WITH", "RELATED_TO") and (src, tgt) in specific_relations:
            continue  # Prefer specific relation
        filtered_relations.append((src, typ, tgt))
    
    return filtered_relations


def _write_entities_to_neo4j(session, entities: List[Dict[str, Any]]) -> int:
    """
    Write entities to Neo4j database.
    
    Args:
        session: Neo4j database session
        entities: List of entity dictionaries
        
    Returns:
        Number of entities successfully written
    """
    written_count = 0
    
    for ent in entities:
        cypher = "MERGE (e:Entity {id: $id}) SET e.label = $label, e.name = $name, e.aliases = $aliases"
        params = {
            "id": ent["id"], 
            "label": ent["label"], 
            "name": ent["name"], 
            "aliases": ent.get("aliases", [])
        }
        
        try:
            result = session.run(cypher, **params)
            written_count += 1
            logging.debug(f"[NEO4J][ENTITY] Created/updated entity: {ent['id']}")
        except Exception as e:
            logging.error(f"[NEO4J][ENTITY][ERROR] {cypher} | Params: {params} | Exception: {e}")
    
    return written_count


def _write_relations_to_neo4j(session, relations: List[tuple]) -> int:
    """
    Write relations to Neo4j database.
    
    Args:
        session: Neo4j database session
        relations: List of (source, type, target) relation tuples
        
    Returns:
        Number of relations successfully written
    """
    written_count = 0
    
    for src, typ, tgt in relations:
        sanitized_type = sanitize_relation_type(typ)
        cypher = (
            "MATCH (e1:Entity {id: $src}) "
            "MATCH (e2:Entity {id: $tgt}) "
            f"MERGE (e1)-[r:{sanitized_type.upper()}]->(e2)"
        )
        params = {"src": src, "tgt": tgt}
        
        try:
            result = session.run(cypher, **params)
            written_count += 1
            logging.debug(f"[NEO4J][RELATION] Created relation: {src} -> {tgt}")
        except Exception as e:
            logging.error(f"[NEO4J][RELATION][ERROR] {cypher} | Params: {params} | Exception: {e}")
    
    return written_count


def _perform_advanced_graph_analysis(session, doc, extractor, id_map: Dict[str, str], config) -> None:
    """
    Perform advanced graph analysis including community detection and gap filling.
    
    Args:
        session: Neo4j database session
        doc: IngestedDocument
        extractor: Optional custom extractor function
        id_map: Entity ID mapping from coalescing
        config: Application configuration
    """
    try:
        # Community detection and analysis
        logging.info("[NEO4J] Starting community detection...")
        perform_community_detection(session, config)
        
        # Gap filling for isolated nodes
        logging.info("[NEO4J] Starting gap filling...")
        perform_gap_filling(session, doc, extractor, id_map)
        
        # Cleanup isolated nodes
        logging.info("[NEO4J] Cleaning up isolated nodes...")
        deleted_count = cleanup_isolated_nodes(session)
        
        # Clean up graph projections
        cleanup_graph_projection(session)
        
        logging.info("[NEO4J] Advanced graph analysis completed")
        
    except Exception as e:
        logging.error(f"[NEO4J] Advanced graph analysis failed: {e}")


def get_ingestion_statistics(session) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the ingested graph.
    
    Args:
        session: Neo4j database session
        
    Returns:
        Dictionary with ingestion statistics
    """
    try:
        stats = {}
        
        # Basic counts
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]
        relation_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        
        stats["entities"] = entity_count
        stats["relations"] = relation_count
        
        # Relationship type distribution
        rel_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
        """
        rel_types_result = session.run(rel_types_query)
        stats["relation_types"] = {row["rel_type"]: row["count"] for row in rel_types_result}
        
        # Entity label distribution
        entity_labels_query = """
        MATCH (e:Entity)
        RETURN e.label AS label, count(e) AS count
        ORDER BY count DESC
        """
        entity_labels_result = session.run(entity_labels_query)
        stats["entity_labels"] = {row["label"]: row["count"] for row in entity_labels_result}
        
        return stats
        
    except Exception as e:
        logging.error(f"[NEO4J] Failed to get ingestion statistics: {e}")
        return {"error": str(e)}


def validate_graph_integrity(session) -> Dict[str, Any]:
    """
    Validate the integrity of the ingested graph.
    
    Args:
        session: Neo4j database session
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": True,
        "issues": [],
        "warnings": []
    }
    
    try:
        # Check for entities without names
        no_name_query = "MATCH (e:Entity) WHERE e.name IS NULL OR e.name = '' RETURN count(e) AS count"
        no_name_count = session.run(no_name_query).single()["count"]
        if no_name_count > 0:
            validation_results["warnings"].append(f"{no_name_count} entities without names")
        
        # Check for duplicate entity IDs (shouldn't happen but good to verify)
        duplicate_ids_query = """
        MATCH (e:Entity)
        WITH e.id AS id, count(e) AS count
        WHERE count > 1
        RETURN count(*) AS duplicate_count
        """
        duplicate_count = session.run(duplicate_ids_query).single()["duplicate_count"]
        if duplicate_count > 0:
            validation_results["valid"] = False
            validation_results["issues"].append(f"{duplicate_count} duplicate entity IDs found")
        
        # Check for self-referencing relations
        self_ref_query = "MATCH (e)-[r]->(e) RETURN count(r) AS count"
        self_ref_count = session.run(self_ref_query).single()["count"]
        if self_ref_count > 0:
            validation_results["warnings"].append(f"{self_ref_count} self-referencing relations")
        
        return validation_results
        
    except Exception as e:
        logging.error(f"[NEO4J] Graph validation failed: {e}")
        return {"valid": False, "error": str(e)}
