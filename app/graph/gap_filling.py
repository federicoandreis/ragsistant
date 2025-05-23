"""
Gap filling and isolated node management module for graph ingestion.

This module handles:
- Connecting isolated nodes using heuristics
- Cleaning up orphaned entities
- Entity co-occurrence analysis
"""

import logging
from typing import List, Dict, Any, Optional, Set
from app.entity_extraction import extract_entities_and_relations
from ..utils.normalization import normalize_entities


def perform_gap_filling(session, doc, extractor, id_map: Dict[str, str]) -> None:
    """
    Perform heuristic gap-filling to connect isolated nodes.
    
    Args:
        session: Neo4j database session
        doc: IngestedDocument containing chunks
        extractor: Optional custom extractor function
        id_map: Mapping from original entity IDs to coalesced entity IDs
    """
    logging.info("[NEO4J][GAPFILL] Starting gap-filling process")
    
    # Map entities to their chunk index for co-occurrence analysis
    entity_chunk_map, chunk_entities = _build_entity_chunk_mapping(doc, extractor, id_map)
    
    # Find isolated nodes after writing entities/relations
    isolated_ids = _find_isolated_nodes(session)
    logging.info(f"[NEO4J][GAPFILL] Found {len(isolated_ids)} isolated nodes before gap-filling.")
    
    if not isolated_ids:
        logging.info("[NEO4J][GAPFILL] No isolated nodes found, skipping gap-filling")
        return
    
    # Connect isolated nodes using chunk co-occurrence
    connected_count = _connect_isolated_nodes(session, isolated_ids, entity_chunk_map, chunk_entities, id_map)
    logging.info(f"[NEO4J][GAPFILL] Connected {connected_count} isolated nodes via co-occurrence")


def cleanup_isolated_nodes(session) -> int:
    """
    Delete isolated nodes (entities with no relationships).
    
    Args:
        session: Neo4j database session
        
    Returns:
        Number of nodes deleted
    """
    try:
        del_result = session.run("MATCH (e:Entity) WHERE NOT (e)--() DETACH DELETE e RETURN count(e) as deleted")
        deleted_count = del_result.single()["deleted"] if del_result.peek() else 0
        logging.info(f"[NEO4J][CLEANUP] Deleted {deleted_count} isolated entity nodes.")
        return deleted_count
    except Exception as e:
        logging.warning(f"[NEO4J][CLEANUP][ERROR] Failed to delete isolated nodes: {e}")
        return 0


def _build_entity_chunk_mapping(doc, extractor, id_map: Dict[str, str]) -> tuple:
    """
    Build mapping of entities to chunks for co-occurrence analysis.
    
    Args:
        doc: IngestedDocument containing chunks
        extractor: Optional custom extractor function
        id_map: Mapping from original entity IDs to coalesced entity IDs
        
    Returns:
        Tuple of (entity_chunk_map, chunk_entities)
    """
    entity_chunk_map = {}
    chunk_entities = []
    
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
            extraction_result = extract_entities_and_relations(chunk.text, method="spacy")
            ents = [entity.model_dump() for entity in extraction_result.entities]
        
        ents = normalize_entities(ents)
        chunk_entity_ids = [e['id'] for e in ents]
        chunk_entities.append(chunk_entity_ids)
        
        # Map each entity to its chunk index
        for e in ents:
            entity_chunk_map[e['id']] = idx
    
    return entity_chunk_map, chunk_entities


def _find_isolated_nodes(session) -> List[str]:
    """
    Find entities that have no relationships.
    
    Args:
        session: Neo4j database session
        
    Returns:
        List of isolated entity IDs
    """
    iso_result = session.run("MATCH (e:Entity) WHERE NOT (e)--() RETURN e.id AS id")
    return [row['id'] for row in iso_result]


def _connect_isolated_nodes(
    session, 
    isolated_ids: List[str], 
    entity_chunk_map: Dict[str, int], 
    chunk_entities: List[List[str]], 
    id_map: Dict[str, str]
) -> int:
    """
    Connect isolated nodes to other entities using chunk co-occurrence.
    
    Args:
        session: Neo4j database session
        isolated_ids: List of isolated entity IDs
        entity_chunk_map: Mapping of entity ID to chunk index
        chunk_entities: List of entity IDs per chunk
        id_map: Mapping from original to coalesced entity IDs
        
    Returns:
        Number of connections made
    """
    connected_count = 0
    
    for iso_id in isolated_ids:
        # Find chunk where this isolated entity appears
        chunk_idx = entity_chunk_map.get(iso_id)
        if chunk_idx is None:
            logging.debug(f"[NEO4J][GAPFILL] Isolated entity {iso_id} not found in chunk mapping")
            continue
        
        # Find other entities in the same chunk
        co_occurring_ids = [
            eid for eid in chunk_entities[chunk_idx] 
            if eid != iso_id and eid in id_map.values()
        ]
        
        if not co_occurring_ids:
            logging.debug(f"[NEO4J][GAPFILL] No co-occurring entities found for {iso_id}")
            continue
        
        # Connect to the first co-occurring entity (minimalist approach)
        target_id = co_occurring_ids[0]
        if _create_mentioned_with_relation(session, iso_id, target_id):
            connected_count += 1
            logging.debug(f"[NEO4J][GAPFILL] Linked isolated {iso_id} to {target_id} via MENTIONED_WITH")
            # Only link to one entity to avoid over-connecting
            break
    
    return connected_count


def _create_mentioned_with_relation(session, source_id: str, target_id: str) -> bool:
    """
    Create a MENTIONED_WITH relationship between two entities.
    
    Args:
        session: Neo4j database session
        source_id: Source entity ID
        target_id: Target entity ID
        
    Returns:
        True if relationship was created successfully
    """
    cypher = (
        "MATCH (e1:Entity {id: $src}) "
        "MATCH (e2:Entity {id: $tgt}) "
        "MERGE (e1)-[r:MENTIONED_WITH]->(e2)"
    )
    params = {"src": source_id, "tgt": target_id}
    
    try:
        session.run(cypher, **params)
        return True
    except Exception as e:
        logging.warning(f"[NEO4J][GAPFILL][ERROR] Failed to create relation {source_id} -> {target_id}: {e}")
        return False


def get_isolation_statistics(session) -> Dict[str, Any]:
    """
    Get statistics about isolated nodes in the graph.
    
    Args:
        session: Neo4j database session
        
    Returns:
        Dictionary with isolation statistics
    """
    try:
        # Count isolated nodes
        isolated_query = "MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS isolated_count"
        isolated_result = session.run(isolated_query)
        isolated_count = isolated_result.single()["isolated_count"]
        
        # Count total nodes
        total_query = "MATCH (e:Entity) RETURN count(e) AS total_count"
        total_result = session.run(total_query)
        total_count = total_result.single()["total_count"]
        
        # Calculate isolation percentage
        isolation_percentage = (isolated_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "isolated_nodes": isolated_count,
            "total_nodes": total_count,
            "connected_nodes": total_count - isolated_count,
            "isolation_percentage": round(isolation_percentage, 2)
        }
        
    except Exception as e:
        logging.error(f"[NEO4J][GAPFILL] Failed to get isolation statistics: {e}")
        return {"error": str(e)}


def analyze_connectivity_patterns(session) -> Dict[str, Any]:
    """
    Analyze connectivity patterns in the graph.
    
    Args:
        session: Neo4j database session
        
    Returns:
        Dictionary with connectivity analysis
    """
    try:
        # Node degree distribution
        degree_query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) AS degree
        RETURN 
            min(degree) AS min_degree,
            max(degree) AS max_degree,
            avg(degree) AS avg_degree,
            count(CASE WHEN degree = 0 THEN 1 END) AS isolated_count,
            count(CASE WHEN degree = 1 THEN 1 END) AS leaf_count,
            count(CASE WHEN degree > 5 THEN 1 END) AS hub_count
        """
        
        result = session.run(degree_query)
        stats = result.single()
        
        if stats:
            return {
                "min_degree": stats["min_degree"],
                "max_degree": stats["max_degree"],
                "avg_degree": round(stats["avg_degree"], 2),
                "isolated_nodes": stats["isolated_count"],
                "leaf_nodes": stats["leaf_count"],
                "hub_nodes": stats["hub_count"]
            }
        else:
            return {"error": "No data found"}
            
    except Exception as e:
        logging.error(f"[NEO4J][GAPFILL] Failed to analyze connectivity: {e}")
        return {"error": str(e)}
