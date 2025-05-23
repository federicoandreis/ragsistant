"""
Community detection and analysis module for graph ingestion.

This module handles:
- Neo4j Graph Data Science (GDS) community detection (if available)
- Fallback to standard Cypher-based community detection
- Community summary generation using LLM
- Graph projection and cleanup
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Set
import ollama
from app.config import get_config


def check_gds_availability(session) -> bool:
    """
    Check if Neo4j Graph Data Science library is available.
    
    Args:
        session: Neo4j database session
        
    Returns:
        True if GDS is available, False otherwise
    """
    try:
        session.run("CALL gds.version() YIELD gdsVersion")
        return True
    except Exception:
        return False


def perform_community_detection(session, config) -> None:
    """
    Perform community detection using GDS algorithms (if available) or fallback method.
    
    Args:
        session: Neo4j database session
        config: Application configuration object
    """
    # Check if GDS is available
    if check_gds_availability(session):
        logging.info("[NEO4J][COMMUNITY] Using Neo4j GDS for community detection")
        _perform_gds_community_detection(session, config)
    else:
        logging.info("[NEO4J][COMMUNITY] GDS not available, using fallback method")
        _perform_fallback_community_detection(session, config)


def _perform_gds_community_detection(session, config) -> None:
    """
    Perform community detection using GDS algorithms (Leiden/Louvain).
    
    Args:
        session: Neo4j database session
        config: Application configuration object
    """
    try:
        # Check if graph exists and drop it
        exists_res = session.run("CALL gds.graph.exists('entityGraph') YIELD exists")
        exists = exists_res.single()["exists"] if exists_res.peek() else False
        if exists:
            session.run("CALL gds.graph.drop('entityGraph') YIELD graphName")
            logging.info("[NEO4J][COMMUNITY] Dropped existing 'entityGraph' before projection.")
        
        # Dynamically project only existing relationship types for community detection
        DESIRED_REL_TYPES = [
            'RELATED_TO', 'WORKS_AS_PERSON_AT', 'COLLABORATED_WITH', 
            'PARTICIPATED_IN', 'FUNDED_BY', 'WORKED_AT', 'DECEASED'
        ]
        
        # Query Neo4j for existing relationship types
        reltype_query = "CALL db.relationshipTypes()"
        reltype_res = session.run(reltype_query)
        existing_types = set(row["relationshipType"] for row in reltype_res)
        rel_types_to_project = [rel for rel in DESIRED_REL_TYPES if rel in existing_types]
        
        if not rel_types_to_project:
            logging.warning("[NEO4J][COMMUNITY] No desired relationship types found in DB, falling back to projecting all.")
            rel_types_cypher = "'*'"
        else:
            # For Leiden, relationships must be undirected and use Cypher map syntax
            rel_types_cypher = '{' + ', '.join([
                f"{rel}: {{type: '{rel}', orientation: 'UNDIRECTED'}}" for rel in rel_types_to_project
            ]) + '}'
            logging.info(f"[NEO4J][COMMUNITY] Projecting relationships (UNDIRECTED for Leiden): {rel_types_to_project}")
            session.run(f"CALL gds.graph.project('entityGraph', 'Entity', {rel_types_cypher})")
        
        # Use Leiden for more robust communities; fallback to Louvain if needed
        try:
            result = session.run("CALL gds.leiden.write('entityGraph', { writeProperty: 'community_id' }) YIELD communityCount")
            community_count = result.single()["communityCount"] if result.peek() else 0
            logging.info(f"[NEO4J][COMMUNITY] Leiden detected {community_count} communities.")
        except Exception as e:
            logging.warning(f"[NEO4J][COMMUNITY][Leiden failed, falling back to Louvain] {e}")
            result = session.run("CALL gds.louvain.write('entityGraph', { writeProperty: 'community_id' }) YIELD communityCount")
            community_count = result.single()["communityCount"] if result.peek() else 0
            logging.info(f"[NEO4J][COMMUNITY] Louvain detected {community_count} communities.")
        
        # Generate community summaries
        generate_community_summaries(session, config)
        
    except Exception as e:
        logging.error(f"[NEO4J][COMMUNITY] GDS community detection failed: {e}")
        logging.info("[NEO4J][COMMUNITY] Falling back to standard method")
        _perform_fallback_community_detection(session, config)


def _perform_fallback_community_detection(session, config) -> None:
    """
    Perform basic community detection using connected components analysis.
    This is a fallback when GDS is not available.
    
    Args:
        session: Neo4j database session
        config: Application configuration object
    """
    try:
        logging.info("[NEO4J][COMMUNITY] Starting fallback community detection (no GDS)")
        
        # Clear any existing community assignments
        session.run("MATCH (e:Entity) REMOVE e.community_id")
        
        # Find connected components using iterative approach
        communities = _find_connected_components(session)
        
        # Assign community IDs
        _assign_community_ids(session, communities)
        
        logging.info(f"[NEO4J][COMMUNITY] Detected {len(communities)} communities using fallback method")
        
        # Generate community summaries
        generate_community_summaries(session, config)
        
    except Exception as e:
        logging.error(f"[NEO4J][COMMUNITY] Fallback community detection failed: {e}")


def _find_connected_components(session) -> List[Set[str]]:
    """
    Find connected components in the entity graph using standard Cypher.
    
    Args:
        session: Neo4j database session
        
    Returns:
        List of sets, each containing entity IDs in the same component
    """
    # Get all entities
    entities_result = session.run("MATCH (e:Entity) RETURN e.id AS entity_id")
    all_entities = {row["entity_id"] for row in entities_result}
    
    if not all_entities:
        return []
    
    visited = set()
    communities = []
    
    for entity_id in all_entities:
        if entity_id not in visited:
            # Find all entities connected to this one
            component = _find_component_from_entity(session, entity_id, visited)
            if component:
                communities.append(component)
                visited.update(component)
    
    return communities


def _find_component_from_entity(session, start_entity: str, global_visited: Set[str]) -> Set[str]:
    """
    Find all entities connected to the start entity using BFS.
    
    Args:
        session: Neo4j database session
        start_entity: Starting entity ID
        global_visited: Set of globally visited entities
        
    Returns:
        Set of entity IDs in the same connected component
    """
    if start_entity in global_visited:
        return set()
    
    component = set()
    queue = [start_entity]
    local_visited = set()
    
    while queue:
        current = queue.pop(0)
        if current in local_visited:
            continue
            
        local_visited.add(current)
        component.add(current)
        
        # Find all directly connected entities (both directions)
        neighbors_query = """
        MATCH (e1:Entity {id: $entity_id})-[r]-(e2:Entity)
        WHERE e2.id <> $entity_id
        RETURN DISTINCT e2.id AS neighbor_id
        """
        
        neighbors_result = session.run(neighbors_query, {"entity_id": current})
        neighbors = [row["neighbor_id"] for row in neighbors_result]
        
        for neighbor in neighbors:
            if neighbor not in local_visited and neighbor not in global_visited:
                queue.append(neighbor)
    
    return component


def _assign_community_ids(session, communities: List[Set[str]]) -> None:
    """
    Assign community IDs to entities based on connected components.
    
    Args:
        session: Neo4j database session
        communities: List of sets containing entity IDs for each community
    """
    for i, community in enumerate(communities):
        community_id = i + 1  # Start from 1
        
        # Update all entities in this community
        for entity_id in community:
            session.run(
                "MATCH (e:Entity {id: $entity_id}) SET e.community_id = $community_id",
                {"entity_id": entity_id, "community_id": community_id}
            )
        
        logging.debug(f"[NEO4J][COMMUNITY] Assigned community {community_id} to {len(community)} entities")


def generate_community_summaries(session, config) -> None:
    """
    Generate summaries for each community using LLM.
    
    Args:
        session: Neo4j database session
        config: Application configuration object
    """
    comm_result = session.run("MATCH (e:Entity) RETURN DISTINCT e.community_id AS cid")
    community_ids = [row["cid"] for row in comm_result if row["cid"] is not None]
    
    logging.info(f"[NEO4J][COMMUNITY] Generating summaries for {len(community_ids)} communities")
    
    for cid in community_ids:
        try:
            # Skip singleton communities (only one entity)
            size_res = session.run("MATCH (e:Entity {community_id: $cid}) RETURN count(e) as size", {"cid": cid})
            size = size_res.single()["size"]
            if size <= 1:
                logging.debug(f"[NEO4J][COMMUNITY] Skipping singleton community_id={cid}")
                continue
            
            # Get community entities
            ent_res = session.run(
                "MATCH (e:Entity {community_id: $cid}) RETURN e.name, e.id, e.label, e.aliases", 
                {"cid": cid}
            )
            entity_data = list(ent_res)
            entity_names = [row["e.name"] for row in entity_data if row["e.name"] and row["e.name"].strip()]
            entity_props = [
                {
                    "name": row["e.name"], 
                    "id": row["e.id"], 
                    "label": row["e.label"], 
                    "aliases": row["e.aliases"]
                }
                for row in entity_data
            ]
            
            # Gather all relationships in the community
            rel_res = session.run(
                """
                MATCH (e1:Entity {community_id: $cid})-[r]->(e2:Entity {community_id: $cid}) 
                RETURN e1.name AS src, type(r) AS rel, e2.name AS tgt, r as rel_props
                """,
                {"cid": cid}
            )
            relations = [
                {
                    "source": row['src'], 
                    "type": row['rel'], 
                    "target": row['tgt'], 
                    "properties": dict(row['rel_props']) if row['rel_props'] else {}
                }
                for row in rel_res
            ]
            
            # Generate community summary using LLM
            summary = _generate_llm_summary(entity_props, relations, config)
            
            # Store community summary
            cypher = "MERGE (c:Community {community_id: $cid}) SET c.summary = $summary, c.entity_names = $entity_names"
            params = {"cid": cid, "summary": summary, "entity_names": entity_names}
            session.run(cypher, **params)
            logging.info(f"[NEO4J][COMMUNITY] Generated summary for community {cid} ({size} entities)")
            
        except Exception as e:
            logging.error(f"[NEO4J][COMMUNITY][ERROR] Failed to generate summary for community {cid}: {e}")


def _generate_llm_summary(entity_props: List[Dict], relations: List[Dict], config) -> str:
    """
    Generate a community summary using LLM.
    
    Args:
        entity_props: List of entity property dictionaries
        relations: List of relationship dictionaries
        config: Application configuration object
        
    Returns:
        Generated summary text
    """
    summary_prompt = (
        "Write ONLY a concise, specific summary of the following community of entities and their relationships. "
        "Use entity names, types, and relationship types. Include notable properties or aliases if relevant. "
        "Do NOT include any introductory phrases, preambles, explanations, or boilerplate. Output only the summary text.\n"
        f"Entities: {json.dumps(entity_props, ensure_ascii=False)}\n"
        f"Relationships: {json.dumps(relations, ensure_ascii=False) if relations else 'None'}"
    )
    
    try:
        summary_resp = ollama.chat(
            model=config.models.synthesis_model, 
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = summary_resp.get("message", {}).get("content", "").strip()
        
        # Remove generic/boilerplate lead-ins
        summary = _clean_summary_text(summary)
        
        return summary if summary else "Community of related entities"
        
    except Exception as e:
        logging.warning(f"[NEO4J][COMMUNITY] LLM summary generation failed: {e}")
        return f"Community of {len(entity_props)} entities"


def _clean_summary_text(summary: str) -> str:
    """
    Clean summary text by removing boilerplate phrases.
    
    Args:
        summary: Raw summary text from LLM
        
    Returns:
        Cleaned summary text
    """
    bad_starts = [
        r"(?i)^here is.*summary.*?:?", r"(?i)^summary:?", r"(?i)^the following.*?:?", 
        r"(?i)^entities and their relationships:?", r"(?i)^as an ai.*", 
        r"(?i)^i don't have any information.*", r"(?i)^no information.*", 
        r"(?i)^no context.*", r"(?i)^it seems.*", r"(?i)^in summary:?", 
        r"(?i)^this is a summary:?", r"(?i)^there are no claims.*", 
        r"(?i)^no claims found.*"
    ]
    
    for bad_start in bad_starts:
        summary = re.sub(bad_start, '', summary).strip(" .:\n")
    
    return summary


def cleanup_graph_projection(session) -> None:
    """
    Clean up the graph projection used for community detection.
    
    Args:
        session: Neo4j database session
    """
    if check_gds_availability(session):
        try:
            exists_res = session.run("CALL gds.graph.exists('entityGraph') YIELD exists")
            exists = exists_res.single()["exists"] if exists_res.peek() else False
            if exists:
                session.run("CALL gds.graph.drop('entityGraph') YIELD graphName")
                logging.info("[NEO4J][COMMUNITY] Dropped 'entityGraph' at final cleanup.")
        except Exception as e:
            logging.debug(f"[NEO4J][COMMUNITY] Error cleaning up 'entityGraph': {e}")
    else:
        logging.debug("[NEO4J][COMMUNITY] Cleanup called (no-op for fallback version)")


def get_community_statistics(session) -> Dict[str, Any]:
    """
    Get statistics about detected communities.
    
    Args:
        session: Neo4j database session
        
    Returns:
        Dictionary with community statistics
    """
    try:
        # Count communities and their sizes
        stats_query = """
        MATCH (e:Entity) 
        WHERE e.community_id IS NOT NULL
        WITH e.community_id AS cid, count(e) AS size
        RETURN count(cid) AS total_communities, 
               avg(size) AS avg_community_size,
               min(size) AS min_community_size,
               max(size) AS max_community_size
        """
        result = session.run(stats_query)
        stats = result.single()
        
        if stats:
            return {
                "total_communities": stats["total_communities"],
                "avg_community_size": round(stats["avg_community_size"], 2),
                "min_community_size": stats["min_community_size"],
                "max_community_size": stats["max_community_size"],
                "gds_available": check_gds_availability(session)
            }
        else:
            return {"total_communities": 0, "gds_available": check_gds_availability(session)}
            
    except Exception as e:
        logging.error(f"[NEO4J][COMMUNITY] Failed to get community statistics: {e}")
        return {"error": str(e), "gds_available": False}
