import logging
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any
from app.config import get_config

config = get_config()

def retrieve_entity_context(entity_name: str, hops: int = 1, max_relationships: int = 100) -> Dict[str, Any]:
    """
    Given an entity name, retrieve the entity and its relationships up to N hops from Neo4j.
    Returns a dict with entity info and related nodes/edges.
    max_relationships limits the number of relationships returned.
    """
    logging.info(f"[GRAPH RETRIEVAL] Searching for entity name: '{entity_name}' (hops={hops}, max_relationships={max_relationships})")
    driver = GraphDatabase.driver(
        config.database.neo4j_uri, 
        auth=basic_auth(config.database.neo4j_username, config.database.neo4j_password)
    )
    result = {}
    with driver.session() as session:
        # Find the entity node by name (case-insensitive)
        node_res = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower($name)
            RETURN e LIMIT 1
            """,
            name=entity_name
        )
        node = node_res.single()
        if not node:
            logging.warning(f"[GRAPH RETRIEVAL] No entity found for name: '{entity_name}'")
            result["entity"] = None
            result["relationships"] = []
            driver.close()
            return result
        entity = dict(node["e"])
        logging.info(f"[GRAPH RETRIEVAL] Found entity: {entity}")
        result["entity"] = entity
        # Retrieve relationships up to N hops (must interpolate hops directly in Cypher)
        cypher = f"""
            MATCH (e:Entity {{name: $name}})-[r*1..{hops}]-(n)
            WITH e, n, r
            UNWIND r as rel
            RETURN type(rel) as rel_type, startNode(rel) as from_node, endNode(rel) as to_node
            LIMIT $max_relationships
        """
        rels_res = session.run(
            cypher,
            name=entity_name,
            max_relationships=max_relationships
        )
        relationships = []
        for row in rels_res:
            relationships.append({
                "rel_type": row["rel_type"],
                "from": dict(row["from_node"]),
                "to": dict(row["to_node"])
            })
        logging.info(f"[GRAPH RETRIEVAL] Retrieved {len(relationships)} relationships for '{entity_name}'")
        result["relationships"] = relationships
    driver.close()
    return result

def retrieve_relevant_paths(
    entity_name: str,
    max_paths: int = 5,
    max_length: int = 3,
    community_ids: List[int] = None,
    min_score: float = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant paths (up to max_length hops) from the entity, prioritizing paths within the specified communities.
    Each path is a list of nodes and relationships.
    Optionally restrict to certain communities (by community_id property on nodes).
    """
    driver = GraphDatabase.driver(
        config.database.neo4j_uri, 
        auth=basic_auth(config.database.neo4j_username, config.database.neo4j_password)
    )
    paths = []
    with driver.session() as session:
        # Build community filter if needed
        community_filter = ""
        if community_ids:
            community_filter = " AND all(n in nodes(p) WHERE n.community_id IN $community_ids)"
        # Cypher to get paths
        cypher = f"""
            MATCH p = (e:Entity)-[*1..{max_length}]-(n)
            WHERE toLower(e.name) = toLower($name)
            {community_filter}
            RETURN p
            LIMIT $max_paths
        """
        res = session.run(
            cypher,
            name=entity_name,
            community_ids=community_ids or [],
            max_paths=max_paths
        )
        for row in res:
            path = row["p"]
            nodes = [dict(node) for node in path.nodes]
            rels = [
                {
                    "type": rel.type,
                    "start": dict(rel.start_node),
                    "end": dict(rel.end_node)
                }
                for rel in path.relationships
            ]
            paths.append({"nodes": nodes, "relationships": rels})
    driver.close()
    return paths

def get_top_communities_for_query(entity_name: str, top_n: int = 3) -> list:
    """
    Retrieve the top-N most relevant community IDs for a given entity (by overlap, proximity, or semantic similarity).
    For now, uses the community_id property of the entity and its neighbors.
    """
    driver = GraphDatabase.driver(
        config.database.neo4j_uri, 
        auth=basic_auth(config.database.neo4j_username, config.database.neo4j_password)
    )
    community_ids = set()
    with driver.session() as session:
        # Get the community of the main entity
        res = session.run(
            "MATCH (e:Entity) WHERE toLower(e.name) = toLower($name) RETURN e.community_id AS cid LIMIT 1",
            name=entity_name
        )
        row = res.single()
        if row and row["cid"] is not None:
            community_ids.add(row["cid"])
        # Get communities of direct neighbors
        res2 = session.run(
            """
            MATCH (e:Entity)-[*1..2]-(n:Entity)
            WHERE toLower(e.name) = toLower($name)
            RETURN n.community_id AS cid
            LIMIT 20
            """,
            name=entity_name
        )
        for r in res2:
            if r["cid"] is not None:
                community_ids.add(r["cid"])
    driver.close()
    # Return top N (arbitrary order for now, could be ranked by frequency)
    return list(community_ids)[:top_n]
