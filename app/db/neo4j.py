"""
Neo4j connector for graph storage and retrieval operations.
Provides high-level interface for entity and relationship management.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from app.db.connections import get_db_manager
from app.config import get_config


@dataclass
class Neo4jEntity:
    """Represents an entity node in Neo4j."""
    id: Optional[str]
    name: str
    entity_type: str
    properties: Dict[str, Any]
    labels: List[str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = ["Entity"]


@dataclass
class Neo4jRelationship:
    """Represents a relationship in Neo4j."""
    id: Optional[str]
    from_entity: str  # Entity name or ID
    to_entity: str    # Entity name or ID
    relationship_type: str
    properties: Dict[str, Any]
    direction: str = "OUTGOING"  # OUTGOING, INCOMING, BOTH


@dataclass
class Neo4jPath:
    """Represents a path through the graph."""
    nodes: List[Neo4jEntity]
    relationships: List[Neo4jRelationship]
    length: int
    
    def __post_init__(self):
        self.length = len(self.relationships)


class Neo4jConnector:
    """High-level interface for Neo4j operations."""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()
    
    def create_entity(self, entity: Neo4jEntity) -> bool:
        """
        Create or update an entity in Neo4j.
        
        Args:
            entity: Neo4jEntity object to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.db_manager.neo4j_session() as session:
                # Create labels string
                labels_str = ":".join(entity.labels)
                
                # Prepare properties
                props = {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "created_at": datetime.utcnow().isoformat(),
                    **entity.properties
                }
                
                # Use MERGE to avoid duplicates
                query = f"""
                MERGE (e:{labels_str} {{name: $name}})
                SET e += $properties
                RETURN e.name as name
                """
                
                result = session.run(query, name=entity.name, properties=props)
                record = result.single()
                
                if record:
                    logging.info(f"Created/updated entity: {entity.name}")
                    return True
                else:
                    logging.warning(f"Failed to create entity: {entity.name}")
                    return False
                    
        except Exception as e:
            logging.error(f"Failed to create entity {entity.name}: {e}")
            return False
    
    def create_entities(self, entities: List[Neo4jEntity]) -> bool:
        """
        Create multiple entities in a single transaction.
        
        Args:
            entities: List of Neo4jEntity objects to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not entities:
                return True
                
            with self.db_manager.neo4j_session() as session:
                # Batch create entities
                for entity in entities:
                    labels_str = ":".join(entity.labels)
                    props = {
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "created_at": datetime.utcnow().isoformat(),
                        **entity.properties
                    }
                    
                    query = f"""
                    MERGE (e:{labels_str} {{name: $name}})
                    SET e += $properties
                    """
                    
                    session.run(query, name=entity.name, properties=props)
                
                logging.info(f"Created/updated {len(entities)} entities")
                return True
                
        except Exception as e:
            logging.error(f"Failed to create entities: {e}")
            return False
    
    def create_relationship(self, relationship: Neo4jRelationship) -> bool:
        """
        Create a relationship between two entities.
        
        Args:
            relationship: Neo4jRelationship object to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.db_manager.neo4j_session() as session:
                props = {
                    "created_at": datetime.utcnow().isoformat(),
                    **relationship.properties
                }
                
                query = f"""
                MATCH (from:Entity {{name: $from_name}})
                MATCH (to:Entity {{name: $to_name}})
                MERGE (from)-[r:{relationship.relationship_type}]->(to)
                SET r += $properties
                RETURN r
                """
                
                result = session.run(
                    query,
                    from_name=relationship.from_entity,
                    to_name=relationship.to_entity,
                    properties=props
                )
                
                if result.single():
                    logging.info(f"Created relationship: {relationship.from_entity} -{relationship.relationship_type}-> {relationship.to_entity}")
                    return True
                else:
                    logging.warning(f"Failed to create relationship between {relationship.from_entity} and {relationship.to_entity}")
                    return False
                    
        except Exception as e:
            logging.error(f"Failed to create relationship: {e}")
            return False
    
    def create_relationships(self, relationships: List[Neo4jRelationship]) -> bool:
        """
        Create multiple relationships in a single transaction.
        
        Args:
            relationships: List of Neo4jRelationship objects to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not relationships:
                return True
                
            with self.db_manager.neo4j_session() as session:
                for rel in relationships:
                    props = {
                        "created_at": datetime.utcnow().isoformat(),
                        **rel.properties
                    }
                    
                    query = f"""
                    MATCH (from:Entity {{name: $from_name}})
                    MATCH (to:Entity {{name: $to_name}})
                    MERGE (from)-[r:{rel.relationship_type}]->(to)
                    SET r += $properties
                    """
                    
                    session.run(
                        query,
                        from_name=rel.from_entity,
                        to_name=rel.to_entity,
                        properties=props
                    )
                
                logging.info(f"Created {len(relationships)} relationships")
                return True
                
        except Exception as e:
            logging.error(f"Failed to create relationships: {e}")
            return False
    
    def get_entity(self, entity_name: str) -> Optional[Neo4jEntity]:
        """
        Retrieve an entity by name.
        
        Args:
            entity_name: Name of the entity to retrieve
            
        Returns:
            Neo4jEntity if found, None otherwise
        """
        try:
            with self.db_manager.neo4j_session() as session:
                query = """
                MATCH (e:Entity {name: $name})
                RETURN e, labels(e) as labels
                """
                
                result = session.run(query, name=entity_name)
                record = result.single()
                
                if record:
                    node = record["e"]
                    labels = record["labels"]
                    
                    # Extract properties
                    props = dict(node)
                    name = props.pop("name", entity_name)
                    entity_type = props.pop("entity_type", "Unknown")
                    
                    return Neo4jEntity(
                        id=str(node.id),
                        name=name,
                        entity_type=entity_type,
                        properties=props,
                        labels=labels
                    )
                
                return None
                
        except Exception as e:
            logging.error(f"Failed to get entity {entity_name}: {e}")
            return None
    
    def find_entities(
        self, 
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Neo4jEntity]:
        """
        Find entities matching criteria.
        
        Args:
            entity_type: Filter by entity type
            properties: Filter by properties
            limit: Maximum number of entities to return
            
        Returns:
            List of Neo4jEntity objects
        """
        try:
            with self.db_manager.neo4j_session() as session:
                where_clauses = []
                params = {"limit": limit}
                
                if entity_type:
                    where_clauses.append("e.entity_type = $entity_type")
                    params["entity_type"] = entity_type
                
                if properties:
                    for key, value in properties.items():
                        param_name = f"prop_{key}"
                        where_clauses.append(f"e.{key} = ${param_name}")
                        params[param_name] = value
                
                where_clause = " AND ".join(where_clauses) if where_clauses else "true"
                
                query = f"""
                MATCH (e:Entity)
                WHERE {where_clause}
                RETURN e, labels(e) as labels
                LIMIT $limit
                """
                
                result = session.run(query, **params)
                entities = []
                
                for record in result:
                    node = record["e"]
                    labels = record["labels"]
                    
                    props = dict(node)
                    name = props.pop("name", "")
                    ent_type = props.pop("entity_type", "Unknown")
                    
                    entities.append(Neo4jEntity(
                        id=str(node.id),
                        name=name,
                        entity_type=ent_type,
                        properties=props,
                        labels=labels
                    ))
                
                return entities
                
        except Exception as e:
            logging.error(f"Failed to find entities: {e}")
            return []
    
    def get_entity_relationships(
        self, 
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "BOTH",
        limit: int = 50
    ) -> List[Neo4jRelationship]:
        """
        Get relationships for an entity.
        
        Args:
            entity_name: Name of the entity
            relationship_types: Filter by relationship types
            direction: OUTGOING, INCOMING, or BOTH
            limit: Maximum number of relationships to return
            
        Returns:
            List of Neo4jRelationship objects
        """
        try:
            with self.db_manager.neo4j_session() as session:
                # Build relationship pattern based on direction
                if direction == "OUTGOING":
                    pattern = "(e)-[r]->(other)"
                elif direction == "INCOMING":
                    pattern = "(e)<-[r]-(other)"
                else:  # BOTH
                    pattern = "(e)-[r]-(other)"
                
                # Build type filter
                type_filter = ""
                params = {"name": entity_name, "limit": limit}
                
                if relationship_types:
                    type_filter = ":" + "|".join(relationship_types)
                
                query = f"""
                MATCH {pattern.replace('[r]', f'[r{type_filter}]')}
                WHERE e.name = $name
                RETURN r, startNode(r) as start_node, endNode(r) as end_node, type(r) as rel_type
                LIMIT $limit
                """
                
                result = session.run(query, **params)
                relationships = []
                
                for record in result:
                    rel = record["r"]
                    start_node = record["start_node"]
                    end_node = record["end_node"]
                    rel_type = record["rel_type"]
                    
                    props = dict(rel)
                    
                    relationships.append(Neo4jRelationship(
                        id=str(rel.id),
                        from_entity=start_node["name"],
                        to_entity=end_node["name"],
                        relationship_type=rel_type,
                        properties=props
                    ))
                
                return relationships
                
        except Exception as e:
            logging.error(f"Failed to get relationships for {entity_name}: {e}")
            return []
    
    def find_paths(
        self,
        start_entity: str,
        end_entity: Optional[str] = None,
        max_length: int = 3,
        relationship_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Neo4jPath]:
        """
        Find paths between entities.
        
        Args:
            start_entity: Starting entity name
            end_entity: Ending entity name (if None, finds paths from start_entity)
            max_length: Maximum path length
            relationship_types: Filter by relationship types
            limit: Maximum number of paths to return
            
        Returns:
            List of Neo4jPath objects
        """
        try:
            with self.db_manager.neo4j_session() as session:
                params = {"start_name": start_entity, "limit": limit}
                
                # Build relationship type filter
                rel_filter = ""
                if relationship_types:
                    rel_filter = ":" + "|".join(relationship_types)
                
                if end_entity:
                    # Find paths between specific entities
                    params["end_name"] = end_entity
                    query = f"""
                    MATCH p = (start:Entity {{name: $start_name}})-[r{rel_filter}*1..{max_length}]-(end:Entity {{name: $end_name}})
                    RETURN p
                    LIMIT $limit
                    """
                else:
                    # Find paths from start entity
                    query = f"""
                    MATCH p = (start:Entity {{name: $start_name}})-[r{rel_filter}*1..{max_length}]-(end:Entity)
                    RETURN p
                    LIMIT $limit
                    """
                
                result = session.run(query, **params)
                paths = []
                
                for record in result:
                    path = record["p"]
                    
                    # Extract nodes
                    nodes = []
                    for node in path.nodes:
                        props = dict(node)
                        name = props.pop("name", "")
                        entity_type = props.pop("entity_type", "Unknown")
                        
                        nodes.append(Neo4jEntity(
                            id=str(node.id),
                            name=name,
                            entity_type=entity_type,
                            properties=props,
                            labels=list(node.labels)
                        ))
                    
                    # Extract relationships
                    relationships = []
                    for rel in path.relationships:
                        props = dict(rel)
                        
                        relationships.append(Neo4jRelationship(
                            id=str(rel.id),
                            from_entity=rel.start_node["name"],
                            to_entity=rel.end_node["name"],
                            relationship_type=rel.type,
                            properties=props
                        ))
                    
                    paths.append(Neo4jPath(
                        nodes=nodes,
                        relationships=relationships,
                        length=len(relationships)
                    ))
                
                return paths
                
        except Exception as e:
            logging.error(f"Failed to find paths: {e}")
            return []
    
    def delete_entity(self, entity_name: str) -> bool:
        """
        Delete an entity and all its relationships.
        
        Args:
            entity_name: Name of the entity to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.db_manager.neo4j_session() as session:
                query = """
                MATCH (e:Entity {name: $name})
                DETACH DELETE e
                """
                
                result = session.run(query, name=entity_name)
                logging.info(f"Deleted entity: {entity_name}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to delete entity {entity_name}: {e}")
            return False
    
    def count_entities(self) -> int:
        """
        Get the total number of entities in the graph.
        
        Returns:
            int: Number of entities
        """
        try:
            with self.db_manager.neo4j_session() as session:
                result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                record = result.single()
                return record["count"] if record else 0
                
        except Exception as e:
            logging.error(f"Failed to count entities: {e}")
            return 0
    
    def count_relationships(self) -> int:
        """
        Get the total number of relationships in the graph.
        
        Returns:
            int: Number of relationships
        """
        try:
            with self.db_manager.neo4j_session() as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = result.single()
                return record["count"] if record else 0
                
        except Exception as e:
            logging.error(f"Failed to count relationships: {e}")
            return 0
    
    def clear_graph(self) -> bool:
        """
        Clear all entities and relationships from the graph.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.db_manager.neo4j_session() as session:
                # Delete all relationships and nodes
                session.run("MATCH (n) DETACH DELETE n")
                logging.info("Cleared all entities and relationships from Neo4j")
                return True
                
        except Exception as e:
            logging.error(f"Failed to clear graph: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if Neo4j is accessible and working.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            with self.db_manager.neo4j_session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                
                if record and record["test"] == 1:
                    entity_count = self.count_entities()
                    rel_count = self.count_relationships()
                    logging.info(f"Neo4j health check passed. Graph has {entity_count} entities and {rel_count} relationships.")
                    return True
                else:
                    return False
                    
        except Exception as e:
            logging.error(f"Neo4j health check failed: {e}")
            return False


def create_entity_from_extraction(
    name: str,
    entity_type: str,
    source_document: str,
    confidence: float = 1.0,
    additional_properties: Optional[Dict[str, Any]] = None
) -> Neo4jEntity:
    """
    Create a Neo4jEntity from extracted entity information.
    
    Args:
        name: Entity name
        entity_type: Type of entity (PERSON, ORGANIZATION, etc.)
        source_document: Source document where entity was found
        confidence: Confidence score of the extraction
        additional_properties: Additional properties to include
        
    Returns:
        Neo4jEntity object
    """
    properties = {
        "source_document": source_document,
        "confidence": confidence,
        "extraction_date": datetime.utcnow().isoformat()
    }
    
    if additional_properties:
        properties.update(additional_properties)
    
    return Neo4jEntity(
        id=None,
        name=name,
        entity_type=entity_type,
        properties=properties,
        labels=["Entity", entity_type]
    )


def create_relationship_from_extraction(
    from_entity: str,
    to_entity: str,
    relationship_type: str,
    source_document: str,
    confidence: float = 1.0,
    additional_properties: Optional[Dict[str, Any]] = None
) -> Neo4jRelationship:
    """
    Create a Neo4jRelationship from extracted relationship information.
    
    Args:
        from_entity: Source entity name
        to_entity: Target entity name
        relationship_type: Type of relationship
        source_document: Source document where relationship was found
        confidence: Confidence score of the extraction
        additional_properties: Additional properties to include
        
    Returns:
        Neo4jRelationship object
    """
    properties = {
        "source_document": source_document,
        "confidence": confidence,
        "extraction_date": datetime.utcnow().isoformat()
    }
    
    if additional_properties:
        properties.update(additional_properties)
    
    return Neo4jRelationship(
        id=None,
        from_entity=from_entity,
        to_entity=to_entity,
        relationship_type=relationship_type,
        properties=properties
    )
