"""
Base interfaces and types for entity extraction.
Defines common data structures and abstract interfaces for different extraction methods.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel


class Entity(BaseModel):
    """Standardized entity representation."""
    id: str
    label: str
    name: str
    properties: Optional[Dict[str, Any]] = None
    aliases: Optional[List[str]] = []


class Relation(BaseModel):
    """Standardized relation representation."""
    source: str
    type: str
    target: str
    properties: Optional[Dict[str, Any]] = None


class ExtractionResult(BaseModel):
    """Result of entity and relation extraction."""
    entities: List[Entity]
    relations: List[Relation]
    claims: Optional[List[str]] = []
    summary: Optional[str] = None


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""
    
    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and relations from text.
        
        Args:
            text: Input text to process
            
        Returns:
            ExtractionResult containing entities, relations, and optional claims/summary
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this extractor."""
        pass


def normalize_entity_id(entity_id: str) -> str:
    """Normalize entity ID to a consistent format."""
    if not entity_id or not isinstance(entity_id, str):
        return "unknown_entity"
    
    # Remove invalid characters and normalize
    import re
    normalized = re.sub(r'[^a-zA-Z0-9_]', '_', entity_id.strip())
    normalized = re.sub(r'_+', '_', normalized)
    normalized = normalized.strip('_').lower()
    
    # Ensure it starts with a letter
    if not normalized or not normalized[0].isalpha():
        normalized = f"entity_{normalized}" if normalized else "unknown_entity"
    
    return normalized


def is_valid_entity(entity: Dict[str, Any]) -> bool:
    """Check if an entity is valid and non-trivial."""
    if not isinstance(entity, dict):
        return False
    
    entity_id = entity.get("id", "")
    name = entity.get("name", "")
    
    # Check for required fields
    if not entity_id or not name:
        return False
    
    # Check for trivial/invalid content
    if (len(name.strip()) <= 1 or 
        '?' in name or 
        name.strip().lower() in {'?', 's', 'x', 'unknown'}):
        return False
    
    return True


def is_valid_relation(relation: Tuple[str, str, str], entities: List[Dict[str, Any]]) -> bool:
    """Check if a relation is valid."""
    if not isinstance(relation, (tuple, list)) or len(relation) != 3:
        return False
    
    source, rel_type, target = relation
    
    # Check for valid IDs
    if not source or not target or source == target:
        return False
    
    # Check if entities exist
    entity_ids = {e.get("id") for e in entities if is_valid_entity(e)}
    if source not in entity_ids or target not in entity_ids:
        return False
    
    # Check relation type
    if not rel_type or not isinstance(rel_type, str):
        return False
    
    return True


def convert_legacy_format(entities: List[Dict[str, Any]], 
                         relations: List[Tuple[str, str, str]], 
                         claims: Optional[List[str]] = None,
                         summary: Optional[str] = None) -> ExtractionResult:
    """
    Convert legacy tuple-based format to standardized ExtractionResult.
    
    Args:
        entities: List of entity dicts
        relations: List of (source, type, target) tuples
        claims: Optional list of claims
        summary: Optional summary text
        
    Returns:
        Standardized ExtractionResult
    """
    # Convert entities
    standardized_entities = []
    for ent in entities:
        if is_valid_entity(ent):
            try:
                entity = Entity(
                    id=normalize_entity_id(ent["id"]),
                    label=ent.get("label", "Entity"),
                    name=ent["name"],
                    properties=ent.get("properties", {}),
                    aliases=ent.get("aliases", [])
                )
                standardized_entities.append(entity)
            except Exception as e:
                import logging
                logging.warning(f"Failed to convert entity {ent}: {e}")
    
    # Convert relations
    standardized_relations = []
    entity_dict = {e.get("id"): e for e in entities if is_valid_entity(e)}
    
    for rel in relations:
        if is_valid_relation(rel, entities):
            try:
                source, rel_type, target = rel
                relation = Relation(
                    source=normalize_entity_id(source),
                    type=rel_type.upper().replace(' ', '_'),
                    target=normalize_entity_id(target),
                    properties={}
                )
                standardized_relations.append(relation)
            except Exception as e:
                import logging
                logging.warning(f"Failed to convert relation {rel}: {e}")
    
    return ExtractionResult(
        entities=standardized_entities,
        relations=standardized_relations,
        claims=claims or [],
        summary=summary
    )
