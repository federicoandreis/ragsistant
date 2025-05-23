"""
Entity and relation normalization utilities.

This module provides functions to normalize and clean entity and relation data
extracted from text, ensuring consistency and quality for graph ingestion.
"""

import re
import logging
from typing import List, Dict, Any, Tuple


def normalize_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a list of entity dictionaries.
    
    Args:
        entities: List of entity dictionaries with keys like 'name', 'label', 'id'
        
    Returns:
        List of normalized entity dictionaries
    """
    normalized = []
    
    for entity in entities:
        if not isinstance(entity, dict):
            logging.warning(f"[NORMALIZE] Skipping non-dict entity: {entity}")
            continue
            
        # Ensure required fields exist
        name = entity.get('name', '').strip()
        if not name:
            logging.warning(f"[NORMALIZE] Skipping entity without name: {entity}")
            continue
            
        # Normalize name
        normalized_name = _normalize_text(name)
        if not normalized_name:
            logging.warning(f"[NORMALIZE] Skipping entity with empty normalized name: {name}")
            continue
            
        # Create normalized entity
        normalized_entity = {
            'id': entity.get('id', normalized_name.lower().replace(' ', '_')),
            'name': normalized_name,
            'label': entity.get('label', 'Entity'),
            'aliases': entity.get('aliases', [])
        }
        
        # Normalize aliases
        if normalized_entity['aliases']:
            normalized_entity['aliases'] = [
                _normalize_text(alias) for alias in normalized_entity['aliases']
                if _normalize_text(alias)
            ]
        
        normalized.append(normalized_entity)
    
    return normalized


def normalize_relations(relations: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Normalize a list of relation tuples.
    
    Args:
        relations: List of (source, relation_type, target) tuples
        
    Returns:
        List of normalized relation tuples
    """
    normalized = []
    
    for relation in relations:
        if not isinstance(relation, (tuple, list)) or len(relation) != 3:
            logging.warning(f"[NORMALIZE] Skipping invalid relation format: {relation}")
            continue
            
        source, rel_type, target = relation
        
        # Normalize components
        norm_source = _normalize_text(str(source))
        norm_target = _normalize_text(str(target))
        norm_rel_type = _normalize_relation_type(str(rel_type))
        
        # Skip if any component is empty
        if not all([norm_source, norm_target, norm_rel_type]):
            logging.warning(f"[NORMALIZE] Skipping relation with empty components: {relation}")
            continue
            
        # Skip self-relations
        if norm_source == norm_target:
            logging.warning(f"[NORMALIZE] Skipping self-relation: {norm_source} -> {norm_target}")
            continue
            
        normalized.append((norm_source, norm_rel_type, norm_target))
    
    return normalized


def _normalize_text(text: str) -> str:
    """
    Normalize text by cleaning whitespace and removing unwanted characters.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
        
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove quotes and other unwanted characters
    text = text.strip('"\'`')
    
    # Remove leading/trailing punctuation except for meaningful ones
    text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
    
    return text.strip()


def _normalize_relation_type(rel_type: str) -> str:
    """
    Normalize relation type for consistency.
    
    Args:
        rel_type: Relation type to normalize
        
    Returns:
        Normalized relation type
    """
    if not rel_type:
        return ""
        
    # Convert to uppercase and replace spaces/hyphens with underscores
    normalized = rel_type.upper().replace(' ', '_').replace('-', '_')
    
    # Remove unwanted characters
    normalized = re.sub(r'[^\w_]', '', normalized)
    
    # Ensure it starts with a letter
    if normalized and not normalized[0].isalpha():
        normalized = 'REL_' + normalized
        
    return normalized


def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate entities based on name similarity.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        List of deduplicated entities
    """
    seen_names = set()
    deduplicated = []
    
    for entity in entities:
        name = entity.get('name', '').lower()
        if name not in seen_names:
            seen_names.add(name)
            deduplicated.append(entity)
        else:
            logging.debug(f"[NORMALIZE] Skipping duplicate entity: {entity.get('name')}")
    
    return deduplicated


def deduplicate_relations(relations: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Remove duplicate relations.
    
    Args:
        relations: List of relation tuples
        
    Returns:
        List of deduplicated relations
    """
    seen_relations = set()
    deduplicated = []
    
    for relation in relations:
        if relation not in seen_relations:
            seen_relations.add(relation)
            deduplicated.append(relation)
        else:
            logging.debug(f"[NORMALIZE] Skipping duplicate relation: {relation}")
    
    return deduplicated


def validate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate entities and filter out invalid ones.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        List of valid entities
    """
    valid_entities = []
    
    for entity in entities:
        if not isinstance(entity, dict):
            continue
            
        name = entity.get('name', '').strip()
        if not name or len(name) < 2:
            continue
            
        # Skip entities that are just numbers or single characters
        if name.isdigit() or len(name) == 1:
            continue
            
        # Skip common stop words that might be extracted as entities
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if name.lower() in stop_words:
            continue
            
        valid_entities.append(entity)
    
    return valid_entities


def validate_relations(relations: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Validate relations and filter out invalid ones.
    
    Args:
        relations: List of relation tuples
        
    Returns:
        List of valid relations
    """
    valid_relations = []
    
    for relation in relations:
        if not isinstance(relation, (tuple, list)) or len(relation) != 3:
            continue
            
        source, rel_type, target = relation
        
        # Skip if any component is too short
        if any(len(str(comp).strip()) < 2 for comp in [source, rel_type, target]):
            continue
            
        # Skip if source and target are the same
        if str(source).strip().lower() == str(target).strip().lower():
            continue
            
        valid_relations.append(relation)
    
    return valid_relations
