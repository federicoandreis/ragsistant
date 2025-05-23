"""
Entity coalescing and deduplication module for graph ingestion.

This module handles:
- Entity merging based on name and aliases
- Similarity detection using string matching
- ID mapping for relation updates
"""

import logging
import difflib
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict


def coalesce_entities(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Merge/coalesce entities that refer to the same real-world entity based on name and aliases.
    Returns a list of unique entities and a mapping from old IDs to new IDs.
    
    Args:
        entities: List of entity dictionaries with 'id', 'name', and optional 'aliases'
        
    Returns:
        Tuple of (unique_entities, id_mapping)
        - unique_entities: Deduplicated list of entities
        - id_mapping: Dict mapping old entity IDs to new (canonical) entity IDs
    """
    unique = []
    id_map = {}
    seen_names = {}
    
    # Build alias sets for each entity
    for ent in entities:
        names = set([ent.get("name", "").lower()])
        aliases = set([a.lower() for a in ent.get("aliases", [])])
        all_names = names | aliases
        found = False
        
        # Check against existing unique entities
        for u in unique:
            u_names = set([u.get("name", "").lower()]) | set([a.lower() for a in u.get("aliases", [])])
            
            # Check for exact overlap or high string similarity
            if _entities_should_merge(all_names, u_names):
                # Merge aliases
                u["aliases"] = list(set(u.get("aliases", [])) | aliases | names)
                id_map[ent["id"]] = u["id"]
                found = True
                logging.debug(f"Merged entity '{ent.get('name')}' into '{u.get('name')}'")
                break
        
        if not found:
            unique.append(ent)
            id_map[ent["id"]] = ent["id"]
    
    logging.info(f"Entity coalescing: {len(entities)} -> {len(unique)} entities ({len(entities) - len(unique)} merged)")
    return unique, id_map


def _entities_should_merge(names1: Set[str], names2: Set[str], similarity_threshold: float = 0.85) -> bool:
    """
    Determine if two entities should be merged based on name similarity.
    
    Args:
        names1: Set of names/aliases for first entity (lowercase)
        names2: Set of names/aliases for second entity (lowercase)
        similarity_threshold: Minimum similarity ratio for merging
        
    Returns:
        True if entities should be merged
    """
    # Direct overlap check
    if names1 & names2:
        return True
    
    # String similarity check
    for n1 in names1:
        for n2 in names2:
            if _calculate_similarity(n1, n2) > similarity_threshold:
                return True
    
    return False


def _calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity ratio between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    if not str1 or not str2:
        return 0.0
    
    # Skip very short strings to avoid false positives
    if len(str1) <= 2 or len(str2) <= 2:
        return 1.0 if str1 == str2 else 0.0
    
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def remap_relations_with_coalesced_entities(
    relations: List[Tuple[str, str, str]], 
    id_map: Dict[str, str]
) -> List[Tuple[str, str, str]]:
    """
    Remap relation entity IDs using the coalesced entity ID mapping.
    Filters out self-loops that may result from coalescing.
    
    Args:
        relations: List of (source_id, relation_type, target_id) tuples
        id_map: Mapping from old entity IDs to new (canonical) entity IDs
        
    Returns:
        List of remapped relations with self-loops removed
    """
    remapped_relations = []
    
    for src, typ, tgt in relations:
        if src in id_map and tgt in id_map:
            src_id = id_map[src]
            tgt_id = id_map[tgt]
            
            # Skip self-loops that result from entity coalescing
            if src_id == tgt_id:
                logging.debug(f"Skipping self-loop relation: ({src_id})-[:{typ}]->({tgt_id})")
                continue
                
            remapped_relations.append((src_id, typ, tgt_id))
        else:
            # Log missing entities (shouldn't happen in normal flow)
            missing = []
            if src not in id_map:
                missing.append(f"source '{src}'")
            if tgt not in id_map:
                missing.append(f"target '{tgt}'")
            logging.warning(f"Relation ({src})-[:{typ}]->({tgt}) references missing entities: {', '.join(missing)}")
    
    logging.info(f"Relation remapping: {len(relations)} -> {len(remapped_relations)} relations ({len(relations) - len(remapped_relations)} removed)")
    return remapped_relations


def validate_entity_consistency(entities: List[Dict[str, Any]]) -> List[str]:
    """
    Validate entity data consistency and return list of issues found.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        List of validation error messages
    """
    issues = []
    seen_ids = set()
    
    for i, ent in enumerate(entities):
        # Check required fields
        if not ent.get("id"):
            issues.append(f"Entity {i}: Missing 'id' field")
            continue
            
        if not ent.get("name"):
            issues.append(f"Entity {ent['id']}: Missing 'name' field")
        
        # Check for duplicate IDs
        if ent["id"] in seen_ids:
            issues.append(f"Entity {ent['id']}: Duplicate entity ID")
        seen_ids.add(ent["id"])
        
        # Check aliases format
        aliases = ent.get("aliases", [])
        if aliases and not isinstance(aliases, list):
            issues.append(f"Entity {ent['id']}: 'aliases' should be a list, got {type(aliases)}")
    
    return issues
