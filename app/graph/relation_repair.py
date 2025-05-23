"""
Relation repair and validation module for graph ingestion.

This module handles:
- Relationship type sanitization for Cypher
- Entity ID validation
- Malformed relation repair using heuristics and LLM
- Graph context fetching for repair assistance
"""

import logging
import json
import re
from typing import List, Tuple, Dict, Any, Optional, Set
import ollama
from app.config import get_config
from app.db.connections import get_db_manager
from app.entity_extraction import get_spacy_extractor


def sanitize_relation_type(rel: str) -> str:
    """
    Convert a string to a valid Cypher relationship type (letters, numbers, underscores, no spaces).
    If rel is None, not a string, empty, or numeric, default to 'related_to'.
    """
    # If rel is not a string or is empty, default to 'related_to'
    if not isinstance(rel, str) or not rel.strip():
        rel = 'related_to'
    # If rel is numeric, default to 'related_to'
    if isinstance(rel, (int, float)) or (isinstance(rel, str) and rel.isdigit()):
        rel = 'related_to'
    rel = re.sub(r'[^a-zA-Z0-9]', '_', rel)
    rel = re.sub(r'_+', '_', rel)
    rel = rel.strip('_').lower()
    # If rel doesn't start with a letter, default to 'related_to'
    if not rel or not rel[0].isalpha():
        rel = 'related_to'
    return rel


def is_valid_entity_id(entity_id: str, entities: List[Dict[str, Any]]) -> bool:
    """
    Return True if the entity_id is non-trivial, not a fragment, not a single char, 
    not containing '?' and present in entities.
    """
    if not entity_id or not isinstance(entity_id, str):
        return False
    if len(entity_id.strip()) <= 1 or '?' in entity_id or entity_id.strip().lower() in {'?', 's', 'x'}:
        return False
    # Check if present in entities
    for e in entities:
        if e.get("id") == entity_id:
            name = e.get("name", "")
            if len(name.strip()) <= 1 or '?' in name or name.strip().lower() in {'?', 's', 'x'}:
                return False
            return True
    return False


def fetch_graph_context_for_entities(entity_ids: Set[str], max_hops: int = 1, max_rels: int = 50) -> Dict[str, Dict]:
    """
    Fetch a summary of the existing graph for a set of entity IDs (by id), including aliases and relations.
    Returns a dict: {id: {entity, relations: [...]}}
    """
    context = {}
    db_manager = get_db_manager()
    
    with db_manager.neo4j_session() as session:
        for eid in entity_ids:
            node_res = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e LIMIT 1", id=eid
            )
            node = node_res.single()
            if not node:
                continue
            e = dict(node["e"])
            context[eid] = {"entity": e, "relations": []}
            rels_res = session.run(
                f"""
                MATCH (e:Entity {{id: $id}})-[r*1..{max_hops}]-(n:Entity)
                WITH e, n, r
                UNWIND r as rel
                RETURN type(rel) as rel_type, startNode(rel) as from_node, endNode(rel) as to_node
                LIMIT $max_rels
                """,
                id=eid, max_rels=max_rels
            )
            for row in rels_res:
                context[eid]["relations"].append({
                    "rel_type": row["rel_type"],
                    "from": dict(row["from_node"]),
                    "to": dict(row["to_node"])
                })
    return context


def repair_malformed_relations(
    malformed_relations: List[Any], 
    entities: List[Dict[str, Any]], 
    chunk_text: str, 
    model: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """
    Enhanced: Efficiently repair malformed relations using heuristics, LLM, and spaCy fallback, 
    strictly enforcing entity/relation validity as per the extraction prompt.
    Now also uses existing graph context to aid repair and disambiguation.
    """
    config = get_config()
    if model is None:
        model = config.models.entity_extraction_model
    
    entity_ids = {e.get("id") for e in entities if is_valid_entity_id(e.get("id"), entities)}
    graph_context = fetch_graph_context_for_entities(entity_ids)
    repaired = []
    
    # Quick heuristic repair for simple cases
    for rel in malformed_relations:
        if (
            isinstance(rel, list)
            and len(rel) == 2
            and all(r in entity_ids for r in rel)
            and rel[0] != rel[1]
        ):
            repaired.append((rel[0], "related_to", rel[1]))
    
    remaining = [rel for rel in malformed_relations if rel not in repaired]
    if not remaining:
        return repaired
    
    # Try LLM repair with multiple models
    llm_models = [model, "gemma3:1b", "phi4"]
    llm_success = False
    graph_context_str = json.dumps({
        eid: {"entity": ctx["entity"], "relations": ctx["relations"]} 
        for eid, ctx in graph_context.items()
    }, default=str)[:2000]  # truncate for prompt size
    
    for llm_model in llm_models:
        try:
            prompt = (
                f"You are repairing malformed knowledge graph relations. "
                f"Entities (valid IDs): {list(entity_ids)}. "
                f"Malformed relations: {remaining}. "
                f"Existing graph context (partial): {graph_context_str}\n"
                f"Text: {chunk_text}\n"
                f"Rules: Only output triplets (entity1_id, relation, entity2_id) where both IDs are valid, non-trivial, not fragments, not single-char, not containing '?'. "
                f"Use graph context to infer likely relation types or disambiguate entities. "
                f"If a [MENTIONS] relation, only output if both ends are valid. "
                f"Relation type must be a descriptive verb phrase. "
                f"Output a valid JSON list of triplets only."
            )
            response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
            content = response.get("message", {}).get("content", "")
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            json_str = content[json_start:json_end]
            parsed = json.loads(json_str)
            for triplet in parsed:
                if (
                    isinstance(triplet, list)
                    and len(triplet) == 3
                    and triplet[0] in entity_ids
                    and triplet[2] in entity_ids
                    and triplet[0] != triplet[2]
                ):
                    repaired.append(tuple(triplet))
            llm_success = True
            break
        except Exception as e:
            if 'content' in locals():
                logging.warning(f"LLM repair failed with model {llm_model}: {e}\nRaw output: {content}")
            else:
                logging.warning(f"LLM repair failed with model {llm_model}: {e}")
    
    # Fallback to spaCy extraction
    if not llm_success:
        logging.info("Falling back to spaCy-based extraction for relation repair.")
        spacy_extractor = get_spacy_extractor()
        result = spacy_extractor.extract(chunk_text)
        spacy_relations = [(rel.source, rel.type, rel.target) for rel in result.relations]
        
        for rel in spacy_relations:
            if (
                isinstance(rel, (list, tuple))
                and len(rel) == 3
                and rel[0] in entity_ids
                and rel[2] in entity_ids
                and rel[0] != rel[2]
            ):
                repaired.append(tuple(rel))
    
    # Final validation filter
    filtered = [r for r in repaired if is_valid_entity_id(r[0], entities) and is_valid_entity_id(r[2], entities) and r[0] != r[2]]
    if len(filtered) < len(repaired):
        logging.info(f"Dropped {len(repaired)-len(filtered)} relations as invalid after final filter.")
    return filtered


def process_relations_with_repair(
    relations: List[Tuple], 
    entities: List[Dict[str, Any]], 
    chunk_text: str, 
    model: Optional[str] = None, 
    repair: bool = False
) -> List[Tuple[str, str, str]]:
    """
    If repair is enabled, separate valid and malformed relations, attempt to repair malformed, and return all valid triplets.
    If repair is False, return only valid triplets (skip malformed, log).
    """
    config = get_config()
    if model is None:
        model = config.models.entity_extraction_model
    
    valid = []
    malformed = []
    
    for rel in relations:
        if (
            isinstance(rel, (list, tuple))
            and len(rel) == 3
            and all(rel)
            and rel[0] != rel[2]
            and is_valid_entity_id(rel[0], entities)
            and is_valid_entity_id(rel[2], entities)
        ):
            valid.append(tuple(rel))
        else:
            malformed.append(rel)
    
    if malformed:
        if repair:
            logging.info(f"Attempting to repair {len(malformed)} malformed relations...")
            repaired = repair_malformed_relations(malformed, entities, chunk_text, model=model)
            if repaired:
                logging.info(f"Repaired {len(repaired)} relations via heuristics/LLM.")
            valid.extend(repaired)
        else:
            logging.info(f"Skipped {len(malformed)} malformed relations.")
    
    return valid
