"""
Entity extraction module.
Provides a unified interface for different entity extraction methods.
"""
from typing import Optional, Union, List, Dict, Any, Tuple

from .base import (
    Entity, 
    Relation, 
    ExtractionResult, 
    EntityExtractor,
    convert_legacy_format,
    normalize_entity_id,
    is_valid_entity,
    is_valid_relation
)
from .spacy_extractor import SpacyExtractor, create_spacy_extractor
from .llm_extractor import LLMExtractor, create_llm_extractor


# Default extractors
_default_spacy_extractor: Optional[SpacyExtractor] = None
_default_llm_extractor: Optional[LLMExtractor] = None


def get_spacy_extractor(model_name: str = "en_core_web_sm") -> SpacyExtractor:
    """Get or create a spaCy extractor."""
    global _default_spacy_extractor
    if _default_spacy_extractor is None or _default_spacy_extractor.model_name != model_name:
        _default_spacy_extractor = create_spacy_extractor(model_name)
    return _default_spacy_extractor


def get_llm_extractor(model_name: Optional[str] = None) -> LLMExtractor:
    """Get or create an LLM extractor."""
    global _default_llm_extractor
    if _default_llm_extractor is None or (model_name and _default_llm_extractor.model_name != model_name):
        _default_llm_extractor = create_llm_extractor(model_name)
    return _default_llm_extractor


def extract_entities_and_relations(text: str, method: str = "spacy", **kwargs) -> ExtractionResult:
    """
    Extract entities and relations from text using the specified method.
    
    Args:
        text: Input text to process
        method: Extraction method ("spacy" or "llm")
        **kwargs: Additional arguments for the extractor
        
    Returns:
        ExtractionResult containing entities, relations, claims, and summary
    """
    if method.lower() == "spacy":
        extractor = get_spacy_extractor(kwargs.get("model_name", "en_core_web_sm"))
    elif method.lower() == "llm":
        extractor = get_llm_extractor(kwargs.get("model_name"))
    else:
        raise ValueError(f"Unknown extraction method: {method}. Use 'spacy' or 'llm'.")
    
    return extractor.extract(text)


def extract_entities_and_relations_spacy(text: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], List[str]]:
    """
    Legacy interface for spaCy extraction.
    Returns the old tuple format for backward compatibility.
    """
    result = get_spacy_extractor().extract(text)
    
    # Convert to legacy format
    entities = [entity.model_dump() for entity in result.entities]
    relations = [(rel.source, rel.type, rel.target) for rel in result.relations]
    claims = result.claims or []
    
    return entities, relations, claims


def extract_entities_and_relations_llm(text: str, model: str = None) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], List[str]]:
    """
    Legacy interface for LLM extraction.
    Returns the old tuple format for backward compatibility.
    """
    result = get_llm_extractor(model).extract(text)
    
    # Convert to legacy format
    entities = [entity.model_dump() for entity in result.entities]
    relations = [(rel.source, rel.type, rel.target) for rel in result.relations]
    claims = result.claims or []
    
    return entities, relations, claims


# Backward compatibility aliases
def extract_entities_and_relations_llama(text: str, model: str = None) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], List[str]]:
    """Alias for extract_entities_and_relations_llm for backward compatibility."""
    return extract_entities_and_relations_llm(text, model)


# Export main interfaces
__all__ = [
    # New standardized interface
    "Entity",
    "Relation", 
    "ExtractionResult",
    "EntityExtractor",
    "extract_entities_and_relations",
    "get_spacy_extractor",
    "get_llm_extractor",
    
    # Legacy interfaces for backward compatibility
    "extract_entities_and_relations_spacy",
    "extract_entities_and_relations_llm",
    "extract_entities_and_relations_llama",
    
    # Utility functions
    "convert_legacy_format",
    "normalize_entity_id",
    "is_valid_entity",
    "is_valid_relation",
    
    # Extractor classes
    "SpacyExtractor",
    "LLMExtractor",
    "create_spacy_extractor",
    "create_llm_extractor",
]
