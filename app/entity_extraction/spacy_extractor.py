"""
spaCy-based entity and relation extraction.
Consolidates the spaCy implementation from multiple files into a single, clean implementation.
"""
import logging
from typing import List, Dict, Any
import spacy

from .base import EntityExtractor, ExtractionResult, Entity, Relation, normalize_entity_id


class SpacyExtractor(EntityExtractor):
    """spaCy-based entity and relation extractor."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the spaCy extractor.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.model_name = model_name
        self._nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the spaCy model, downloading if necessary."""
        try:
            self._nlp = spacy.load(self.model_name)
            logging.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logging.info(f"spaCy model {self.model_name} not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", self.model_name])
            self._nlp = spacy.load(self.model_name)
            logging.info(f"Downloaded and loaded spaCy model: {self.model_name}")
    
    @property
    def name(self) -> str:
        """Return the name of this extractor."""
        return f"spacy_{self.model_name}"
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and relations using spaCy.
        
        Args:
            text: Input text to process
            
        Returns:
            ExtractionResult with entities, relations, and claims
        """
        if not text or not text.strip():
            return ExtractionResult(entities=[], relations=[], claims=[])
        
        doc = self._nlp(text)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Extract relations
        relations = self._extract_relations(doc, entities)
        
        # Extract claims (sentences as basic claims)
        claims = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            claims=claims,
            summary=claims[0] if claims else ""
        )
    
    def _extract_entities(self, doc) -> List[Entity]:
        """Extract entities from spaCy doc."""
        entities = []
        entity_map = {}
        
        for ent in doc.ents:
            # Create unique ID based on position to avoid duplicates
            ent_id = f"{ent.label_}_{ent.start_char}_{ent.end_char}"
            normalized_id = normalize_entity_id(ent_id)
            
            # Skip trivial entities
            if (len(ent.text.strip()) <= 1 or 
                '?' in ent.text or 
                ent.text.strip().lower() in {'?', 's', 'x'}):
                continue
            
            entity = Entity(
                id=normalized_id,
                label=ent.label_,
                name=ent.text.strip(),
                properties={
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "confidence": getattr(ent, 'confidence', 1.0)
                },
                aliases=[]
            )
            
            entities.append(entity)
            entity_map[(ent.start_char, ent.end_char)] = normalized_id
        
        return entities
    
    def _extract_relations(self, doc, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities in the same sentence."""
        relations = []
        entity_positions = {(ent.properties["start_char"], ent.properties["end_char"]): ent.id 
                          for ent in entities}
        
        for sent in doc.sents:
            # Find entities in this sentence
            sent_entities = []
            for ent in sent.ents:
                pos_key = (ent.start_char, ent.end_char)
                if pos_key in entity_positions:
                    sent_entities.append((ent, entity_positions[pos_key]))
            
            # Only extract relations for sentences with 2-4 entities
            if len(sent_entities) < 2 or len(sent_entities) > 4:
                continue
            
            # Extract relations between entity pairs
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    ent1, ent1_id = sent_entities[i]
                    ent2, ent2_id = sent_entities[j]
                    
                    # Look for a verb between the entities to determine relation type
                    rel_type = self._find_relation_type(sent, ent1, ent2)
                    
                    # Only add meaningful relations
                    if rel_type != "RELATED_TO" or len(sent_entities) == 2:
                        relation = Relation(
                            source=ent1_id,
                            type=rel_type,
                            target=ent2_id,
                            properties={
                                "sentence": sent.text.strip(),
                                "confidence": 0.8 if rel_type != "RELATED_TO" else 0.5
                            }
                        )
                        relations.append(relation)
        
        return relations
    
    def _find_relation_type(self, sent, ent1, ent2) -> str:
        """Find the relation type between two entities in a sentence."""
        # Look for verbs between the entities
        start_pos = min(ent1.end, ent2.end)
        end_pos = max(ent1.start, ent2.start)
        
        for token in sent[start_pos:end_pos]:
            if token.pos_ == "VERB":
                return token.lemma_.upper().replace(" ", "_")
        
        # Look for prepositions or other connecting words
        for token in sent[start_pos:end_pos]:
            if token.pos_ in ["ADP", "SCONJ", "CCONJ"]:  # Prepositions, conjunctions
                return token.text.upper().replace(" ", "_")
        
        # Default relation type
        return "RELATED_TO"


def create_spacy_extractor(model_name: str = "en_core_web_sm") -> SpacyExtractor:
    """Factory function to create a spaCy extractor."""
    return SpacyExtractor(model_name)


def extract_entities_and_relations_spacy(text: str, model_name: str = "en_core_web_sm"):
    """
    Legacy function for backward compatibility.
    Extract entities and relations using spaCy and return in the old format.
    
    Args:
        text: Input text to process
        model_name: spaCy model name to use
        
    Returns:
        Tuple of (entities, relations, summary)
    """
    extractor = SpacyExtractor(model_name)
    result = extractor.extract(text)
    
    # Convert to old format
    entities = []
    for entity in result.entities:
        entities.append({
            "id": entity.id,
            "name": entity.name,
            "label": entity.label,
            "aliases": entity.aliases
        })
    
    relations = []
    for relation in result.relations:
        relations.append((relation.source, relation.type, relation.target))
    
    return entities, relations, result.summary
