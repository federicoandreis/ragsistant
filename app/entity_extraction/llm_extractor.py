"""
LLM-based entity and relation extraction.
Consolidates the LLM implementation from multiple files into a single, clean implementation.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional
import ollama

from .base import EntityExtractor, ExtractionResult, Entity, Relation, normalize_entity_id
from app.config import get_config


class LLMExtractor(EntityExtractor):
    """LLM-based entity and relation extractor using Ollama."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the LLM extractor.
        
        Args:
            model_name: Name of the Ollama model to use (defaults to config)
        """
        self.config = get_config()
        self.model_name = model_name or self.config.models.entity_extraction_model
        self.ollama_url = self.config.models.ollama_base_url
    
    @property
    def name(self) -> str:
        """Return the name of this extractor."""
        return f"llm_{self.model_name}"
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities and relations using LLM.
        
        Args:
            text: Input text to process
            
        Returns:
            ExtractionResult with entities, relations, claims, and summary
        """
        if not text or not text.strip():
            return ExtractionResult(entities=[], relations=[], claims=[])
        
        try:
            # Generate the prompt
            prompt = self._create_extraction_prompt(text)
            
            # Call the LLM
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",  # Request JSON response format
                options={
                    "temperature": 0.1,  # Lower temperature for more consistent results
                    "num_ctx": 8192,    # Request larger context window if supported
                }
            )
            
            content = response.get("message", {}).get("content", "").strip()
            logging.debug(f"[LLM Extraction] Raw response: {content[:500]}...")
            
            # Parse the JSON response
            parsed_data = self._safe_extract_json(content)
            
            # Convert to standardized format
            return self._convert_to_extraction_result(parsed_data, text)
            
        except Exception as e:
            logging.error(f"LLM extraction failed: {str(e)}")
            # Return basic fallback
            return ExtractionResult(
                entities=[],
                relations=[],
                claims=[],
                summary=text.split(". ")[0][:128] + ("..." if len(text) > 128 else "")
            )
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create the extraction prompt for the LLM."""
        return f"""Extract entities, relationships, and key claims from the following text. Return as JSON:

{{
  "entities": [
    {{"id": "unique_entity_id", "label": "entity_type", "name": "entity_name", "properties": {{}}, "aliases": []}}
  ],
  "relations": [
    {{"source": "entity_id", "type": "relation_type", "target": "entity_id", "properties": {{}}}}
  ],
  "claims": ["important fact 1", "important fact 2"],
  "summary": "Brief summary of the text"
}}

IMPORTANT RULES:
1. Only extract information explicitly present in the text
2. Entity IDs should be unique, descriptive, lowercase with underscores
3. Relation types should be descriptive verb phrases (e.g., "works_at", "married_to")
4. Only include meaningful entities (not single characters or fragments)
5. Claims should be key facts or statements from the text
6. Summary should be 1-2 sentences

Text: {text}"""
    
    def _safe_extract_json(self, content: str) -> Dict[str, Any]:
        """Safely extract JSON from LLM response with multiple fallback strategies."""
        if not content:
            return {}
        
        # Clean the content
        content = self._clean_llm_json(content)
        
        # Try 1: Direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try 2: Extract JSON from markdown code block
        try:
            json_match = re.search(r'```(?:json)?\n({[\s\S]*?})\n```', content)
            if json_match:
                return json.loads(json_match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try 3: Find the first JSON object in the text
        try:
            json_str = re.search(r'\{[\s\S]*\}', content)
            if json_str:
                return json.loads(json_str.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try 4: Look for the last JSON-like structure
        try:
            json_blocks = re.findall(r'\{[\s\S]*?\}', content)
            if json_blocks:
                # Try the last one first (most likely the complete response)
                for json_str in reversed(json_blocks):
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        logging.warning(f"Failed to extract JSON from LLM response: {content[:200]}...")
        return {}
    
    def _clean_llm_json(self, raw: str) -> str:
        """Clean LLM JSON output."""
        # Replace curly quotes with straight quotes
        raw = raw.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        # Remove other non-printable/control characters except newline/tab
        raw = re.sub(r'[\x00-\x09\x0b-\x1F\x7F]', '', raw)
        return raw.strip()
    
    def _convert_to_extraction_result(self, parsed_data: Dict[str, Any], original_text: str) -> ExtractionResult:
        """Convert parsed LLM data to ExtractionResult."""
        entities = []
        relations = []
        claims = parsed_data.get("claims", [])
        summary = parsed_data.get("summary", "")
        
        # Process entities
        raw_entities = parsed_data.get("entities", [])
        if isinstance(raw_entities, list):
            for ent_data in raw_entities:
                if not isinstance(ent_data, dict):
                    continue
                
                # Validate required fields
                if not all(k in ent_data for k in ("id", "label", "name")):
                    continue
                
                # Skip invalid entities
                name = ent_data.get("name", "").strip()
                if (len(name) <= 1 or 
                    '?' in name or 
                    name.lower() in {'?', 's', 'x', 'unknown'} or
                    name not in original_text):  # Entity name must appear in text
                    continue
                
                try:
                    entity = Entity(
                        id=normalize_entity_id(ent_data["id"]),
                        label=ent_data["label"],
                        name=name,
                        properties=ent_data.get("properties", {}),
                        aliases=ent_data.get("aliases", [])
                    )
                    entities.append(entity)
                except Exception as e:
                    logging.warning(f"Failed to create entity from {ent_data}: {e}")
        
        # Process relations
        raw_relations = parsed_data.get("relations", [])
        entity_ids = {e.id for e in entities}
        
        if isinstance(raw_relations, list):
            for rel_data in raw_relations:
                if not isinstance(rel_data, dict):
                    continue
                
                # Validate required fields
                if not all(k in rel_data for k in ("source", "type", "target")):
                    continue
                
                source = normalize_entity_id(rel_data["source"])
                target = normalize_entity_id(rel_data["target"])
                rel_type = rel_data["type"]
                
                # Validate relation
                if (source not in entity_ids or 
                    target not in entity_ids or 
                    source == target or
                    not rel_type or
                    not isinstance(rel_type, str)):
                    continue
                
                try:
                    relation = Relation(
                        source=source,
                        type=rel_type.upper().replace(' ', '_'),
                        target=target,
                        properties=rel_data.get("properties", {})
                    )
                    relations.append(relation)
                except Exception as e:
                    logging.warning(f"Failed to create relation from {rel_data}: {e}")
        
        # Generate summary if not provided
        if not summary and (entities or relations or len(original_text) > 20):
            summary = self._generate_summary(original_text)
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            claims=claims if isinstance(claims, list) else [],
            summary=summary
        )
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary if the LLM didn't provide one."""
        try:
            summary_prompt = f"""Summarize the following text in 1-2 sentences. 
            Only use information explicitly present in the text. 
            Do NOT add or infer details.

{text}"""
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": summary_prompt}],
                options={"temperature": 0.1}
            )
            summary = response.get("message", {}).get("content", "").strip()
            return summary
        except Exception as e:
            logging.warning(f"Failed to generate summary: {e}")
            # Fallback to first sentence
            return text.split(". ")[0][:128] + ("..." if len(text) > 128 else "")


def create_llm_extractor(model_name: Optional[str] = None) -> LLMExtractor:
    """Factory function to create an LLM extractor."""
    return LLMExtractor(model_name)
