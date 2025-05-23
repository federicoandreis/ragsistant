import pytest
from app.graph_retrieval import retrieve_entity_context

def test_retrieve_entity_context():
    # Try retrieving an entity known to exist from previous ingestion
    entity_name = "Renzo"  # Example: adjust to a real entity in your partial_promessi_sposi.txt
    # Test 1-hop retrieval
    result_1hop = retrieve_entity_context(entity_name, hops=1, max_relationships=10)
    assert isinstance(result_1hop, dict)
    assert "entity" in result_1hop
    assert "relationships" in result_1hop
    print(f"[1-hop] Entity: {result_1hop['entity']}")
    print(f"[1-hop] Relationships count: {len(result_1hop['relationships'])}")
    print(f"[1-hop] Sample relationships: {result_1hop['relationships'][:2]}")
    # Test 2-hop retrieval for multi-hop context
    result_2hop = retrieve_entity_context(entity_name, hops=2, max_relationships=10)
    assert isinstance(result_2hop, dict)
    assert "entity" in result_2hop
    assert "relationships" in result_2hop
    print(f"[2-hop] Relationships count: {len(result_2hop['relationships'])}")
    print(f"[2-hop] Sample relationships: {result_2hop['relationships'][:2]}")
