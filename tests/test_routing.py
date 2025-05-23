import pytest
from unittest.mock import patch, Mock
from app.routing import route_query, llm_route_query, hybrid_synthesize_answer
from app.models import RoutingDecision

def test_route_query_entity():
    query = "Who is Renzo's wife?"
    decision = route_query(query)
    assert isinstance(decision, RoutingDecision)
    assert decision.backend == "graph"
    assert decision.entity_match is True
    assert decision.entity_name == "Renzo"
    assert "entity" in decision.reason

def test_route_query_semantic():
    query = "What is the story about?"
    decision = route_query(query)
    assert isinstance(decision, RoutingDecision)
    assert decision.backend == "vector"
    assert decision.entity_match is False
    assert decision.entity_name is None
    assert "semantic" in decision.reason.lower()

def mock_ollama_streaming_response_graph(*args, **kwargs):
    class FakeResponse:
        def __init__(self):
            self.text = (
                '{"message": {"content": "{\\n"}}\n'
                '{"message": {"content": "  \\\"backend\\\": \\\"graph\\\","}}\n'
                '{"message": {"content": "  \\\"reason\\\": \\\"Query mentions entity Renzo.\\\","}}\n'
                '{"message": {"content": "  \\\"entity_match\\\": true,"}}\n'
                '{"message": {"content": "  \\\"entity_name\\\": \\\"Renzo\\\""}}\n'
                '{"message": {"content": "}"}}\n'
            )
        def raise_for_status(self):
            pass
    return FakeResponse()

def mock_ollama_streaming_response_vector(*args, **kwargs):
    class FakeResponse:
        def __init__(self):
            self.text = (
                '{"message": {"content": "{\\n"}}\n'
                '{"message": {"content": "  \\\"backend\\\": \\\"vector\\\","}}\n'
                '{"message": {"content": "  \\\"reason\\\": \\\"No known entity; semantic search.\\\","}}\n'
                '{"message": {"content": "  \\\"entity_match\\\": false,"}}\n'
                '{"message": {"content": "  \\\"entity_name\\\": null"}}\n'
                '{"message": {"content": "}"}}\n'
            )
        def raise_for_status(self):
            pass
    return FakeResponse()

def test_llm_route_query_graph():
    with patch("requests.post", mock_ollama_streaming_response_graph):
        decision = llm_route_query("Who is Renzo's wife?")
        assert decision.backend == "graph"
        assert decision.entity_match is True
        assert decision.entity_name == "Renzo"
        assert "entity" in decision.reason

def test_llm_route_query_vector():
    with patch("requests.post", mock_ollama_streaming_response_vector):
        decision = llm_route_query("What is the story about?")
        assert decision.backend == "vector"
        assert decision.entity_match is False
        assert decision.entity_name is None
        assert "semantic" in decision.reason.lower()

def test_hybrid_synthesize_answer_graph_shortcut():
    rd = RoutingDecision(backend="graph", reason="Entity match", entity_match=True, entity_name="Renzo")
    graph_result = "Renzo is married to Lucia."
    answer = hybrid_synthesize_answer("Who is Renzo's wife?", rd, graph_result=graph_result, vector_result=None)
    assert answer.startswith("[GRAPH]")
    assert "Lucia" in answer

def test_hybrid_synthesize_answer_vector_shortcut():
    rd = RoutingDecision(backend="vector", reason="No entity", entity_match=False)
    vector_result = "The story is about Renzo and his struggles."
    answer = hybrid_synthesize_answer("What is the story about?", rd, graph_result=None, vector_result=vector_result)
    assert answer.startswith("[VECTOR]")
    assert "Renzo" in answer

def test_hybrid_synthesize_answer_llm(monkeypatch):
    rd = RoutingDecision(backend="hybrid", reason="Ambiguous", entity_match=False)
    graph_result = "Renzo is married to Lucia."
    vector_result = "The story is about Renzo and his struggles."
    # Patch requests.post to simulate LLM synthesis
    class FakeResponse:
        def __init__(self):
            self.text = '{"message": {"content": "Renzo is married to Lucia, who is central to the story."}}\n'
        def raise_for_status(self):
            pass
    def fake_post(*args, **kwargs):
        return FakeResponse()
    monkeypatch.setattr("requests.post", fake_post)
    answer = hybrid_synthesize_answer("Who is Renzo's wife?", rd, graph_result, vector_result)
    assert answer.startswith("[LLM SYNTHESIS]")
    assert "Lucia" in answer
