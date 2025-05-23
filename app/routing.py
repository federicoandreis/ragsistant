from app.models import RoutingDecision
from app.graph.retrieval import retrieve_entity_context, retrieve_relevant_paths, get_top_communities_for_query
from app.utils.status import get_backend_status
import requests
import json
import logging
import sys
import re
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from typing import Optional

def route_query(query: str, entity_list=None, doc_name: str = None) -> RoutingDecision:
    """
    Rule-based router with backend status awareness:
    - If doc_name is provided, only consider backends marked 'ready'.
    - If query contains a known entity and 'graph' is ready, use 'graph'.
    - If only 'vector' is ready, use 'vector'.
    - If both are ready but no entity match, use 'vector'.
    - If neither ready, raise error.
    """
    if entity_list is None:
        entity_list = ["Renzo", "Lucia", "Don Rodrigo"]
    ready_backends = set()
    if doc_name:
        status = get_backend_status(doc_name)
        ready_backends = {k for k, v in status.items() if v == 'ready'}
        if not ready_backends:
            raise RuntimeError(f"No backends ready for doc '{doc_name}'. Ingestion may still be running.")
    else:
        ready_backends = {"graph", "vector"}
    for entity in entity_list:
        if entity.lower() in query.lower() and "graph" in ready_backends:
            return RoutingDecision(
                backend="graph",
                reason=f"Query mentions known entity '{entity}' and graph backend is ready.",
                entity_match=True,
                entity_name=entity
            )
    if "vector" in ready_backends:
        return RoutingDecision(
            backend="vector",
            reason="No known entity mentioned or graph not ready; using semantic retrieval.",
            entity_match=False
        )
    raise RuntimeError(f"No suitable backend is ready for doc '{doc_name}'. Current status: {status}")

def extract_json_from_text(text: str) -> dict:
    """
    Extract the first JSON object from a string, even if surrounded by text or markdown.
    This version is robust to multiline JSON and does not use unsupported regex extensions.
    """
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        return json.loads(json_str)
    raise ValueError("No JSON object found in LLM output.")

def coerce_routing_decision_types(parsed: dict) -> dict:
    """
    Coerce LLM output to match expected RoutingDecision types (model-agnostic).
    - entity_match: should be boolean (True/False), not entity name or string.
    - entity_name: should be string or None.
    """
    coerced = dict(parsed)
    # Coerce entity_match
    em = coerced.get("entity_match")
    en = coerced.get("entity_name")
    if isinstance(em, str):
        # If matches entity_name, set True
        if en and em.strip().lower() == str(en).strip().lower():
            coerced["entity_match"] = True
        elif em.strip().lower() in ("true", "yes"):  # Accept yes/true
            coerced["entity_match"] = True
        else:
            coerced["entity_match"] = False
    elif not isinstance(em, bool):
        coerced["entity_match"] = False
    return coerced

def llm_route_query(query: str, model: str = "llama3.1", ollama_url: str = "http://localhost:11434") -> RoutingDecision:
    """
    Uses an LLM (via Ollama) to decide routing for a user query.
    PATCH: Make hybrid recall explicit and reliable.
    PATCH2: Robust to Ollama endpoint errors (fallback to /api/generate and /api/chat)
    PATCH3: If both endpoints return 404, try root endpoint and provide a clear error message.
    PATCH4: Prefer /api/generate for stateless routing, fallback to /api/chat only if message history is present or /api/generate fails.
    PATCH5: Fallback to simple entity extraction if LLM routing fails to match an entity.
    """
    import re
    from app.graph.retrieval import retrieve_entity_context
    from app.models import RoutingDecision
    import logging
    entity_keywords = ["Renzo", "Lucia", "Don Rodrigo"]  # Extend as needed
    hybrid_context_keywords = [
        "combine", "both", "hybrid", "graph and vector", "mix", "together",
        "impact", "effect", "relationship", "relationships", "influence", "how", "compare", "connections", "describe", "synthesis", "context", "summary", "summarize"
    ]
    # Detect entity and context cues for hybrid
    mentioned_entity = None
    for entity in entity_keywords:
        if entity.lower() in query.lower():
            mentioned_entity = entity
            break
    hybrid_cue = any(kw in query.lower() for kw in hybrid_context_keywords)
    if mentioned_entity and hybrid_cue:
        return RoutingDecision(
            backend="hybrid",
            reason=f"Entity '{mentioned_entity}' and hybrid/contextual cue present in query.",
            entity_match=True,
            entity_name=mentioned_entity
        )
    if hybrid_cue:
        return RoutingDecision(
            backend="hybrid",
            reason=f"Hybrid/contextual cue present in query.",
            entity_match=False
        )
    if mentioned_entity:
        # Allow LLM to override to hybrid if it suggests it
        pass  # Let LLM decide below

    # Stateless routing: always prefer /api/generate
    system_prompt = (
        "You are a router. Decide if the query should go to the graph, vector, or hybrid backend. "
        "Output a JSON object with keys: backend, reason, entity_match (bool), entity_name (if any)."
    )
    user_prompt = f"Route this query: {query}"
    prompt = f"{system_prompt}\n\n{user_prompt}"
    payload_generate = {"model": model, "prompt": prompt, "stream": False}
    payload_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    last_404 = None
    try:
        # Try /api/generate first
        response = requests.post(f"{ollama_url}/api/generate", json=payload_generate, timeout=30)
        if response.status_code == 404:
            last_404 = "/api/generate"
            raise requests.HTTPError("/api/generate not found", response=response)
        response.raise_for_status()
        try:
            obj = json.loads(response.text)
            full_content = obj.get("response", "")
        except Exception:
            full_content = response.text
        print("[DEBUG] Aggregated LLM content:", full_content)
        logging.debug("[DEBUG] Aggregated LLM content:\n%s", full_content)
        parsed = extract_json_from_text(full_content)
        parsed = coerce_routing_decision_types(parsed)
        decision = RoutingDecision(**parsed)
        if decision.entity_match and decision.entity_name:
            return decision
    except Exception as e:
        logging.error(f"[ROUTER] LLM routing failed: {e}")
    # --- PATCH: Fallback entity extraction ---
    logging.info("[ROUTER] Attempting fallback entity extraction from query...")
    # Simple regex for capitalized words (naive proper noun extraction)
    candidates = re.findall(r"[A-Z][a-zA-Z0-9_'-]+", query)
    logging.info(f"[ROUTER] Fallback entity candidates: {candidates}")
    # Try to match any candidate against graph
    for cand in candidates:
        ctx = retrieve_entity_context(cand, hops=1, max_relationships=1)
        if ctx and ctx.get("entity"):
            logging.info(f"[ROUTER] Fallback matched entity: {cand}")
            return RoutingDecision(backend="graph", reason="Fallback entity match", entity_match=True, entity_name=cand)
    # No match, fallback to vector
    logging.info("[ROUTER] No entity match found in fallback, using vector backend.")
    return RoutingDecision(backend="vector", reason="No entity found in LLM or fallback extraction", entity_match=False, entity_name=None)

def format_graph_result(entity_context: dict, max_relationships: int = 3) -> str:
    """
    Extract and format a concise summary from the graph entity context for answer synthesis.
    Shows entity name, label, and up to N relationships in readable form.
    Tries to use 'name', then 'label', then 'id' for each node.
    """
    def get_node_display(node: dict) -> str:
        if not node:
            return "?"
        return node.get("name") or node.get("label") or node.get("id") or "?"

    if not entity_context or "entity" not in entity_context:
        logging.warning(f"[GRAPH FORMAT] No entity context found: {entity_context}")
        return "[No entity context found in graph DB]"
    entity = entity_context["entity"]
    name = entity.get("name", "<unknown>")
    label = entity.get("label", "Entity")
    rels = entity_context.get("relationships", [])
    logging.info(f"[GRAPH FORMAT] Formatting entity '{name}' ({label}) with {len(rels)} relationships.")
    summary = f"Entity: {name} ({label})\n"
    if not rels:
        summary += "No relationships found."
        return summary
    # Group relationships by type for clarity
    from collections import defaultdict
    grouped = defaultdict(list)
    for rel in rels[:max_relationships]:
        grouped[rel.get("rel_type","-")].append(rel)
    summary += f"Relationships (showing up to {max_relationships}):\n"
    for rel_type, rel_list in grouped.items():
        summary += f"  [{rel_type}]\n"
        for rel in rel_list:
            from_disp = get_node_display(rel.get("from"))
            to_disp = get_node_display(rel.get("to"))
            summary += f"    - {from_disp} --[{rel_type}]--> {to_disp}\n"
    logging.debug(f"[GRAPH FORMAT] Summary sent to LLM:\n{summary.strip()}")
    return summary.strip()

def hybrid_synthesize_answer(
    query: str,
    routing_decision: RoutingDecision,
    graph_result: str = None,
    vector_result: str = None,
    llm_model: str = "llama3.1:latest",
    ollama_url: str = "http://localhost:11434",
    community_ids: list = None,
    entity_name: str = None
) -> str:
    """
    Always synthesize an answer using an LLM, combining both contexts as needed.
    Now includes PathRAG-style retrieval: retrieves top-K relevant paths from the graph (prioritizing relevant communities).
    """
    # Retrieve top relevant paths for the entity (if available)
    relevant_paths = []
    if entity_name:
        try:
            relevant_paths = retrieve_relevant_paths(
                entity_name=entity_name,
                max_paths=5,
                max_length=3,
                community_ids=community_ids
            )
        except Exception as e:
            import logging
            logging.warning(f"[PATHRAG] Failed to retrieve relevant paths: {e}")
    # Format paths for LLM
    def format_path(path):
        node_names = [n.get("name", n.get("id", "?")) for n in path["nodes"]]
        rel_types = [r["type"] for r in path["relationships"]]
        # Interleave nodes and rels
        items = []
        for i, name in enumerate(node_names):
            items.append(name)
            if i < len(rel_types):
                items.append(f"--[{rel_types[i]}]-->")
        return " ".join(items)
    path_strs = [format_path(p) for p in relevant_paths]
    paths_section = "\n".join(f"- {s}" for s in path_strs) if path_strs else "[No relevant paths found]"
    # Build context for LLM
    context = f"Relevant graph paths:\n{paths_section}\n"
    if graph_result:
        context += f"Graph context: {graph_result}\n"
    if vector_result:
        context += f"Vector context: {vector_result}\n"
    # Models known to support /api/chat
    chat_models = [
        "llama3.1:latest", "llama3.2", "gemma3", "gemma2", "qwen3"
    ]
    # Completion-only models (use /api/generate)
    # This list is for logic, not for the UI selector
    def is_chat_model(model_name: str) -> bool:
        base = model_name.split(":")[0].split(".")[0].lower()
        return any(base.startswith(m) for m in chat_models)

    system_prompt = {
        "role": "system",
        "content": (
            """
You are an expert assistant. Your goal is to answer the user's question using ONLY the information provided below.

Instructions:
- Write a complete, compelling, and naturally flowing answer that fully addresses the user's question.
- Seamlessly integrate all relevant information from the evidence into your answer, presenting it as a unified, human-readable narrative.
- Do NOT mention or cite sources, evidence sections, or context types (e.g., do not write 'According to the Vector Context...' or 'The document states...').
- Do NOT mention if any evidence section is empty or irrelevant.
- Avoid mechanical, formulaic, or bullet-pointed structures unless the question specifically calls for them.
- Do NOT hedge, apologize, or mention your own limitations. If the answer is not present in the evidence, state this clearly and concisely.
- Focus on clarity, completeness, and readability, as if writing for an intelligent human reader.
"""
        )
    }

    user_prompt = (
        f"User Question:\n{query}\n\n"
        f"---\nKNOWLEDGE GRAPH PATHS (chains of entities/relationships):\n{paths_section}\n"
        f"---\nGRAPH CONTEXT (structured facts/summaries):\n{graph_result if graph_result else '[None]'}\n"
        f"---\nVECTOR CONTEXT (document excerpts):\n{vector_result if vector_result else '[None]'}\n"
    )

    try:
        used_chat = False
        if is_chat_model(llm_model):
            payload = {
                "model": llm_model,
                "messages": [
                    system_prompt,
                    {"role": "user", "content": user_prompt}
                ]
            }
            response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=30)
            response.raise_for_status()
            try:
                # Try standard JSON
                final_answer = response.json().get("message", {}).get("content", "").strip()
            except Exception:
                # Fallback: parse line by line (streamed JSON)
                final_answer = ""
                for line in response.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        msg = obj.get("message", {})
                        if "content" in msg:
                            final_answer += msg["content"]
                    except Exception:
                        continue
            used_chat = True
        else:
            prompt = f"{system_prompt['content']}\n\n{user_prompt}"
            payload = {
                "model": llm_model,
                "prompt": prompt
            }
            response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            try:
                final_answer = response.json().get("response", "").strip()
            except Exception:
                # Fallback: parse line by line (streamed JSON)
                final_answer = ""
                for line in response.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "response" in obj:
                            final_answer += obj["response"]
                    except Exception:
                        continue
        # Remove hedging/disclaimer patterns
        import re
        hedging_patterns = [
            r"^(unfortunately|sorry|i (do not|don't) have enough information|i cannot answer|i'm not sure|as an ai|as a language model)[\s,:-]*",
            r"^(i'm sorry,? )?but i (do not|don't) have enough information[\s,:-]*",
            r"^(apologies,? )?but i (do not|don't) know[\s,:-]*",
            r"^(note:|please note:)[\s,:-]*",
        ]
        answer_stripped = final_answer
        for pat in hedging_patterns:
            answer_stripped = re.sub(pat, "", answer_stripped, flags=re.IGNORECASE)
        answer_stripped = re.sub(r"^[\s,.:;-]+", "", answer_stripped)
        return answer_stripped
    except Exception as e:
        return f"[LLM SYNTHESIS ERROR] {e}"

# Use a small, fast model for routing and the main user-selected model for answer synthesis
ROUTING_MODEL = "gemma3:1b"
SYNTHESIS_MODEL_DEFAULT = "llama3.1:latest"

def main_rag_pipeline(query: str, llm_model: str = SYNTHESIS_MODEL_DEFAULT, ollama_url: str = "http://localhost:11434", max_relationships: int = 10, hops: int = 1) -> dict:
    """
    Main RAG pipeline:
    1. Route the query using LLM-based router.
    2. Retrieve from graph/vector/hybrid as needed.
    3. Synthesize a final answer using hybrid/orchestrated logic.
    Now logs timing for each major step to help identify bottlenecks.
    """
    import logging
    import time
    t0 = time.perf_counter()

    # Step 1: Route the query (use fast routing model)
    t_route_start = time.perf_counter()
    routing_decision = llm_route_query(query, model=ROUTING_MODEL, ollama_url=ollama_url)
    t_route_end = time.perf_counter()
    logging.info(f"[TIMING] Routing decision: {routing_decision} (took {t_route_end - t_route_start:.2f}s)")

    # Step 2: Retrieve from backends
    graph_result = None
    vector_result = None
    raw_vector_chunks = None
    community_ids = None
    t_vec_start = t_graph_start = t_vec_end = t_graph_end = None
    if routing_decision.backend == "graph":
        t_graph_start = time.perf_counter()
        graph_result = retrieve_entity_context(routing_decision.entity_name, hops=hops, max_relationships=max_relationships)
        community_ids = get_top_communities_for_query(query)
        t_graph_end = time.perf_counter()
        logging.info(f"[TIMING] Graph retrieval took {t_graph_end - t_graph_start:.2f}s")
    elif routing_decision.backend == "vector":
        from app.vectorstore import query_chunks
        t_vec_start = time.perf_counter()
        vector_result = query_chunks(query)
        raw_vector_chunks = vector_result.results if hasattr(vector_result, 'results') else None
        t_vec_end = time.perf_counter()
        logging.info(f"[TIMING] Vector retrieval took {t_vec_end - t_vec_start:.2f}s")
    elif routing_decision.backend == "hybrid":
        t_graph_start = time.perf_counter()
        graph_result = retrieve_entity_context(routing_decision.entity_name, hops=hops, max_relationships=max_relationships)
        community_ids = get_top_communities_for_query(query)
        t_graph_end = time.perf_counter()
        logging.info(f"[TIMING] Graph retrieval took {t_graph_end - t_graph_start:.2f}s")
        from app.vectorstore import query_chunks
        t_vec_start = time.perf_counter()
        vector_result = query_chunks(query)
        raw_vector_chunks = vector_result.results if hasattr(vector_result, 'results') else None
        t_vec_end = time.perf_counter()
        logging.info(f"[TIMING] Vector retrieval took {t_vec_end - t_vec_start:.2f}s")

    # Step 3: Synthesize answer
    t_llm_start = time.perf_counter()
    answer = hybrid_synthesize_answer(
        query=query,
        routing_decision=routing_decision,
        graph_result=graph_result,
        vector_result=vector_result,
        llm_model=llm_model,  # Synthesis model (user-selected or default)
        ollama_url=ollama_url,
        community_ids=community_ids,
        entity_name=routing_decision.entity_name
    )
    t_llm_end = time.perf_counter()
    logging.info(f"[TIMING] LLM synthesis took {t_llm_end - t_llm_start:.2f}s")
    t1 = time.perf_counter()
    logging.info(f"[TIMING] Total pipeline time: {t1 - t0:.2f}s")
    return {
        "answer": answer,
        "graph_result": graph_result,
        "vector_result": vector_result,
        "routing_decision": routing_decision,
        "raw_vector_chunks": [c.text for c in raw_vector_chunks] if raw_vector_chunks else None
    }

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Who is Renzo's wife?"
    llm_model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:latest"
    ollama_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:11434"
    print(f"[Pipeline Test] Query: {query}")
    print(f"Model: {llm_model}")
    result = main_rag_pipeline(query, llm_model, ollama_url)
    print("\n[Final Answer]:\n" + (result.get("answer") if isinstance(result, dict) else str(result)))
    if isinstance(result, dict):
        if "graph_result" in result:
            print("\n[GRAPH RESULT]:\n" + str(result["graph_result"]))
        if "vector_result" in result:
            print("\n[VECTOR RESULT]:\n" + str(result["vector_result"]))
        if "routing_decision" in result:
            print("\n[ROUTING DECISION]:\n" + str(result["routing_decision"]))
