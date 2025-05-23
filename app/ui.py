import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import datetime
from pathlib import Path

# Suppress noisy PyTorch warning about torch.classes '__path__._path'
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Tried to instantiate class '__path__._path', but it does not exist!.*"
)
# This warning is harmless and occurs when PyTorch or a dependent library probes for optional C++ extensions that are not present. It does not affect functionality unless you rely on custom torch.classes, which this app does not.

st.set_page_config(page_title="GraphRAG-Chat", layout="wide")

# --- Conversation History State ---
if "history" not in st.session_state:
    st.session_state["history"] = []
# Model state is now managed in the Model & Retrieval section

# --- Ingestion Progress State ---
if "ingest_status" not in st.session_state:
    st.session_state["ingest_status"] = None
if "ingest_doc_name" not in st.session_state:
    st.session_state["ingest_doc_name"] = None

# --- Sidebar Organization ---
with st.sidebar:
    st.title("Settings")
    
    # Document Upload Section (always visible)
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload files (PDF, DOCX, TXT, MD, CSV)",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=False,
        help="Upload a document to start asking questions"
    )
    
    # Document Processing Section
    with st.expander("⚙️ Document Processing", expanded=True):
        st.subheader("Chunking Options")
        if "chunk_strategy" not in st.session_state:
            st.session_state["chunk_strategy"] = "content-aware"
        if "chunk_size" not in st.session_state:
            st.session_state["chunk_size"] = 300
        if "overlap" not in st.session_state:
            st.session_state["overlap"] = 1
            
        st.session_state["chunk_strategy"] = st.selectbox(
            "Chunking strategy",
            ["content-aware", "char", "word", "paragraph"],
            index=0,
            help="How to split text into chunks. 'content-aware' is recommended for better context."
        )
        
        if st.session_state["chunk_strategy"] == "content-aware":
            st.caption("ℹ️ Content-aware chunking preserves semantic context.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["chunk_size"] = st.number_input(
                "Chunk size", 
                min_value=128, 
                max_value=4096, 
                value=st.session_state["chunk_size"],
                step=64,
                help="Size of each chunk in characters"
            )
        with col2:
            st.session_state["overlap"] = st.number_input(
                "Overlap", 
                min_value=0, 
                max_value=1024, 
                value=st.session_state["overlap"],
                step=8,
                help="Overlap between chunks"
            )
    
    # Model & Retrieval Settings
    with st.expander("🧠 Model & Retrieval", expanded=False):
        st.subheader("Model Selection")
        model_options = [
            "gemma3:1b",  # Set as first to be default
            "llama3.1:latest",
            "llama3.2:latest",
            "llama3:latest",
            "gemma3:27b",
            "mistral:latest",
            "qwen3:latest",
            "phi4:latest"
        ]
        
        # Initialize model state if not set
        if "model" not in st.session_state:
            st.session_state["model"] = model_options[0]
        elif st.session_state["model"] not in model_options:
            st.session_state["model"] = model_options[0]
            
        st.session_state["model"] = st.selectbox(
            "LLM Model",
            model_options,
            index=model_options.index(st.session_state["model"]),
            help="Model used for generating responses"
        )
        
        extractor_options = ["spaCy (fast, generic)"] + model_options
        if "extractor_model" not in st.session_state:
            st.session_state["extractor_model"] = extractor_options[0]
            
        st.session_state["extractor_model"] = st.selectbox(
            "Entity Extractor",
            extractor_options,
            index=extractor_options.index(st.session_state["extractor_model"]),
            help="Model for extracting entities and relationships"
        )
        
        st.subheader("Graph Retrieval")
        
        if "max_relationships" not in st.session_state:
            st.session_state["max_relationships"] = 10
        if "hops" not in st.session_state:
            st.session_state["hops"] = 1
            
        st.session_state["max_relationships"] = st.number_input(
            "Max relationships", 
            min_value=1, 
            max_value=100, 
            value=st.session_state["max_relationships"],
            step=1,
            help="Maximum relationships to retrieve"
        )
        st.session_state["hops"] = st.number_input(
            "Max hops", 
            min_value=1, 
            max_value=5, 
            value=st.session_state["hops"],
            step=1,
            help="Degrees of separation to traverse"
        )
    
    # Advanced Settings
    with st.expander("⚙️ Advanced", expanded=False):
        st.subheader("Advanced Options")
        
        if "verbose" not in st.session_state:
            st.session_state["verbose"] = False
            
        st.session_state["verbose"] = st.checkbox(
            "Verbose logging", 
            value=st.session_state["verbose"],
            help="Show detailed debug information"
        )
        
        if "repair_relations" not in st.session_state:
            st.session_state["repair_relations"] = False
            
        st.session_state["repair_relations"] = st.checkbox(
            "Repair malformed relations", 
            value=st.session_state["repair_relations"],
            help="Attempt to fix malformed relationships"
        )
        
        # Configure logging based on verbose setting
        import logging
        if st.session_state["verbose"]:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
            for noisy_logger in ("urllib3", "neo4j"):
                logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Import remaining required modules
from app.utils.parallel import run_vector_ingest
from app.core.serialization import save_ingested_document
from app.core.ingestion import extract_with_markitdown
from app.utils.status import get_backend_status
import tempfile
import time

# --- Handle File Upload and Ingestion ---
# Allow ingestion when a new file is uploaded, regardless of previous ingestion status.
if "last_uploaded_file_name" not in st.session_state:
    st.session_state["last_uploaded_file_name"] = None
if (
    uploaded_file is not None
    and st.session_state["last_uploaded_file_name"] != uploaded_file.name
):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded_file.name.split(".")[-1]) as tf:
        tf.write(uploaded_file.read())
        temp_path = tf.name
    # Extract & chunk
    doc = extract_with_markitdown(
        Path(temp_path),
        chunk_strategy=st.session_state["chunk_strategy"],
        chunk_size=st.session_state["chunk_size"],
        overlap=st.session_state["overlap"]
    )
    # Set ingest_doc_name to the doc.metadata.filename (temp name used in ingestion)
    st.session_state["ingest_doc_name"] = doc.metadata.filename
    st.session_state["ingest_status"] = {"vector": "processing", "graph": "waiting"}
    st.session_state["last_uploaded_file_name"] = uploaded_file.name
    st.sidebar.info(f"Ingesting {uploaded_file.name}...")
    # Save chunks to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".chunks.json") as tf2:
        save_ingested_document(doc, tf2.name)
        chunks_path = tf2.name
    # Start vector ingestion (blocking)
    run_vector_ingest(chunks_path, doc.metadata.filename)
    st.session_state["ingest_status"]["vector"] = "ready"
    st.sidebar.success("ChromaDB (vector) ingestion complete.")
    # Run graph ingestion with the selected extractor
    from app.utils.parallel import run_graph_ingest
    
    # Determine extractor type and model
    if st.session_state["extractor_model"] == "spaCy (fast, generic)":
        entity_extractor = "spacy"
        extractor_model = ""  # Not used for spacy
    else:
        entity_extractor = "llama"
        extractor_model = st.session_state["extractor_model"]
    
    # Run the graph ingestion
    run_graph_ingest(
        chunks_path=chunks_path,
        doc_id=doc.metadata.filename,
        extractor_model=extractor_model,
        entity_extractor=entity_extractor
    )
    
    st.session_state["ingest_status"]["graph"] = "ready"
    st.sidebar.success("Neo4j (graph) ingestion complete.")
    
    # Clean up temp files
    try:
        os.remove(temp_path)
        os.remove(chunks_path)
    except Exception as e:
        logging.warning(f"Error cleaning up temp files: {e}")
        
    st.sidebar.info("Ingestion finished. You can now ask questions.")

# --- Main Area: Chat ---
st.title(":speech_balloon: GraphRAG-Chat")
st.caption("A hybrid RAG assistant for intelligent document querying.")

# --- Display Chunk Summaries ---
doc_name = st.session_state.get("ingest_doc_name")
if doc_name and st.session_state.get("ingest_status", {}).get("graph") == "ready":
    from app.core.serialization import load_ingested_document
    import os
    # Try to find the chunked document file if it exists
    chunks_file = None
    for f in os.listdir("."):
        if doc_name in f and f.endswith(".json"):
            chunks_file = f
            break
    if chunks_file:
        doc = load_ingested_document(chunks_file)
        st.subheader("Document Chunk Summaries")
        for idx, chunk in enumerate(doc.chunks):
            summary = getattr(chunk, "summary", None)
            st.markdown(f"**Chunk {idx}:** {summary if summary else '[No summary]'}")
    else:
        st.info("No chunked document file found for displaying summaries.")

# --- Status Bar at Top of Main Content ---
status_bar = st.container()
with status_bar:
    st.markdown("""
    <style>
        /* Status bar styling */
        .status-bar {
            position: fixed;
            top: 0;
            right: 0;
            left: 0;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(5px);
            padding: 8px 20px;
            border-bottom: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
        }
        .status-progress {
            flex-grow: 1;
            margin: 0 15px;
        }
        .status-message {
            font-size: 0.85rem;
            color: #4b5563;
            white-space: nowrap;
            margin-left: 10px;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-left: 8px;
        }
        .status-vector { background-color: #dbeafe; color: #1e40af; }
        .status-graph { background-color: #d1fae5; color: #065f46; }
        .status-waiting { background-color: #f3f4f6; color: #4b5563; }
        .status-processing { background-color: #fef3c7; color: #92400e; }
        .status-ready { background-color: #dcfce7; color: #166534; }
    </style>
    """, unsafe_allow_html=True)
    
    # Only show if there's an active ingestion
    if st.session_state["ingest_doc_name"] is not None:
        status = get_backend_status(st.session_state["ingest_doc_name"])
        vector_status = status.get("vector", "waiting")
        graph_status = status.get("graph", "waiting")
        
        # Calculate progress
        progress = 0
        if vector_status == "ready":
            progress += 0.5
        if graph_status == "ready":
            progress += 0.5
            
        # Show status bar
        st.markdown(f"""
        <div class="status-bar">
            <div class="status-message">
                Document: <strong>{st.session_state["ingest_doc_name"]}</strong>
                <span class="status-badge status-vector">Vector: {vector_status}</span>
                <span class="status-badge status-graph">Graph: {graph_status}</span>
            </div>
        </div>
        <script>
            // Add padding to main content to prevent overlap with fixed status bar
            document.addEventListener('DOMContentLoaded', function() {{
                const mainContent = document.querySelector('.main .block-container');
                if (mainContent) {{
                    mainContent.style.paddingTop = '50px';
                }}
            }});
        </script>
        """, unsafe_allow_html=True)
        
        # Show progress bar only when processing
        if progress < 1.0:
            st.progress(progress, text=f"Processing document: {int(progress * 100)}% complete")
        else:
            st.success("Ingestion complete. You can now ask questions.")

# --- Ingestion Progress Bar (move to main area) ---
ingest_progress_placeholder = st.empty()
if st.session_state["ingest_doc_name"]:
    status = get_backend_status(st.session_state["ingest_doc_name"])
    progress = 0
    if status.get("vector") == "ready":
        progress += 0.5
    if status.get("graph") == "ready":
        progress += 0.5
    with ingest_progress_placeholder.container():
        st.progress(progress, text=f"Vector: {status.get('vector', 'waiting').capitalize()} | Graph: {status.get('graph', 'waiting').capitalize()}")
        if progress < 1.0:
            st.info("Ingestion in progress. Please wait...")
        else:
            st.success("Ingestion complete. You can now ask questions.")

# # --- Model Selector ---
# st.sidebar.header("Model Selector")
# st.session_state["model"] = st.sidebar.selectbox(
#     "Choose LLM model:", model_options, index=model_options.index(st.session_state["model"])
# )

st.sidebar.markdown("---")
st.sidebar.header("Conversation History")
if st.session_state["history"]:
    for idx, turn in enumerate(reversed(st.session_state["history"])):
        st.sidebar.markdown(f"<div style='font-size:13px; color:#555;'><b>You:</b> {turn['question']}</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='font-size:13px; color:#2b6cb0;'><b>Assistant:</b> {turn['answer']}</div>", unsafe_allow_html=True)
        st.sidebar.write("---")
    if st.sidebar.button("Clear Conversation", key="clear_history"):
        st.session_state["history"] = []
        st.rerun()
else:
    st.sidebar.info("No conversation yet.")

from app.routing import main_rag_pipeline

# Chat interface styles
st.markdown("""
    <style>
        /* Reset some default Streamlit styles */
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        
        /* Welcome message */
        .welcome-message {
            text-align: center;
            color: #4b5563;
            margin: 3rem 0;
            padding: 0 1rem;
        }
        
        .welcome-message h2 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        
        .welcome-message p {
            font-size: 1.125rem;
            color: #6b7280;
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Message bubbles */
        .message {
            margin: 1rem 0;
            max-width: 80%;
            clear: both;
        }
        
        .user-message {
            float: right;
            margin-left: 20%;
        }
        
        .assistant-message {
            float: left;
            margin-right: 20%;
        }
        
        .message-bubble {
            padding: 0.75rem 1.25rem;
            border-radius: 1.25rem;
            line-height: 1.5;
            word-wrap: break-word;
            font-size: 0.95rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .user-bubble {
            background-color: #3b82f6;
            color: white;
            border-bottom-right-radius: 0.5rem;
        }
        
        .assistant-bubble {
            background-color: #f3f4f6;
            color: #111827;
            border-bottom-left-radius: 0.5rem;
        }
        
        .timestamp {
            font-size: 0.75rem;
            color: #9ca3af;
            margin: 0.25rem 0.75rem 0.5rem;
            text-align: right;
            clear: both;
        }
        
        /* Code blocks */
        pre {
            background-color: #1f2937;
            color: #f9fafb;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.875em;
            margin: 0.75rem 0;
            line-height: 1.5;
        }
        
        /* Input area */
        .stTextInput > div > div > input {
            border-radius: 1.5rem !important;
            padding: 0.875rem 1.25rem !important;
            font-size: 0.95rem !important;
            border: 1px solid #d1d5db !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Make sure content is visible above input */
        .main > div {
            padding-bottom: 6rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main content area
st.markdown("""
    <style>
        .stApp {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .main > div {
            flex: 1 0 auto;
        }
        .stTextInput {
            flex-shrink: 0;
        }
    </style>
""", unsafe_allow_html=True)

# Main chat container
with st.container():
    st.markdown("<div id='chat-container'>", unsafe_allow_html=True)
    
    # Welcome message (only shown when there's no history)
    if not st.session_state["history"]:
        st.markdown("""
            <div class="welcome-message">
                <h2>Welcome to GraphRAG Assistant</h2>
                <p>Upload a document using the sidebar and start asking questions about its content.</p>
            </div>
        """, unsafe_allow_html=True)

# Show all messages in the conversation
for turn in st.session_state["history"]:
    # Format timestamp
    timestamp = datetime.now().strftime("%H:%M")
    
    # User message
    st.markdown(
        f"""
        <div class="message user-message">
            <div class="message-bubble user-bubble">
                {turn['question']}
            </div>
            <div class="timestamp">{timestamp}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Format answer with proper line breaks and code blocks
    answer = turn['answer'].replace('```', '\n```\n').replace('\n\n', '\n')
    answer_html = ''
    in_code_block = False
    
    for line in answer.split('\n'):
        if line.startswith('```'):
            if in_code_block:
                answer_html += '</pre>'
            else:
                answer_html += '<pre>'
            in_code_block = not in_code_block
        else:
            answer_html += f"{line}<br>" if not in_code_block else f"{line}\n"
    
    if in_code_block:  # In case the code block wasn't properly closed
        answer_html += '</pre>'
    
    # Assistant message
    st.markdown(
        f"""
        <div class="message assistant-message">
            <div class="message-bubble assistant-bubble">
                {answer_html}
            </div>
            <div class="timestamp">{timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat container

# Auto-scroll to bottom when new messages are added
st.markdown("""
    <script>
        // Scroll to bottom when page loads
        window.onload = function() {
            var container = document.getElementById('chat-container');
            if (container) container.scrollTop = container.scrollHeight;
        };
        
        // Also scroll when the page is fully loaded and after Streamlit updates
        document.addEventListener('DOMContentLoaded', function() {
            var container = document.getElementById('chat-container');
            if (container) container.scrollTop = container.scrollHeight;
        });
    </script>
""", unsafe_allow_html=True)

# User input area
# Sticky input area at the bottom
st.markdown("""
    <style>
        /* Input area styling */
        .stTextInput {
            position: sticky;
            bottom: 0;
            background: white;
            padding: 16px 0;
            border-top: 1px solid #e2e8f0;
            z-index: 100;
            margin: 0;
        }
        
        /* Ensure proper spacing */
        .stTextInput > div {
            margin-bottom: 0;
        }
        
        /* Fix for Streamlit's default padding */
        .main > div {
            padding-bottom: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

with st.form("chat_form"):
    doc_name = st.session_state.get("ingest_doc_name")
    user_input_disabled = False
    if doc_name:
        user_input_disabled = get_backend_status(doc_name).get("graph") != "ready"
    
    # Create a two-column layout for input and button
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask anything...", 
            "", 
            key="user_input", 
            disabled=user_input_disabled,
            label_visibility="collapsed",
            placeholder="Type your question here...",
        )
    
    with col2:
        submit_button = st.form_submit_button(
            "Send",
            type="primary",
            disabled=user_input_disabled,
            use_container_width=True,
            help="Press Enter to send"
        )
    
    if user_input_disabled:
        st.warning(" Please wait for document ingestion to complete before asking questions.")
        
    # Add some space at the bottom
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
if submit_button and user_input.strip() and not user_input_disabled:
    with st.spinner("Processing query..."):
        # Pass graph retrieval options to main_rag_pipeline
        result = main_rag_pipeline(
            user_input,
            llm_model=st.session_state["model"],
            max_relationships=st.session_state["max_relationships"],
            hops=st.session_state["hops"]
        )
        answer = result.get("answer", "[No answer returned]")
        routing = result.get("routing_decision", None)
        # Add to conversation history with timestamp
        st.session_state["history"].append({
            "question": user_input,
            "answer": answer,
            "routing": routing,
        })
        # Show answer and routing reasoning in a slick way
        st.markdown(
            f"<div style='background-color:#f8fafc; border-radius:8px; padding:18px 16px 10px 16px; margin-bottom:10px; box-shadow:0 1px 4px #e2e8f0;'>"
            f"<span style='font-size:15px; color:#555;'><b>You:</b> {user_input}</span><br>"
            f"<span style='font-size:15px; color:#2b6cb0;'><b>Assistant:</b> {answer}</span>"
            f"</div>", unsafe_allow_html=True
        )
        if routing:
            st.markdown(
                f"<div style='background-color:#e0f7fa; border-radius:8px; padding:12px 14px 8px 14px; margin-bottom:10px; border-left: 4px solid #00bcd4;'>"
                f"<span style='font-size:13px; color:#006064;'><b>Routing Decision:</b> {getattr(routing, 'backend', routing)}<br>"
                f"<b>Reason:</b> {getattr(routing, 'reason', routing)}<br>"
                f"<b>Entity Match:</b> {getattr(routing, 'entity_match', routing)}<br>"
                f"<b>Entity Name:</b> {getattr(routing, 'entity_name', routing)}</span>"
                f"</div>", unsafe_allow_html=True
            )
        with st.expander("Show routing details and backend results"):
            st.json({
                "routing_decision": str(result.get("routing_decision")),
                "graph_result": str(result.get("graph_result")),
                "vector_result": str(result.get("vector_result")),
                "raw_vector_chunks": result.get("raw_vector_chunks")
            })

# Footer
st.markdown("---")
st.caption("GraphRAG Assistant v1.0 | Ready")

# Add some space at the bottom to prevent content from being hidden behind the input
st.markdown("""
    <style>
        .main > div:last-child {
            padding-bottom: 100px !important;
        }
    </style>
""", unsafe_allow_html=True)
