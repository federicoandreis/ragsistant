# RAGsistant

A production-ready hybrid RAG (Retrieval-Augmented Generation) system that combines Neo4j graph database and ChromaDB vector store for intelligent document querying. Features a clean Streamlit interface and modular architecture for easy extension.

> **Latest Updates**
> - 🛡️ Enhanced security with required environment variables for database credentials
> - 🗃️ Implemented robust ChromaDB and Neo4j connectors with comprehensive error handling
> - 📊 Added detailed configuration options in `.env.example` with clear documentation
> - 🧪 Improved test coverage and code quality

## 🚀 Features

- **Hybrid Storage**: Combines graph (Neo4j) and vector (ChromaDB) databases for optimal retrieval
- **Document Processing**: Supports PDF, DOCX, TXT, Markdown, and CSV files
- **Intelligent Routing**: Pattern-based and LLM-based query routing
- **Entity Extraction**: Multiple extraction methods (spaCy, LLM-based)
- **Graph Enhancement**: Entity coalescing, relation repair, and community detection
- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Production Ready**: Comprehensive configuration, logging, and error handling

## 📋 Prerequisites

- Python 3.9 or higher
- Neo4j (local installation or Docker) **with the Graph Data Science (GDS) plugin**
- Ollama (for local LLM inference)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ragsistant.git
cd ragsistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Setup Configuration
```bash
# Copy and edit the environment variables
cp .env.example .env

# Important: Set your Neo4j password in the .env file
# NEO4J_PASSWORD is required and has no default for security
```

### 6. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### 7. Start Required Services

#### Neo4j
> **IMPORTANT:** RAGsistant requires the Neo4j Graph Data Science (GDS) module. Make sure your Neo4j instance has GDS installed and enabled.

```bash
# Using Docker (with GDS)
docker run --name neo4j-gds \
    --publish 7474:7474 \
    --publish 7687:7687 \
    --env NEO4J_AUTH=neo4j/test1234 \
    --env NEO4J_PLUGINS='["graph-data-science"]' \
    neo4j:latest
```

# Local install (with GDS)
- Download Neo4j and the Graph Data Science plugin from the official Neo4j website: https://neo4j.com/download-center/#graph-data-science
- Follow Neo4j's instructions to place the GDS JAR in the plugins directory and enable it in your config.


# Or start local Neo4j installation
```

#### Ollama
```bash
# Install Ollama from https://ollama.ai
ollama pull gemma3:1b
ollama pull mxbai-embed-large

# Verify the models are available
ollama list
```

### 8. Initialize the Application
Before first use, you'll need to initialize the application with:

```bash
# This will set up the necessary database schemas and indexes
python -m app.init
```

## 🚀 Quick Start

### 1. Start the Application
```bash
streamlit run app/ui.py
```

### 2. Upload Documents
- Navigate to the web interface (usually http://localhost:8501)
- Use the file uploader to add documents
- Wait for processing to complete

### 3. Query Your Documents
- Enter questions in the chat interface
- The system will automatically route queries to the best retrieval method
- View transparent results showing sources and reasoning

## 📁 Project Structure

```
ragsistant/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package configuration
├── .env.example             # Configuration template
├── .gitignore              # Git ignore rules
├── prd.txt                 # Product requirements
├── app/                    # Main application
│   ├── __init__.py
│   ├── ui.py              # Streamlit interface
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models
│   ├── routing.py         # Query routing logic
│   ├── embedding.py       # Embedding functionality
│   ├── vectorstore.py     # Vector store operations
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   ├── ingestion.py   # Document ingestion
│   │   ├── chunking.py    # Text chunking
│   │   └── serialization.py # Data serialization
│   ├── db/                # Database connectors
│   │   ├── __init__.py
│   │   ├── connections.py # Database connections
│   │   ├── chroma.py     # ChromaDB operations
│   │   └── neo4j.py      # Neo4j operations
│   ├── entity_extraction/ # Entity extraction
│   │   ├── __init__.py
│   │   ├── base.py       # Base extractor
│   │   ├── spacy_extractor.py
│   │   └── llm_extractor.py
│   ├── graph/             # Graph operations
│   │   ├── __init__.py
│   │   ├── ingestion_core.py
│   │   ├── retrieval.py
│   │   ├── entity_coalescing.py
│   │   ├── relation_repair.py
│   │   ├── community_detection.py
│   │   └── gap_filling.py
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── status.py     # Status tracking
│       └── parallel.py   # Parallel processing
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_chunking.py
│   ├── test_graph_ingest.py
│   ├── test_graph_retrieval.py
│   ├── test_routing.py
│   └── test_vectorstore.py
├── docs/                  # Documentation
├── data/                  # Data storage
│   ├── documents/         # Source documents
│   └── chunks/           # Processed chunks
└── storage/              # Database storage
    ├── chroma_db/        # ChromaDB files
    └── neo4j_data/       # Neo4j data
```

## ⚙️ Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

### Database Settings
- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USERNAME/PASSWORD`: Neo4j credentials
- `CHROMA_DB_PATH`: ChromaDB storage path

### Model Settings
- `OLLAMA_BASE_URL`: Ollama server URL
- `OLLAMA_MODEL`: Default LLM model
- `EMBEDDING_MODEL`: Embedding model name

### Processing Settings
- `DEFAULT_CHUNK_SIZE`: Text chunk size
- `DEFAULT_CHUNK_OVERLAP`: Chunk overlap size

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_chunking.py -v
```

## 🔧 Development

### Code Style
```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

### Adding New Features
1. Create feature branch
2. Add tests in `tests/`
3. Implement functionality in appropriate module
4. Update documentation
5. Submit pull request

## 📊 Architecture

RAGsistant uses a hybrid approach:

1. **Document Ingestion**: Files are processed, chunked, and stored
2. **Dual Storage**: Text chunks go to ChromaDB, entities/relations to Neo4j
3. **Query Routing**: Intelligent routing based on query type
4. **Retrieval**: Vector similarity and graph traversal
5. **Response Generation**: LLM synthesis with source attribution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check the [documentation](docs/)
- Review [issues](https://github.com/your-username/ragsistant/issues)
- Join our [discussions](https://github.com/your-username/ragsistant/discussions)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Neo4j](https://neo4j.com/) and [ChromaDB](https://www.trychroma.com/)
- LLM inference via [Ollama](https://ollama.ai/)
