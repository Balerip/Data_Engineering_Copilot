# Data Engineering Copilot 

A Retrieval-Augmented Generation (RAG) application that provides intelligent assistance for data engineering questions, specifically focused on Apache Spark, dbt (data build tool), and Apache Airflow documentation.

## Overview

This application creates an AI-powered copilot that can answer questions about specific data engineering tools by crawling and indexing their documentation. It uses a combination of web scraping, vector embeddings, and a local LLM to provide accurate, documentation-based responses.

### Key Features
- **Web Crawling**: Automatically crawls and indexes documentation from specified URLs
- **Vector Search**: Uses ChromaDB for efficient semantic search across indexed documents
- **Local LLM Integration**: Leverages Ollama with Llama3 for response generation
- **Chat Memory**: Maintains conversation history for each user
- **Streamlit UI**: User-friendly web interface for interactions
- **Strict Documentation Mode**: Answers only from indexed documentation, preventing hallucinations

## Architecture

```
├── Rag_agent/
│   ├── services/
│   │   ├── chat_service.py      # Service layer for chat functionality
│   │   ├── data_loader.py       # Web scraping and document loading
│   │   ├── index.py             # Vector index creation and management
│   │   └── simple_agent.py      # ReAct agent implementation
│   ├── dataStorage/              # Scraped documentation storage
│   └── indexStorage/             # ChromaDB vector index storage
├── streamlit.py                        # Streamlit web application
└── chat_memory_*.json           # User-specific chat histories
```

## 🔧 Prerequisites

### System Requirements
- Python 3.8+
- 8GB+ RAM (for embedding models and vector operations)
- 5GB+ free disk space

### Required Services
- **Ollama**: Local LLM runtime
  ```bash
  # Install Ollama (macOS/Linux)
  curl -fsSL https://ollama.com/install.sh | sh
  
  # Pull Llama3 model
  ollama pull llama3
  ```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data-engineering-copilot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create required directories**
   ```bash
   mkdir -p Rag_agent/dataStorage
   mkdir -p Rag_agent/indexStorage
   ```

## Usage

### Running the Application

1. **Start Ollama service** (if not already running)
   ```bash
   ollama serve
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run streamlit.py
   ```



### Using the Copilot

1. Enter a User ID (for chat history tracking)
2. Type your question about Spark, dbt, or Airflow
3. Click "Run Copilot" to get documentation-based answers

### Example Queries
- "What is Apache Spark?"
- "How do I create a DAG in Airflow?"
- "Explain dbt models and references"
- "Generate 5 quiz questions about Spark SQL"

## Configuration

### Customizing Documentation Sources

Edit the URLs in `simple_agent.py`:

```python
self.urls = [
    "https://spark.apache.org/docs/latest/sql-getting-started.html",
    "https://docs.getdbt.com/reference/references-overview",
    # Add more URLs here
]
```

### Crawling Settings

Adjust crawling depth and page limits in `simple_agent.py`:

```python
self.index = Index(directory, storage_directory).load_index(
    urls=self.urls,
    crawl_depth=1,      # How deep to crawl links
    max_pages=50        # Maximum pages to index
)
```

### Model Configuration

**Embedding Model** (in `index.py`):
```python
self.embedding = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # Lightweight model
    embed_batch_size=16,
    max_length=512
)
```

**LLM Settings** (in `simple_agent.py`):
```python
llm = Ollama(
    model="llama3",
    request_timeout=120.0,
    temperature=0.0  # Set to 0 for deterministic responses
)
```

## Features in Detail

### Web Crawler (`data_loader.py`)
- Crawls websites to specified depth
- Respects same-domain restriction
- Converts HTML to clean text documents
- Handles pagination and link extraction

### Vector Index (`index.py`)
- Uses ChromaDB for persistent vector storage
- Semantic chunking for better context preservation
- Efficient similarity search with cosine distance
- Automatic index persistence and loading

### ReAct Agent (`simple_agent.py`)
- Strict documentation-only mode prevents hallucinations
- Tool-based architecture for extensibility
- Context-aware responses with source attribution
- Fallback handling for out-of-scope questions

### Chat Memory
- JSON-based persistent storage
- User-specific conversation histories
- Clear history option for fresh starts

## Troubleshooting

### Common Issues

1. **"Ollama not found" error**
   - Ensure Ollama is installed and running
   - Check if Llama3 model is pulled: `ollama list`

2. **Out of memory errors**
   - Reduce `embed_batch_size` in index.py
   - Use smaller embedding model
   - Reduce `max_pages` for crawling

3. **Slow responses**
   - Reduce `similarity_top_k` in query engine
   - Ensure Ollama has sufficient resources
   - Consider using GPU acceleration

4. **Index not found**
   - Check if dataStorage and indexStorage directories exist
   - Verify URLs are accessible
   - Check crawling logs for errors

## Development

### Adding New Documentation Sources

1. Add URLs to the `self.urls` list
2. Clear existing index: `rm -rf Rag_agent/indexStorage/*`
3. Restart the application to re-index

### Extending Functionality

- Add new tools to the ReAct agent
- Implement additional response modes
- Add support for more document formats
- Integrate external APIs or databases

## Performance Considerations

- **Initial Indexing**: First run takes 5-10 minutes to crawl and index
- **Query Response**: Typical response time 3-10 seconds
- **Memory Usage**: ~2-4GB during operation
- **Storage**: ~100MB per 50 indexed pages

## Security Notes

- Chat histories are stored locally in plain JSON
- No authentication implemented by default
- Crawling respects robots.txt is not implemented
- Consider adding rate limiting for production use

