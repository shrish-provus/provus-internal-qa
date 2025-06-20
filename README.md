# Provus Internal Q&A System

A sophisticated natural language question-answering system for internal documentation that provides accurate, source-attributed answers from your organization's knowledge base using advanced vector search and local LLM processing.

## 🚀 Overview

This project enables internal users to ask natural language questions and get accurate answers from:

- **Provus Documentation** (`docs.provusinc.com`)
- **Notion** pages shared with a connected integration
- **Local document collections** in various formats

The system uses advanced Retrieval-Augmented Generation (RAG) with ChromaDB's optimized HNSW indexing and local Ollama models to provide high-quality, source-attributed answers from your internal documentation.

## ✨ Key Features

- 🔍 **Advanced Vector Search**: ChromaDB with optimized HNSW indexing for fast, accurate retrieval
- 🤖 **Local LLM Processing**: Uses Ollama for privacy-focused answer generation
- 📊 **Multi-Strategy Search**: Combines similarity search and MMR (Maximum Marginal Relevance) for diverse results
- 🔄 **Smart Deduplication**: Advanced result ranking and deduplication across multiple search strategies
- 📚 **Source Attribution**: Detailed source tracking with relevance scores and URLs
- ⚡ **Performance Monitoring**: Built-in performance metrics and benchmarking
- 🖥️ **Interactive CLI**: Rich command-line interface with real-time statistics
- 🌐 **Optional Web UI**: Streamlit-based browser interface
- 📄 **Flexible Input**: Supports multiple document formats and sources

## 🏗️ Project Structure

```
provus-internal-qa/
├── data/
│   ├── docs/                    # Crawled documentation corpus
│   │   └── docs_corpus.txt      # Main documentation file
│   └── notion/                  # Notion pages (if used)
│
├── db/                          # ChromaDB vector database (auto-generated)
│   ├── chroma.sqlite3          # Vector index storage
│   └── [index files]           # HNSW index files
│
├── scripts/
│   ├── fetch_docs.py           # Document crawling script
│   ├── fetch_notion.py         # Notion API integration
│   ├── embed_docs.py           # Advanced indexing with ChromaDB
│   └── query_docs.py           # Interactive Q&A interface
│
├── venv/                       # Python virtual environment
├── app.py                      # Streamlit web interface
├── .env                        # Environment configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

## 🔧 Prerequisites

Before getting started, ensure you have:

- **Python 3.9+** installed
- **[Ollama](https://ollama.com/)** running locally with a model:
  ```bash
  # Install and run Ollama
  ollama pull llama3
  ollama serve
  ```
- **Notion integration** (optional) with API access
- **Chrome/Chromium** for web crawling (if using browser automation)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd provus-internal-qa

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Create `.env` file for Notion integration:

```env
NOTION_API_KEY=your-notion-api-key-here
```

### 3. Prepare Documentation

**Option A: Use existing corpus (if available)**
- Ensure `data/docs/docs_corpus.txt` exists with your documentation

**Option B: Crawl documentation**
```bash
python scripts/fetch_docs.py
```

**Option C: Add Notion pages**
```bash
# Edit fetch_notion.py with your page IDs first
python scripts/fetch_notion.py
```

### 4. Build Vector Database

```bash
python scripts/embed_docs.py
```

**Expected output:**
```
🚀 Advanced ChromaDB Indexing Setup
==================================================
🔧 Initializing ChromaDB with optimized settings...
📄 Loading documents...
📂 Found 3 file(s)
✅ Loaded 347 unique documents
✂️  Creating optimized chunks...
✅ Created 4139 unique chunks
🧠 Initializing embedding model...
🏗️  Creating optimized ChromaDB collection...
📊 Collection created with optimized HNSW settings
📥 Adding chunks in batches...
🎉 Indexing Complete!
📈 Performance Summary:
   • Total chunks: 4139
   • Indexing speed: 45.2 chunks/sec
   • Average chunk size: 380 chars
```

### 5. Start Querying!

```bash
python scripts/query_docs.py
```

## 💡 Usage Examples

### Interactive CLI Session

```
🔍 Advanced ChromaDB QA System Ready!
Commands:
  - Ask any question about your documentation
  - Type 'stats' to see database statistics  
  - Type 'test' to run performance tests
  - Type 'quit' to exit
------------------------------------------------------------

>>> How do I configure user permissions?

🔍 Advanced search: 'How do I configure user permissions?'
   📊 Similarity search: 4 results
   🎯 MMR search: 4 results  
   ⚡ Search completed in 0.123s
📋 Using 3 unique sources

💡 Answer:
To configure user permissions in Provus, you need to:

1. **Access the Admin Panel**: Navigate to Settings > User Management
2. **Select User Roles**: Choose from predefined roles (Admin, Manager, User, Viewer)
3. **Custom Permissions**: For granular control, use the Permission Matrix to set specific access rights
4. **Apply Changes**: Save and the permissions take effect immediately

For enterprise customers, you can also integrate with SAML/SSO providers for centralized permission management.

📚 Sources (3):
  1. User Management Guide (Score: 0.89)
     🔗 https://docs.provusinc.com/admin/user-permissions
     📊 Found via: similarity search
  2. Security Configuration (Score: 0.82)  
     🔗 https://docs.provusinc.com/security/access-control
     📊 Found via: mmr search
  3. Enterprise SSO Setup (Score: 0.78)
     🔗 https://docs.provusinc.com/enterprise/sso-integration
     📊 Found via: similarity search

⏱️  Performance:
   • Search: 0.123s
   • LLM: 2.45s
   • Total: 2.57s  
   • Sources: 3
   • Context: 1847 chars
```

### Advanced Commands

```bash
>>> stats
📊 Database Statistics:
   • Collection: provus_docs
   • Total vectors: 4139
   • Index metadata: {'hnsw:space': 'cosine', 'hnsw:M': 32, ...}

>>> test
🏃 Running performance tests...
   Test 1: 2.34s (4 sources)
   Test 2: 1.89s (3 sources)
   Test 3: 2.12s (5 sources)
   Test 4: 1.76s (2 sources)
   Test 5: 2.28s (4 sources)

📈 Performance Summary:
   • Average query time: 2.08s
   • Total test time: 10.39s
   • Queries per second: 0.5
```

## ⚙️ Advanced Configuration

### ChromaDB Optimization Settings

The system uses optimized HNSW (Hierarchical Navigable Small World) indexing:

```python
CHROMA_SETTINGS = {
    "hnsw:space": "cosine",              # Distance metric for text
    "hnsw:M": 32,                        # Connections per node (16-64)
    "hnsw:construction_ef": 400,         # Build-time candidate list size
    "hnsw:search_ef": 100,               # Search-time candidate list size
    "hnsw:batch_size": 500,              # Batch processing size
    "hnsw:sync_threshold": 2000,         # Disk sync threshold
    "hnsw:num_threads": 4,               # Parallel processing threads
    "hnsw:resize_factor": 1.2,           # Index growth factor
}
```

### Search Strategies

The system employs multiple search strategies:

1. **Similarity Search**: Traditional cosine similarity with score thresholding
2. **MMR (Maximum Marginal Relevance)**: Balances relevance and diversity
3. **Hybrid Search**: Combines both strategies with intelligent deduplication

### Chunking Configuration

```python
CHUNK_SIZE = 400          # Optimal for retrieval performance
CHUNK_OVERLAP = 40        # Maintains context continuity
MIN_CHUNK_SIZE = 50       # Filters out tiny fragments
```

## 🌐 Web Interface

Launch the Streamlit web UI:

```bash
streamlit run app.py
```

Access at `http://localhost:8501` for a browser-based interface with:
- Interactive chat interface
- Source document viewer
- Performance metrics dashboard
- Search result visualization

## 🔧 Troubleshooting

### Common Issues and Solutions

**ChromaDB Parameter Errors**
```
Error: Invalid HNSW parameters: unknown field 'hnsw:ef'
```
- **Solution**: Use correct parameter names (`hnsw:search_ef` instead of `hnsw:ef`)
- **Fixed in**: Updated `embed_docs.py` script

**LangChain Deprecation Warnings**
```
LangChainDeprecationWarning: The class `SentenceTransformerEmbeddings` was deprecated
```
- **Impact**: Functionality still works, warning only
- **Solution**: Consider upgrading to `langchain-huggingface` package

**Ollama Connection Issues**
```
Error: Ollama not responding
```
- **Check**: `ollama serve` is running
- **Verify**: Model is pulled (`ollama pull llama3`)
- **Test**: `curl http://localhost:11434` should respond

**Empty Search Results**
- **Check**: Vector database exists in `./db/`
- **Verify**: Documents were properly indexed (check console output)
- **Test**: Run `stats` command to see vector count

**Performance Issues**
- **Large datasets**: Increase `hnsw:search_ef` for better recall
- **Slow searches**: Decrease `k` parameter or increase score threshold
- **Memory usage**: Reduce `CHUNK_SIZE` or batch size

### Performance Optimization

For large document collections:

1. **Increase batch size**: `"hnsw:batch_size": 1000`
2. **Tune search parameters**: Higher `hnsw:search_ef` for better results
3. **Optimize chunking**: Adjust `CHUNK_SIZE` based on document types
4. **Use SSD storage**: Place `./db/` on fast storage for better I/O

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   ChromaDB       │    │   Ollama LLM    │
│   Sources       │───▶│   Vector Store   │───▶│   (llama3)      │
│                 │    │   (HNSW Index)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  • Provus Docs  │    │  • Similarity    │    │  • Context      │
│  • Notion Pages │    │  • MMR Search    │    │  • Generation   │
│  • Local Files  │    │  • Deduplication │    │  • Attribution  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the existing code style and patterns
4. Add tests for new functionality
5. Update documentation as needed
6. Submit pull request with detailed description

## 📝 Document Format Requirements

For proper indexing, documents should follow this format:

```
========================
TITLE: Document Title Here
URL: https://source-url.com/page
Content goes here...
More content...
========================
TITLE: Next Document Title
URL: https://another-url.com/page
Next document content...
```

## 🔒 Privacy & Security

- **Local Processing**: All LLM inference happens locally via Ollama
- **No Data Transmission**: Documents never leave your environment
- **Secure Storage**: Vector embeddings stored locally in ChromaDB
- **Access Control**: Configure based on your internal security policies

## 📈 Performance Benchmarks

Typical performance on modern hardware:

- **Indexing Speed**: 40-60 chunks/second
- **Search Latency**: 100-200ms for similarity search
- **Answer Generation**: 2-4 seconds (depends on LLM model)
- **Memory Usage**: ~500MB for 10K document chunks
- **Storage**: ~100MB vector database for 10K chunks

## 🔄 Version History

- **v1.2**: Advanced ChromaDB integration with HNSW optimization
- **v1.1**: Multi-strategy search with MMR and deduplication
- **v1.0**: Basic RAG implementation with ChromaDB and Ollama

---

For additional support or feature requests, please open an issue in the repository.