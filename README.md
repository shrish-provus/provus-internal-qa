# Provus Internal Q&A System

A natural language question-answering system for internal documentation that provides concise, accurate answers sourced from the organization's knowledge base.

##  Overview

This project enables internal users to ask natural language questions and get accurate answers from:

- **Provus Docs** (`docs.provusinc.com`)
- **Notion** pages shared with a connected integration

The system uses Retrieval-Augmented Generation (RAG) with a local Ollama model to provide high-quality, source-attributed answers from your internal documentation.

##  Features

-  **Automatic crawling** and parsing of internal docs site
-  **Notion API integration** to ingest selected pages
-  **Vector database** embedding with Chroma
-  **Command-line interface** for quick queries
-  **Source attribution** with URL/title references
-  **Optional Streamlit UI** for browser-based access

##  Project Structure

```
provus-internal-qa/
├── data/
│   ├── docs/                    # Crawled text from docs.provusinc.com
│   └── notion/                  # Exported Notion pages (.txt format)
│
├── db/                          # Chroma vector database (auto-generated)
│
├── scripts/
│   ├── fetch_docs_site.py       # Login and crawl Provus docs
│   ├── fetch_notion.py          # Download Notion pages via API
│   ├── embed_docs.py            # Chunk, embed, and store into Chroma DB
│   └── query_docs.py            # Interactive CLI question interface
│
├── app.py                       # Optional Streamlit web UI
├── .env                         # Environment variables (not tracked)
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

##  Prerequisites

Before getting started, ensure you have:

- **Python 3.9+** installed
- **[Ollama](https://ollama.com/)** running locally with a model (e.g., `ollama run llama3`)
- **Notion integration** created with access to your pages
- **Chrome dependencies** for Playwright web automation

##  Setup Instructions

### 1. Clone and Setup Environment

```bash
git clone https://github.com/your-org/provus-internal-qa.git
cd provus-internal-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
playwright install
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
NOTION_API_KEY=your-notion-api-key-here
```

> **Note:** Get your Notion API key from the [Notion Developers](https://developers.notion.com/) page.

##  Usage Guide

### Step 1: Crawl Provus Documentation

```bash
python scripts/fetch_docs_site.py
```

This script will:
- Log into `docs.provusinc.com` using headless browser automation
- Recursively crawl and save all documentation
- Output content to `data/docs/docs_corpus.txt`

### Step 2: Fetch Notion Pages (Optional)

1. Edit `scripts/fetch_notion.py` and update the page IDs:
   ```python
   PAGE_IDS = ["your-page-id-1", "your-page-id-2"]
   ```

2. Run the script:
   ```bash
   python scripts/fetch_notion.py
   ```

This saves Notion page content into the `data/notion/` directory.

### Step 3: Build Vector Database

```bash
python scripts/embed_docs.py
```

This script will:
- Parse text files from `data/docs/` and `data/notion/`
- Chunk documents into manageable pieces
- Generate embeddings and store in ChromaDB (`./db/`)

### Step 4: Start Asking Questions!

```bash
python scripts/query_docs.py
```

**Example interaction:**
```
>>> How do I create a quote from an estimate template?

Answer:
To create a quote from an estimate template, follow these steps:
1. Navigate to the Estimates section in your dashboard
2. Select the template you want to use as a base
3. Click "Convert to Quote" button
4. Review and modify the details as needed
5. Save your new quote

 Sources:
• https://docs.provusinc.com/docs/create-quote-from-quote-template
• https://provus.notion.site/Quote-Management-abc123...
```

##  Web Interface (Optional)

For a browser-based interface, launch the Streamlit app:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the web UI.

##  Configuration Notes

- **Document Format:** Files must include `TITLE:` and `URL:` headers for proper source attribution
- **Notion Access:** Ensure your Notion integration has access to the pages you want to index
- **Browser Automation:** `fetch_docs_site.py` uses Playwright to handle login flows automatically
- **Model Requirements:** The system expects Ollama to be running with a compatible model

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Troubleshooting

**Common Issues:**

- **Ollama not responding:** Ensure Ollama is running (`ollama serve`) and a model is loaded
- **Playwright errors:** Run `playwright install` to ensure browser dependencies are installed
- **Notion API errors:** Verify your API key and page permissions in the Notion integration settings
- **Empty results:** Check that documents were properly crawled and embedded by examining the `data/` directories
