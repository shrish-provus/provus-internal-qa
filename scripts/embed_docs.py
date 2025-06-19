import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# === CONFIGURATION ===
DATA_DIR = "./data"
DB_DIR = "./db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === STEP 1: Read and Parse Docs ===
docs = []

for filepath in glob.glob(os.path.join(DATA_DIR, "docs", "*.txt")):
    print(f"📄 Loading from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    for section in raw_text.split("========================"):
        lines = section.strip().splitlines()
        title, url = "Untitled", "Unknown"
        content_lines = []

        for line in lines:
            if line.startswith("TITLE: "):
                title = line.replace("TITLE: ", "").strip()
            elif line.startswith("URL: "):
                url = line.replace("URL: ", "").strip()
            else:
                content_lines.append(line)

        content = "\n".join(content_lines).strip()
        if content:
            docs.append(Document(page_content=content, metadata={"title": title, "source": url}))

# === STEP 2: Split into Chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = text_splitter.split_documents(docs)

# === STEP 3: Embed and Store ===
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=DB_DIR)
db.persist()

print("✅ Documents embedded and saved to ./db")
