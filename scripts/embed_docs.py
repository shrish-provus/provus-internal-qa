import os
import glob
import hashlib
import time
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings

# === CONFIGURATION ===
DATA_DIR = "./data"
DB_DIR = "./db"
COLLECTION_NAME = "provus_docs"
CHUNK_SIZE = 400  # Optimal size for retrieval
CHUNK_OVERLAP = 40
MIN_CHUNK_SIZE = 50

# === ADVANCED CHROMA INDEXING SETTINGS ===
# Fixed parameter names to match ChromaDB expectations
CHROMA_SETTINGS = {
    # HNSW Algorithm Parameters
    "hnsw:space": "cosine",                # Distance metric (cosine is better for text)
    "hnsw:M": 32,                         # Number of connections per node (16-64 range)
    "hnsw:construction_ef": 400,          # Size of dynamic candidate list during construction
    "hnsw:search_ef": 100,                # Size of dynamic candidate list during search
    
    # Performance Parameters
    "hnsw:batch_size": 500,               # Batch size for adding vectors (default: 100)
    "hnsw:sync_threshold": 2000,          # Sync to disk threshold (default: 1000)
    "hnsw:num_threads": 4,                # Number of threads for indexing
    "hnsw:resize_factor": 1.2,            # Resize factor for index growth
}

print("🚀 Advanced ChromaDB Indexing Setup")
print("=" * 50)

def clean_content(content):
    """Clean and normalize content"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    cleaned = '\n'.join(lines)
    return cleaned

def create_content_hash(content):
    """Create hash for deduplication"""
    return hashlib.md5(content.encode()).hexdigest()

# === STEP 1: Initialize ChromaDB with Advanced Settings ===
print("🔧 Initializing ChromaDB with optimized settings...")

# Remove existing database for fresh start
if os.path.exists(DB_DIR):
    import shutil
    shutil.rmtree(DB_DIR)
    print("🗑️  Cleared existing database")

# Create ChromaDB client with optimized settings
chroma_client = chromadb.PersistentClient(
    path=DB_DIR,
    settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=DB_DIR
    )
)

# === STEP 2: Load and Process Documents ===
print("📄 Loading documents...")

docs = []
seen_hashes = set()

# Try multiple possible locations for the txt files
txt_patterns = [
    os.path.join(DATA_DIR, "docs", "*.txt"),
    os.path.join(DATA_DIR, "*.txt"),
    os.path.join("data", "docs", "*.txt"),
    os.path.join("data", "*.txt"),
    "*.txt"
]

found_files = []
for pattern in txt_patterns:
    files = glob.glob(pattern)
    found_files.extend(files)

found_files = list(set(found_files))

if not found_files:
    print("❌ No .txt files found!")
    print("Please run fetch_docs.py first to create the documentation corpus.")
    exit(1)

print(f"📂 Found {len(found_files)} file(s)")

# Process documents
for filepath in found_files:
    print(f"📄 Processing: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sections = raw_text.split("========================")
    print(f"   Found {len(sections)} sections")
    
    for section in sections:
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

        content = clean_content("\n".join(content_lines))
        
        if content and len(content) > MIN_CHUNK_SIZE:
            content_hash = create_content_hash(content)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                docs.append(Document(
                    page_content=content, 
                    metadata={
                        "title": title, 
                        "source": url,
                        "content_hash": content_hash,
                        "content_length": len(content)
                    }
                ))

print(f"✅ Loaded {len(docs)} unique documents")

if len(docs) == 0:
    print("❌ No documents loaded! Please check your data files.")
    exit(1)

# === STEP 3: Smart Chunking ===
print("✂️  Creating optimized chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

chunks = []
chunk_hashes = set()

for i, doc in enumerate(docs):
    doc_chunks = text_splitter.split_documents([doc])
    
    for chunk in doc_chunks:
        if len(chunk.page_content) < MIN_CHUNK_SIZE:
            continue
            
        chunk_hash = create_content_hash(chunk.page_content)
        if chunk_hash not in chunk_hashes:
            chunk_hashes.add(chunk_hash)
            # Enhanced metadata
            chunk.metadata.update({
                "chunk_hash": chunk_hash,
                "chunk_size": len(chunk.page_content),
                "doc_index": i,
                "chunk_id": f"doc_{i}_chunk_{len(chunks)}"
            })
            chunks.append(chunk)

print(f"✅ Created {len(chunks)} unique chunks")

if len(chunks) == 0:
    print("❌ No chunks created!")
    exit(1)

# === STEP 4: Initialize Embedding Model ===
print("🧠 Initializing embedding model...")

embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# === STEP 5: Create Optimized ChromaDB Collection ===
print("🏗️  Creating optimized ChromaDB collection...")

start_time = time.time()

try:
    # Delete collection if it exists
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("🗑️  Deleted existing collection")
    except:
        pass
    
    # Create collection with optimized metadata
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata=CHROMA_SETTINGS
    )
    
    print(f"📊 Collection created with settings:")
    for key, value in CHROMA_SETTINGS.items():
        print(f"   {key}: {value}")
    
    # Create vector store with the optimized collection
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
    )
    
    # Add documents in batches for better performance
    batch_size = CHROMA_SETTINGS.get("hnsw:batch_size", 500)
    
    print(f"📥 Adding {len(chunks)} chunks in batches of {batch_size}...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_start = time.time()
        
        vectorstore.add_documents(batch)
        
        batch_time = time.time() - batch_start
        print(f"   ✅ Batch {i//batch_size + 1}: {len(batch)} chunks in {batch_time:.2f}s ({len(batch)/batch_time:.1f} chunks/sec)")
    
    # Force sync to disk
    if hasattr(collection, 'persist'):
        collection.persist()
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Indexing Complete!")
    print(f"📈 Performance Summary:")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • Total time: {total_time:.2f}s")
    print(f"   • Indexing speed: {len(chunks)/total_time:.1f} chunks/sec")
    print(f"   • Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    
    # === STEP 6: Verify Index Performance ===
    print("\n🔍 Testing index performance...")
    
    test_queries = [
        "How do I configure user permissions?",
        "What is the deployment process?",
        "How to set up authentication?"
    ]
    
    for query in test_queries:
        query_start = time.time()
        results = vectorstore.similarity_search_with_score(query, k=3)
        query_time = time.time() - query_start
        
        print(f"   Query: '{query[:30]}...' -> {len(results)} results in {query_time*1000:.1f}ms")
    
    print(f"\n✅ Advanced ChromaDB indexing completed successfully!")
    print(f"📁 Database saved to: {os.path.abspath(DB_DIR)}")
    
except Exception as e:
    print(f"❌ Error during indexing: {e}")
    raise

print("\n🚀 Ready for optimized querying!")