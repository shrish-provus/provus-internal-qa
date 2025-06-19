import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
import time

# === Page Configuration ===
st.set_page_config(
    page_title="Provus Internal QA", 
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# === Initialize Session State ===
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# === Sidebar Configuration ===
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model settings
    k_value = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=8)
    
    # Database status
    st.subheader("📊 System Status")
    if st.session_state.model_loaded:
        st.success("✅ Models loaded")
    else:
        st.warning("⏳ Models not loaded")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# === Load Models (Cached) ===
@st.cache_resource
def load_qa_system(k_docs=8):
    """Load and cache the QA system components"""
    try:
        DB_DIR = "./db"
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": k_docs})
        llm = Ollama(model="llama3")
        
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
        
        return qa_chain, True
    except Exception as e:
        st.error(f"Error loading QA system: {str(e)}")
        return None, False

# === Main App ===
st.title("🤖 Provus Internal Q&A Assistant")
st.markdown("Ask questions based on internal documentation powered by LangChain + Ollama + Chroma.")

# Load QA system
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading models and vector database..."):
        qa_chain, success = load_qa_system(k_value)
        if success:
            st.session_state.qa_chain = qa_chain
            st.session_state.model_loaded = True
            st.success("✅ System ready!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Failed to load system. Please check your setup.")
            st.stop()
else:
    # Update retriever if k_value changed
    if st.session_state.qa_chain:
        current_k = st.session_state.qa_chain.retriever.search_kwargs.get("k", 8)
        if current_k != k_value:
            with st.spinner("🔄 Updating retriever settings..."):
                st.session_state.qa_chain, _ = load_qa_system(k_value)

# === Chat Interface ===
st.markdown("---")

# Display chat history
if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {question[:50]}..." if len(question) > 50 else f"Q{i+1}: {question}"):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    title = source.get('title', 'Untitled')
                    source_path = source.get('source', 'Unknown')
                    st.markdown(f"• {title} — `{source_path}`")

# Query input
query = st.text_input("🔍 Enter your question:", placeholder="What would you like to know?")

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("🚀 Ask Question", type="primary")

# Process query
if (ask_button or query) and query.strip() and st.session_state.model_loaded:
    with st.spinner("🔎 Searching documents and generating answer..."):
        try:
            # Get answer
            result = st.session_state.qa_chain.invoke({"question": query})
            answer = result.get("answer", "Sorry, I couldn't find an answer.")
            
            # Process sources
            sources_info = []
            source_documents = result.get("source_documents", [])
            
            for doc in source_documents:
                source_info = {
                    'title': doc.metadata.get("title", "Untitled"),
                    'source': doc.metadata.get("source", "Unknown"),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources_info.append(source_info)
            
            # Display current answer
            st.markdown("### 💡 Answer")
            st.success(answer)
            
            if sources_info:
                st.markdown("### 📚 Sources")
                for i, source in enumerate(sources_info, 1):
                    with st.expander(f"📄 Source {i}: {source['title']}", expanded=False):
                        st.markdown(f"**File:** `{source['source']}`")
                        st.markdown(f"**Preview:** {source['content_preview']}")
            
            # Add to chat history
            st.session_state.chat_history.append((query, answer, sources_info))
            
            # Success message
            st.info(f"✨ Found {len(sources_info)} relevant sources")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

elif query.strip() and not st.session_state.model_loaded:
    st.warning("⚠️ Please wait for the system to load before asking questions.")

# === Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    Powered by LangChain, Ollama, and ChromaDB | 
    Retrieving top {} documents per query
    </div>
    """.format(k_value), 
    unsafe_allow_html=True
)