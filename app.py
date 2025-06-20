import streamlit as st
import time
import json
from collections import defaultdict
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import chromadb
from chromadb.config import Settings
import plotly.express as px
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === CONFIGURATION ===
DB_DIR = "./db"
COLLECTION_NAME = "provus_docs"

# Page configuration
st.set_page_config(
    page_title="Provus QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .source-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .answer-box {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        color: #1a1a1a;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .search-stats {
        background: #fff3cd;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAdvancedChromaQA:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.initialized = False
        
    @st.cache_resource
    def initialize_system(_self):
        """Initialize the QA system with caching"""
        try:
            # Initialize embedding model
            _self.embedding_model = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize ChromaDB client
            _self.chroma_client = chromadb.PersistentClient(
                path=DB_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=DB_DIR
                )
            )
            
            # Connect to existing collection
            _self.collection = _self.chroma_client.get_collection(name=COLLECTION_NAME)
            
            # Initialize vector store
            _self.vectorstore = Chroma(
                client=_self.chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=_self.embedding_model,
            )
            
            # Initialize LLM
            _self.llm = Ollama(
                model="llama3",
                temperature=0.1,
                num_ctx=4096,
                num_predict=512,
            )
            
            # Enhanced prompt template
            qa_prompt = PromptTemplate(
                template="""You are an expert assistant for Provus company documentation. Answer questions based ONLY on the provided context.

Context from Provus documentation:
{context}

Question: {question}

Instructions:
- Provide a direct, comprehensive answer based on the context
- If the context doesn't contain sufficient information, clearly state what's missing
- Include specific details and examples from the context when available
- Structure your response clearly with bullet points or sections if helpful
- Cite which document sections your answer comes from

Answer:""",
                input_variables=["context", "question"]
            )
            
            _self.qa_chain = LLMChain(llm=_self.llm, prompt=qa_prompt)
            _self.initialized = True
            
            return {
                "status": "success",
                "collection_count": _self.collection.count(),
                "metadata": _self.collection.metadata
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def advanced_search(self, query, k=6, score_threshold=0.3, search_type="hybrid"):
        """Advanced search with multiple strategies"""
        start_time = time.time()
        results = {}
        
        # Strategy 1: Similarity search
        if search_type in ["similarity", "hybrid"]:
            try:
                sim_results = self.vectorstore.similarity_search_with_score(query, k=k)
                filtered_results = [(doc, score) for doc, score in sim_results if score >= score_threshold]
                results["similarity"] = filtered_results
            except Exception:
                sim_docs = self.vectorstore.similarity_search(query, k=k)
                results["similarity"] = [(doc, 0.8) for doc in sim_docs]
        
        # Strategy 2: MMR search
        if search_type in ["mmr", "hybrid"]:
            try:
                mmr_results = self.vectorstore.max_marginal_relevance_search(
                    query, k=k, fetch_k=k*2, lambda_mult=0.7
                )
                results["mmr"] = [(doc, 0.8) for doc in mmr_results]
            except Exception:
                pass
        
        search_time = time.time() - start_time
        return results, search_time

    def deduplicate_and_rank(self, search_results):
        """Deduplicate and rank results"""
        all_results = []
        
        for strategy, results in search_results.items():
            for doc, score in results:
                all_results.append({
                    'document': doc,
                    'score': score,
                    'strategy': strategy,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'title': doc.metadata.get('title', 'Untitled'),
                    'content_length': len(doc.page_content)
                })
        
        # Group by source
        source_groups = defaultdict(list)
        for result in all_results:
            source_groups[result['source']].append(result)
        
        # Select best result per source
        final_results = []
        for source, group in source_groups.items():
            best_result = max(group, key=lambda x: (x['score'], x['content_length']))
            final_results.append(best_result)
        
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results

    def format_context(self, results, max_context_length=3000):
        """Format results into context"""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            doc = result['document']
            title = result['title']
            content = doc.page_content.strip()
            score = result['score']
            
            section = f"[Source {i+1}: {title} (Relevance: {score:.2f})]\n{content}\n"
            
            if current_length + len(section) > max_context_length and context_parts:
                break
                
            context_parts.append(section)
            current_length += len(section)
        
        return "\n---\n".join(context_parts)

    def answer_question(self, query, k=6, score_threshold=0.3, search_type="hybrid"):
        """Complete QA pipeline"""
        total_start = time.time()
        
        # Search
        search_results, search_time = self.advanced_search(
            query, k=k, score_threshold=score_threshold, search_type=search_type
        )
        
        if not any(search_results.values()):
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "search_time": search_time,
                "total_time": time.time() - total_start,
                "search_strategy": search_type
            }
        
        # Process results
        ranked_results = self.deduplicate_and_rank(search_results)
        context = self.format_context(ranked_results)
        
        # Generate answer
        llm_start = time.time()
        try:
            answer = self.qa_chain.run(context=context, question=query)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        llm_time = time.time() - llm_start
        
        # Prepare response
        sources = []
        for result in ranked_results:
            sources.append({
                "title": result['title'],
                "url": result['source'],
                "relevance_score": result['score'],
                "strategy": result['strategy'],
                "content_preview": result['document'].page_content[:200] + "..." if len(result['document'].page_content) > 200 else result['document'].page_content
            })
        
        total_time = time.time() - total_start
        
        return {
            "answer": answer.strip(),
            "sources": sources,
            "search_time": search_time,
            "llm_time": llm_time,
            "total_time": total_time,
            "num_sources": len(sources),
            "search_strategy": search_type,
            "context_length": len(context)
        }

# Initialize the QA system
@st.cache_resource
def get_qa_system():
    return StreamlitAdvancedChromaQA()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Provus Advanced QA System</h1>
        <p>Intelligent document search powered by ChromaDB & LLaMA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize QA system
    qa_system = get_qa_system()
    
    # Initialize session state
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = ""
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        if not st.session_state.system_initialized:
            with st.spinner("Initializing QA System..."):
                init_result = qa_system.initialize_system()
                
            if init_result["status"] == "success":
                st.session_state.system_initialized = True
                st.success("✅ System Initialized!")
                
                # Display system info
                st.markdown(f"""
                <div class="metric-card">
                    <strong>📊 Database Info</strong><br>
                    Collection: {COLLECTION_NAME}<br>
                    Documents: {init_result['collection_count']:,}<br>
                    Status: Ready
                </div>
                """, unsafe_allow_html=True)
                
                # Display index configuration
                if init_result.get('metadata'):
                    st.markdown("### ⚙️ Index Configuration")
                    for key, value in init_result['metadata'].items():
                        st.text(f"{key}: {value}")
                        
            else:
                st.error(f"❌ Initialization failed: {init_result['error']}")
                st.stop()
        else:
            st.success("✅ System Ready!")
        
        st.markdown("---")
        
        # Search Configuration
        st.markdown("### 🔍 Search Settings")
        search_type = st.selectbox(
            "Search Strategy",
            ["hybrid", "similarity", "mmr"],
            help="Hybrid combines similarity and MMR search"
        )
        
        num_results = st.slider(
            "Number of Results",
            min_value=3,
            max_value=15,
            value=6,
            help="Maximum number of documents to retrieve"
        )
        
        score_threshold = st.slider(
            "Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum relevance score for results"
        )
        
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.qa_history:
            st.markdown("### 📈 Session Stats")
            avg_time = sum(q['total_time'] for q in st.session_state.qa_history) / len(st.session_state.qa_history)
            st.metric("Questions Asked", len(st.session_state.qa_history))
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            if st.button("🗑️ Clear History"):
                st.session_state.qa_history = []
                st.rerun()
    
    # Main content area
    if st.session_state.system_initialized:
        # Question input - Fixed approach
        st.markdown("### 💬 Ask Your Question")
        
        # Use the selected example if available, otherwise use empty string
        default_value = st.session_state.selected_example if st.session_state.selected_example else ""
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_question = st.text_input(
                "Enter your question about Provus documentation:",
                value=default_value,
                placeholder="e.g., How do I configure user permissions?",
                key="question_input"
            )
        
        with col2:
            ask_button = st.button("🚀 Ask Question", type="primary")
        
        # Example questions - Fixed approach
        st.markdown("**💡 Example Questions:**")
        example_questions = [
            "How do I configure user permissions?",
            "What is the deployment process?",
            "How to set up authentication?",
            "Database configuration options",
            "API documentation and endpoints"
        ]
        
        cols = st.columns(len(example_questions))
        for i, example in enumerate(example_questions):
            with cols[i]:
                if st.button(f"📝 {example[:20]}...", key=f"example_{i}"):
                    # Set the selected example and rerun
                    st.session_state.selected_example = example
                    st.rerun()
        
        # Clear the selected example after it's been used
        if st.session_state.selected_example and user_question == st.session_state.selected_example:
            st.session_state.selected_example = ""
        
        # Process question
        if (ask_button or user_question) and user_question.strip():
            with st.spinner("🔍 Searching documents and generating answer..."):
                start_time = time.time()
                
                # Get answer
                result = qa_system.answer_question(
                    user_question,
                    k=num_results,
                    score_threshold=score_threshold,
                    search_type=search_type
                )
                
                # Add to history
                result['question'] = user_question
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.qa_history.append(result)
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.markdown(f"""
            <div class="answer-box">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display search statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔍 Search Time", f"{result['search_time']:.3f}s")
            with col2:
                st.metric("🤖 LLM Time", f"{result['llm_time']:.2f}s")
            with col3:
                st.metric("⏱️ Total Time", f"{result['total_time']:.2f}s")
            with col4:
                st.metric("📚 Sources Found", result['num_sources'])
            
            # Display sources
            if result['sources']:
                st.markdown("### 📚 Sources")
                
                for i, source in enumerate(result['sources'], 1):
                    with st.expander(f"📄 Source {i}: {source['title']} (Score: {source['relevance_score']:.2f})"):
                        st.markdown(f"**🔗 URL:** {source['url']}")
                        st.markdown(f"**📊 Strategy:** {source['strategy']}")
                        st.markdown(f"**📝 Content Preview:**")
                        st.text(source['content_preview'])
        
        # Question History
        if st.session_state.qa_history:
            st.markdown("---")
            st.markdown("### 📋 Question History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:]), 1):
                with st.expander(f"❓ {qa['question'][:50]}... ({qa['timestamp']})"):
                    st.markdown(f"**Answer:** {qa['answer'][:200]}...")
                    st.markdown(f"**Sources:** {qa['num_sources']} | **Time:** {qa['total_time']:.2f}s")
        
        # Performance Analytics
        if len(st.session_state.qa_history) > 1:
            st.markdown("---")
            st.markdown("### 📊 Performance Analytics")
            
            # Create performance chart
            df = pd.DataFrame([
                {
                    'Question': f"Q{i+1}",
                    'Search Time': qa['search_time'],
                    'LLM Time': qa['llm_time'],
                    'Total Time': qa['total_time'],
                    'Sources': qa['num_sources']
                }
                for i, qa in enumerate(st.session_state.qa_history[-10:])
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(df, x='Question', y=['Search Time', 'LLM Time'], 
                             title="Response Time Breakdown")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.line(df, x='Question', y='Sources', 
                              title="Sources Found per Question")
                st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()