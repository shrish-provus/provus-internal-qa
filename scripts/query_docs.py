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

DB_DIR = "./db"
COLLECTION_NAME = "provus_docs"

class AdvancedChromaQA:
    def __init__(self):
        print("🚀 Initializing Advanced ChromaDB QA System")
        print("=" * 50)
        
        # Initialize embedding model (must match the one used for indexing)
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=DB_DIR
            )
        )
        
        # Connect to existing collection
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            collection_count = self.collection.count()
            print(f"✅ Connected to collection '{COLLECTION_NAME}' with {collection_count} vectors")
            
            # Print collection metadata
            metadata = self.collection.metadata
            if metadata:
                print("🔧 Index Configuration:")
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
        except Exception as e:
            print(f"❌ Error connecting to collection: {e}")
            print("Please run the embedding script first!")
            exit(1)
        
        # Initialize vector store
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_model,
        )
        
        # Initialize LLM with optimized settings
        self.llm = Ollama(
            model="llama3",
            temperature=0.1,
            num_ctx=4096,
            num_predict=512,
        )
        
        # Enhanced prompt template
        self.qa_prompt = PromptTemplate(
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
        
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
        
        print("✅ QA System initialized successfully!")
        print()

    def advanced_search(self, query, k=6, score_threshold=0.6, search_type="hybrid"):
        """
        Advanced search with multiple strategies
        """
        print(f"🔍 Advanced search: '{query}'")
        start_time = time.time()
        
        results = {}
        
        # Strategy 1: Similarity search (without score_threshold parameter)
        if search_type in ["similarity", "hybrid"]:
            try:
                sim_results = self.vectorstore.similarity_search_with_score(query, k=k)
                # Apply score threshold manually
                filtered_results = [(doc, score) for doc, score in sim_results if score >= score_threshold]
                results["similarity"] = filtered_results
                print(f"   📊 Similarity search: {len(filtered_results)} results (after threshold filter)")
            except Exception as e:
                print(f"   ⚠️ Similarity search failed: {e}")
                # Fallback to regular similarity search
                try:
                    sim_docs = self.vectorstore.similarity_search(query, k=k)
                    # Add dummy scores
                    sim_results = [(doc, 0.8) for doc in sim_docs]
                    results["similarity"] = sim_results
                    print(f"   📊 Fallback similarity search: {len(sim_results)} results")
                except Exception as e2:
                    print(f"   ❌ All similarity searches failed: {e2}")
        
        # Strategy 2: Max Marginal Relevance (MMR) for diversity
        if search_type in ["mmr", "hybrid"]:
            try:
                mmr_results = self.vectorstore.max_marginal_relevance_search(
                    query, k=k, fetch_k=k*2, lambda_mult=0.7
                )
                # Add dummy scores for consistency
                mmr_with_scores = [(doc, 0.8) for doc in mmr_results]
                results["mmr"] = mmr_with_scores
                print(f"   🎯 MMR search: {len(mmr_results)} results")
            except Exception as e:
                print(f"   ⚠️ MMR search failed: {e}")
        
        search_time = time.time() - start_time
        print(f"   ⚡ Search completed in {search_time:.3f}s")
        
        return results, search_time

    def deduplicate_and_rank(self, search_results):
        """
        Advanced deduplication and ranking of results
        """
        all_results = []
        
        # Combine results from different search strategies
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
        
        # Group by source to avoid duplicates
        source_groups = defaultdict(list)
        for result in all_results:
            source_groups[result['source']].append(result)
        
        # Select best result per source
        final_results = []
        for source, group in source_groups.items():
            # Sort by score (higher is better) and content length
            best_result = max(group, key=lambda x: (x['score'], x['content_length']))
            final_results.append(best_result)
        
        # Sort final results by score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results

    def format_context(self, results, max_context_length=3000):
        """
        Format results into context with length management
        """
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
        """
        Complete question answering pipeline with advanced indexing
        """
        total_start = time.time()
        
        # Advanced search
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
        
        # Deduplicate and rank results
        ranked_results = self.deduplicate_and_rank(search_results)
        print(f"📋 Using {len(ranked_results)} unique sources")
        
        # Format context
        context = self.format_context(ranked_results)
        
        # Generate answer
        llm_start = time.time()
        try:
            answer = self.qa_chain.run(context=context, question=query)
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            answer = "Error generating answer. Please try rephrasing your question."
        
        llm_time = time.time() - llm_start
        
        # Prepare sources
        sources = []
        for result in ranked_results:
            sources.append({
                "title": result['title'],
                "url": result['source'],
                "relevance_score": result['score'],
                "strategy": result['strategy']
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

    def interactive_qa(self):
        """
        Interactive QA loop with advanced features
        """
        print("🔍 Advanced ChromaDB QA System Ready!")
        print("Commands:")
        print("  - Ask any question about your documentation")
        print("  - Type 'stats' to see database statistics")
        print("  - Type 'test' to run performance tests")
        print("  - Type 'quit' or Ctrl+C to exit")
        print("-" * 60)
        
        try:
            while True:
                query = input("\n>>> ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit']:
                    break
                    
                if query.lower() == 'stats':
                    self.show_stats()
                    continue
                    
                if query.lower() == 'test':
                    self.run_performance_test()
                    continue
                
                # Answer the question
                result = self.answer_question(query)
                
                print(f"\n💡 Answer:")
                print(f"{result['answer']}")
                
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'], 1):
                        score_text = f" (Score: {source['relevance_score']:.2f})" if source['relevance_score'] > 0 else ""
                        print(f"  {i}. {source['title']}{score_text}")
                        print(f"     🔗 {source['url']}")
                        print(f"     📊 Found via: {source['strategy']} search")
                
                print(f"\n⏱️  Performance:")
                print(f"   • Search: {result['search_time']:.3f}s")
                print(f"   • LLM: {result['llm_time']:.2f}s") 
                print(f"   • Total: {result['total_time']:.2f}s")
                print(f"   • Sources: {result['num_sources']}")
                print(f"   • Context: {result['context_length']} chars")

        except KeyboardInterrupt:
            print("\n\n👋 Exiting Advanced QA System. Goodbye!")

    def show_stats(self):
        """Show database statistics"""
        print("\n📊 Database Statistics:")
        print(f"   • Collection: {COLLECTION_NAME}")
        print(f"   • Total vectors: {self.collection.count()}")
        print(f"   • Index metadata: {self.collection.metadata}")

    def run_performance_test(self):
        """Run performance benchmarks"""
        print("\n🏃 Running performance tests...")
        
        test_queries = [
            "How do I configure user permissions?",
            "What is the deployment process?", 
            "How to set up authentication?",
            "Database configuration options",
            "API documentation and endpoints"
        ]
        
        total_time = 0
        for i, query in enumerate(test_queries, 1):
            start = time.time()
            result = self.answer_question(query, k=4)
            elapsed = time.time() - start
            total_time += elapsed
            
            print(f"   Test {i}: {elapsed:.2f}s ({len(result['sources'])} sources)")
        
        avg_time = total_time / len(test_queries)
        print(f"\n📈 Performance Summary:")
        print(f"   • Average query time: {avg_time:.2f}s")
        print(f"   • Total test time: {total_time:.2f}s")
        print(f"   • Queries per second: {len(test_queries)/total_time:.1f}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    qa_system = AdvancedChromaQA()
    qa_system.interactive_qa()