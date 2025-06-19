from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain

DB_DIR = "./db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 8})
llm = Ollama(model="llama3")
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)

print("🔍 Ask a question (Ctrl+C to exit):")
try:
    while True:
        query = input(">>> ").strip()
        if not query:
            continue

        result = qa_chain.invoke({"question": query})
        print(f"\n💡 Answer:\n{result['answer']}\n")

        print("📚 Sources:")
        for doc in result.get("source_documents", []):
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", "Untitled")
            print(f"• {title} — {source}")


except KeyboardInterrupt:
    print("\n👋 Exiting QA.")