import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# Import the vector_store from our local model.py
from model import vector_store, get_bm25_retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv('/Users/sandeepvaid/Documents/Personal/agentic ai/.env')
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# 1. Initialize the LLM
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
# 2. Define the State
class RAGState(TypedDict):
    question: str
    vector_context: List[Document]
    bm25_context: List[Document]
    context: List[Document]
    answer: str

# 3. Define the Nodes
def retrieve_node(state: RAGState):
    """Retrieves context from the pgvector database based on the question."""
    question = state["question"]
    print(f"\n🔍 [Node: Vector Retrieve] Fetching context for: '{question}'...")
    
    # Retrieve top 5 documents
    docs = vector_store.similarity_search(question, k=5)
    print("Retrieved vector docs", docs)
    return {"vector_context": docs}

def bm25_retrieve_node(state: RAGState):
    """Retrieves context using BM25 sparse retrieval."""
    question = state["question"]
    print(f"\n🔍 [Node: BM25 Retrieve] Fetching context for: '{question}'...")
    
    bm25_retriever = get_bm25_retriever()
    docs = bm25_retriever.invoke(question)
    print("Retrieved bm25 docs", docs)
    return {"bm25_context": docs}

def rrf_node(state: RAGState):
    """Fuses the results from vector and bm25 retrieval using RRF."""
    print("🔄 [Node: RRF] Fusing retrieval results...")
    vector_docs = state.get("vector_context", [])
    bm25_docs = state.get("bm25_context", [])
    
    # RRF parameters
    k = 60
    rrf_scores = {}
    
    # Helper to calculate RRF
    def add_to_rrf(docs):
        for rank, doc in enumerate(docs, 1):
            doc_content = doc.page_content
            if doc_content not in rrf_scores:
                rrf_scores[doc_content] = {"score": 0, "doc": doc}
            rrf_scores[doc_content]["score"] += 1 / (k + rank)
            
    add_to_rrf(vector_docs)
    add_to_rrf(bm25_docs)
    
    # Sort by RRF score descending
    fused_docs = [
        item["doc"] 
        for item in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    ]
    
    # Return top 5 combined documents for reranking
    top_docs = fused_docs[:5]
    return {"context": top_docs}

def rerank_node(state: RAGState):
    """Re-ranks the fused documents using a cross-encoder model."""
    question = state["question"]
    docs = state.get("context", [])
    print(f"\n⚖️ [Node: Rerank] Re-ranking {len(docs)} documents...")
    
    compressor = CrossEncoderReranker(model=model, top_n=2)
    
    compressed_docs = compressor.compress_documents(docs, question)
    print("Reranked top docs", compressed_docs)
    return {"context": compressed_docs}

def generate_node(state: RAGState):
    """Generates an answer using the retrieved context."""
    print("🧠 [Node: Generate] Constructing the answer...")
    
    question = state["question"]
    context_docs = state.get("context", [])
    
    # Format the context cleanly
    formatted_context = "\n\n".join(
        f"Content: {doc.page_content}"
        for doc in context_docs
    )
    
    # Define a strict RAG prompt
    # prompt = f"""
    # You are an expert assistant. Use the following retrieved context to answer the user's question.
    # If the context does not contain the answer, explicitly say "I don't know based on the provided context."
    # Do not use any outside knowledge.
    
    # Context:
    # {formatted_context}
    
    # Question:
    # {question}
    
    # Answer:
    # """
    
    # Call Gemini
    # response = llm.invoke(prompt)
    return {"answer": formatted_context}

# 4. Build and Compile the Graph
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("bm25_retrieve", bm25_retrieve_node)
workflow.add_node("rrf", rrf_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)

# Add edges to define the exact execution sequence
workflow.add_edge(START, "retrieve")
workflow.add_edge(START, "bm25_retrieve")
workflow.add_edge("retrieve", "rrf")
workflow.add_edge("bm25_retrieve", "rrf")
# workflow.add_edge("rrf", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

# Compile into an executable application
app = workflow.compile()

def run_interactive():
    print("\n" + "="*50)
    print("🚀 Welcome to the LangGraph RAG System!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)
    
    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not user_input.strip():
            continue
            
        initial_state = {"question": user_input}
        
        # We can stream the events to show the user what node is currently running
        for event in app.stream(initial_state):
            pass # The print statements inside the nodes handle the UI
            
        # The stream loop modifies the state inplace, but we can also just use invoke 
        # to get the final unified state dictionary easily.
        final_state = app.invoke(initial_state)
        print(f"\n🤖 Final Answer:\n{final_state['answer']}")

if __name__ == "__main__":
    run_interactive()
