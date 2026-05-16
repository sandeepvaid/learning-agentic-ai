import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# Import the vector_store from our local model.py
from model import vector_store

load_dotenv('/Users/sandeepvaid/Documents/Personal/agentic ai/.env')

# 1. Initialize the LLM
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
# 2. Define the State
class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 3. Define the Nodes
def retrieve_node(state: RAGState):
    """Retrieves context from the pgvector database based on the question."""
    question = state["question"]
    print(f"\n🔍 [Node: Retrieve] Fetching context for: '{question}'...")
    
    # Retrieve top 2 documents
    docs = vector_store.similarity_search(question, k=2)
    print("Retrieved docs", docs)
    return {"context": docs}

def generate_node(state: RAGState):
    """Generates an answer using the retrieved context."""
    print("🧠 [Node: Generate] Constructing the answer...")
    
    question = state["question"]
    context_docs = state.get("context", [])
    
    # Format the context cleanly
    formatted_context = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in context_docs
    )
    
    # Define a strict RAG prompt
    prompt = f"""
    You are an expert assistant. Use the following retrieved context to answer the user's question.
    If the context does not contain the answer, explicitly say "I don't know based on the provided context."
    Do not use any outside knowledge.
    
    Context:
    {formatted_context}
    
    Question:
    {question}
    
    Answer:
    """
    
    # Call Gemini
    response = llm.invoke(prompt)
    return {"answer": response.content}

# 4. Build and Compile the Graph
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Add edges to define the exact execution sequence
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
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
