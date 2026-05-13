#Concepts we are targetting here is 
# 1. What are embeddings
# 2. How to generate embeddings
# 3. How to search semantically
# 4. How to build a system using embeddings and vector db
# 5. How we can use text-splitting/chunking strategy to get more accurate results
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore(embeddings)

def load_file_and_split(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits
    
def store_splits_to_vectorstore(splits):
    print(f"Total splits: {len(splits)}")
    print("Storing first 20 splits for testing...")
    vector_store.add_documents(documents=splits[:20])

def search(query):
    return vector_store.similarity_search_with_score(query)
    


current_dir = os.path.dirname(os.path.abspath(__file__))
splits = load_file_and_split(os.path.join(current_dir, 'sample.pdf'))
# print(splits[0])
store_splits_to_vectorstore(splits)

results = search("Why  is  Nginx  so  popular?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

