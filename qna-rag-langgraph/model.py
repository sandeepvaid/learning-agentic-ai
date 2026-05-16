#main goal is to use the source webpage and extract the data and load it in pg-vector

#Document splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from loader import get_webpage_data

embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
engine = PGEngine.from_connection_string(
    url="postgresql+asyncpg://postgres:postgres@localhost:5432/rag-test"
)

from sqlalchemy.exc import ProgrammingError

# Initialize the table schema BEFORE creating the vector store
try:
    engine.init_vectorstore_table(
        table_name='qna',
        vector_size=384,  # all-MiniLM-L6-v2 produces 384-dimensional vectors
    )
except ProgrammingError:
    # Table already exists, which is fine when we are just querying!
    pass

vector_store = PGVectorStore.create_sync(
    engine=engine,
    table_name='qna',
    embedding_service=embedding_model
)

def get_splitted_doc(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

    return text_splitter.split_documents(docs)


def store_into_pgvector():
    print("Getting data from webpage...")
    docs = get_webpage_data()
    print("Splitting data into chunks...")
    all_splits = get_splitted_doc(docs)
    print("Adding data to pg-vector...")
    vector_store.add_documents(documents=all_splits)
    print("Done!")
