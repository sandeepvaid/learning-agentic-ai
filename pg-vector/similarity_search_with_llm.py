import os
import json
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Use Langchain's Gemini wrapper for robust structured JSON extraction
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional

# Load your global .env containing the Gemini API key
load_dotenv('/Users/sandeepvaid/Documents/Personal/agentic ai/.env')

# 1. Initialize models
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading Gemini Model for Query Parsing...")
# We use gemini-1.5-flash as the free option
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Database Connection
DB_URI = "postgresql://postgres:postgres@localhost:5432/test"
conn = psycopg2.connect(DB_URI)
cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# 3. Define the Schema for Gemini to output
class SchoolFilters(BaseModel):
    city: Optional[str] = Field(None, description="City, block, or village name found in the query")
    medium: Optional[str] = Field(None, description="Medium of instruction (e.g. English, Hindi)")
    board: Optional[str] = Field(None, description="Affiliation board (e.g. CBSE, State Board)")
    school_type: Optional[str] = Field(None, description="Type of school (e.g. Primary, Secondary)")

class ParsedQuery(BaseModel):
    semantic_text: str = Field(description="The fuzzy/subjective part of the query to embed. Remove the concrete filter words from this.")
    filters: SchoolFilters = Field(description="Concrete filters found in the query")

def parse_user_query(user_query: str) -> ParsedQuery:
    # Langchain's .with_structured_output() automatically handles the JSON mode and prompt injection
    structured_llm = llm.with_structured_output(ParsedQuery)
    
    prompt = f"""
    Extract structured filters and semantic meaning from this school search query.
    
    Query: "{user_query}"
    
    Separate the concrete location/type filters from the subjective semantic description.
    """
    
    return structured_llm.invoke(prompt)

def search():
    print("\n--- Hybrid PGVector Semantic Search (Powered by Gemini) ---")
    query_text = input("Enter your search query (e.g., 'good primary school in DADRA affiliated to CBSE'): ")
    if not query_text.strip():
        print("Query cannot be empty!")
        return

    # [STEP 1] Ask Gemini to parse the query into filters + semantic text
    print("\n[1/3] Asking Gemini to parse the query...")
    parsed = parse_user_query(query_text)
    
    semantic_text = parsed.semantic_text
    filters = parsed.filters
    
    print("\n✅ Parsed Result from Gemini:")
    print(f"  - Semantic Text for Vector Search: '{semantic_text}'")
    print(f"  - Extracted SQL Filters: {filters.model_dump()}")

    # [STEP 2] Generate embedding ONLY for the semantic part
    print("\n[2/3] Generating embedding for the semantic text...")
    query_embedding = embedding_model.encode(semantic_text)
    embedding_str = str(query_embedding.tolist())
    
    # [STEP 3] Construct and execute Hybrid SQL Query
    print("[3/3] Constructing and executing Hybrid SQL Query...")
    
    # Base query
    sql_base = """
    SELECT 
        udise_sch_code, 
        school_name, 
        address,
        lgd_vill_name, 
        lgd_block_name, 
        sch_type,
        medinstr1,
        affilition_board_sec,
        embedding <=> %s::vector AS distance
    FROM school
    WHERE embedding IS NOT NULL
    """
    # The first parameter is for the distance calculation in the SELECT
    params = [embedding_str] 
    
    # Dynamically build WHERE conditions based on Gemini's extracted filters
    filter_clauses = []
    
    if filters.city:
        # Search across block, village, or address
        filter_clauses.append("(lgd_block_name ILIKE %s OR lgd_vill_name ILIKE %s OR address ILIKE %s)")
        wildcard_loc = f"%{filters.city}%"
        params.extend([wildcard_loc, wildcard_loc, wildcard_loc])
        
    if filters.medium:
        filter_clauses.append("medinstr1 ILIKE %s")
        params.append(f"%{filters.medium}%")
        
    if filters.board:
        filter_clauses.append("affilition_board_sec ILIKE %s")
        params.append(f"%{filters.board}%")
        
    if filters.school_type:
        filter_clauses.append("sch_type ILIKE %s")
        params.append(f"%{filters.school_type}%")
        
    if filter_clauses:
        sql_base += " AND " + " AND ".join(filter_clauses)
        
    # Add ORDER BY (nearest neighbor) and LIMIT
    sql_base += "\nORDER BY embedding <=> %s::vector\nLIMIT 5;"
    # The final parameter is for the ORDER BY clause
    params.append(embedding_str) 
    
    print(f"\n[Executing SQL Query]: \n{sql_base}")
    
    cursor.execute(sql_base, tuple(params))
    results = cursor.fetchall()
    
    if not results:
        print("\n❌ No results found matching both the filters and semantic criteria.")
        return
        
    print(f"\n🎯 Top {len(results)} Results:\n" + "-"*50)
    for row in results:
        udise = row.get('udise_sch_code', 'N/A')
        name = row.get('school_name', 'N/A')
        address = row.get('address', 'N/A')
        village = row.get('lgd_vill_name', 'N/A')
        block = row.get('lgd_block_name', 'N/A')
        sch_type = row.get('sch_type', 'N/A')
        distance = row.get('distance', 0.0)
        
        print(f"School: {name}")
        print(f"UDISE: {udise}")
        print(f"Location: {address} | Village: {village} | Block: {block}")
        print(f"Type: {sch_type}")
        print(f"Distance Score: {distance:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    try:
        search()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()
