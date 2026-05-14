import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to the database
DB_URI = "postgresql://postgres:postgres@localhost:5432/test-1.6-aug"
conn = psycopg2.connect(DB_URI)
# Using DictCursor so you can easily select '*' and access columns by name!
cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

def search():
    print("\n--- PGVector Similarity Search ---")
    query_text = input("Enter your search query (e.g., 'primary school in specific block'): ")
    if not query_text.strip():
        print("Query cannot be empty!")
        return

    print("\nSelect a distance operator:")
    print("1. Cosine Distance (<=>) - Best for text/semantic search (Recommended)")
    print("2. Euclidean (L2) Distance (<->) - Straight-line geometric distance")
    print("3. Negative Inner Product (<#>) - Faster but requires vectors to be normalized")
    
    choice = input("Enter 1, 2, or 3 (default is 1): ").strip()
    
    operator = "<=>"
    dist_name = "Cosine Distance"
    
    if choice == "2":
        operator = "<->"
        dist_name = "Euclidean Distance"
    elif choice == "3":
        operator = "<#>"
        dist_name = "Negative Inner Product"
    
    print(f"\nGenerating embedding for your query...")
    query_embedding = model.encode(query_text)
    # Convert numpy array to list, then string for Postgres pgvector
    embedding_str = str(query_embedding.tolist())
    
    # 3. Formulate the pgvector SQL query
    # We calculate the distance in the SELECT clause so we can display it,
    # and we use it in the ORDER BY clause to find the nearest neighbors.
    sql_query = f"""
        SELECT 
            *, 
            embedding {operator} %s::vector AS distance
        FROM school
        WHERE embedding IS NOT NULL
        ORDER BY embedding {operator} %s::vector
        LIMIT 5;
    """
    
    print(f"Running search using {dist_name}...\n")
    # We pass the embedding string twice (once for SELECT, once for ORDER BY)
    cursor.execute(sql_query, (embedding_str, embedding_str))
    results = cursor.fetchall()
    
    if not results:
        print("No results found. (Make sure you have embedded some schools first!)")
        return
        
    print(f"Top {len(results)} Results:\n" + "-"*50)
    for row in results:
        # Accessing by column name since we are using DictCursor
        udise = row.get('udise_sch_code', 'N/A')
        name = row.get('school_name', 'N/A')
        village = row.get('lgd_vill_name', 'N/A')
        block = row.get('lgd_block_name', 'N/A')
        distance = row.get('distance', 0.0)
        
        print(f"School: {name}")
        print(f"UDISE: {udise}")
        print(f"Location: Village {village}, Block {block}")
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
