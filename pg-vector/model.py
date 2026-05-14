import psycopg2
from sentence_transformers import SentenceTransformer

# 1. Initialize the model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to the database
DB_URI = "postgresql://postgres:postgres@localhost:5432/test-1.6-aug"
print("Connecting to database...")
conn = psycopg2.connect(DB_URI)
cursor = conn.cursor()

# 3. Fetch data from the school table
print("Fetching records...")
# Using udise_sch_code as the unique identifier and selecting useful text columns
query = """
SELECT 
    udise_sch_code, 
    school_name, 
    address, 
    lgd_vill_name, 
    lgd_block_name, 
    pincode,
    sch_type
FROM school 
WHERE embedding IS NULL
"""
cursor.execute(query)
rows = cursor.fetchall()

print(f"Found {len(rows)} schools to process.")

# 4. Generate embeddings and update the table
for row in rows:
    udise_code = row[0]
    
    # Safely get string values, handling NULLs
    school_name = str(row[1]) if row[1] else ""
    address = str(row[2]) if row[2] else ""
    vill_name = str(row[3]) if row[3] else ""
    block_name = str(row[4]) if row[4] else ""
    pincode = str(row[5]) if row[5] else ""
    sch_type = str(row[6]) if row[6] else ""
    
    # Combine the useful text fields into a descriptive string
    text_parts = []
    if school_name: text_parts.append(f"School Name: {school_name}")
    if udise_code: text_parts.append(f"UDISE Code: {udise_code}")
    if sch_type: text_parts.append(f"School Type: {sch_type}")
    if address: text_parts.append(f"Address: {address}")
    if vill_name: text_parts.append(f"Village: {vill_name}")
    if block_name: text_parts.append(f"Block: {block_name}")
    if pincode: text_parts.append(f"Pincode: {pincode}")
    
    text_to_embed = " | ".join(text_parts)
    print(text_to_embed)
    # Generate the embedding (returns a numpy array)
    embedding = model.encode(text_to_embed)
    # breakpoint()
    # Convert numpy array to list, then to string format for pgvector like '[0.1, 0.2, ...]'
    embedding_str = str(embedding.tolist())
    
    # Update the embedding column in the database
    # We cast the string to ::vector to ensure Postgres handles it correctly
    cursor.execute(
        "UPDATE school SET embedding = %s::vector WHERE udise_sch_code = %s",
        (embedding_str, udise_code)
    )

# 5. Commit the changes and close the connection
conn.commit()
cursor.close()
conn.close()

print("Successfully generated and saved embeddings!")