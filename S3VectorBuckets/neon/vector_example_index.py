import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from ~/.env
home_env_path = Path.home() / '.env'
load_dotenv(dotenv_path=home_env_path)

def create_random_vectors(num_vectors, vector_dim):
    """
    Create an array of random vectors.
    Like creating a collection of unique fingerprints for our library books.
    """
    return np.random.random((num_vectors, vector_dim)).astype(np.float32)

def setup_vector_table(cursor):
    """
    Set up the database table with vector extension.
    Like preparing shelves in our library with special labels for fingerprints.
    """
    # Enable the vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drop table if it exists (for clean testing)
    cursor.execute("DROP TABLE IF EXISTS vector_embeddings;")
    
    # Create table with vector column (32 dimensions)
    cursor.execute("""
        CREATE TABLE vector_embeddings (
            id SERIAL PRIMARY KEY,
            embedding vector(32),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create an index for faster similarity search
    # Like creating a catalog system for quick book lookup
    # Comment out the next 4 lines to test without an index
    cursor.execute("""
        CREATE INDEX ON vector_embeddings 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 10);
    """)

def insert_vectors(cursor, vectors):
    """
    Insert vectors into the database.
    Like placing each book with its fingerprint on the shelf.
    """
    for i, vector in enumerate(vectors):
        # Convert numpy array to list for PostgreSQL
        vector_list = vector.tolist()
        cursor.execute(
            "INSERT INTO vector_embeddings (embedding) VALUES (%s);",
            (vector_list,)
        )
    print(f"Inserted {len(vectors)} vectors into the database.")

def find_nearest_neighbors(cursor, query_vector, k=3):
    """
    Find k nearest neighbors using cosine similarity.
    Like finding books with fingerprints most similar to our query book.
    """
    query_list = query_vector.tolist()
    
    # Cast the array to vector type explicitly
    cursor.execute("""
        SELECT id, embedding, (embedding <=> %s::vector) as distance
        FROM vector_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_list, query_list, k))
    
    return cursor.fetchall()

def main():
    """
    Main function that orchestrates the vector database test.
    Like running a complete library management demo.
    """
    conn_string = os.getenv("DATABASE_URL")
    
    if not conn_string:
        print("Error: DATABASE_URL not found in environment variables.")
        print("Make sure you have a .env file in your home directory with DATABASE_URL set.")
        return
    
    conn = None
    
    try:
        with psycopg2.connect(conn_string) as conn:
            print("✅ Connection established to Neon database")
            
            with conn.cursor() as cur:
                # Step 1: Create 100 vectors with 32 dimensions
                print("\n📊 Step 1: Creating 100 random vectors (32 dimensions each)")
                vectors = create_random_vectors(100, 32)
                print(f"Created vectors with shape: {vectors.shape}")
                print(f"Sample vector[0][:5]: {vectors[0][:5]}")  # Show first 5 elements
                
                # Step 2: Set up database and store vectors
                print("\n🗄️  Step 2: Setting up vector table and storing vectors")
                setup_vector_table(cur)
                insert_vectors(cur, vectors)
                
                # Step 3: Generate query vector
                print("\n🔍 Step 3: Generating query vector (32 dimensions)")
                query_vector = np.random.random(32).astype(np.float32)
                print(f"Query vector[:5]: {query_vector[:5]}")  # Show first 5 elements
                
                # Step 4: Find 3 nearest neighbors
                print("\n🎯 Step 4: Finding 3 nearest neighbors")
                neighbors = find_nearest_neighbors(cur, query_vector, k=3)
                
                print("\nNearest Neighbors Found:")
                print("=" * 50)
                for i, (vector_id, embedding, distance) in enumerate(neighbors, 1):
                    print(f"Rank {i}:")
                    print(f"  Vector ID: {vector_id}")
                    print(f"  Distance (cosine): {distance:.6f}")
                    print(f"  Embedding[:5]: {embedding[:5]}")  # Show first 5 elements
                    print()
                
                # Commit all changes
                conn.commit()
                print("✅ All operations completed successfully!")
                
                # Optional: Show some statistics
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                total_vectors = cur.fetchone()[0]
                print(f"\n📈 Database now contains {total_vectors} vectors")
                
    except psycopg2.Error as e:
        print("❌ Database connection failed.")
        print(f"Error: {e}")
    except Exception as e:
        print("❌ An unexpected error occurred.")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

