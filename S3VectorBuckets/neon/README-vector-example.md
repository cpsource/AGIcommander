I'll create a test program that demonstrates using Neon's vector database with pgvector. Think of this like setting up a library where each book (vector) has a unique fingerprint, and you want to find books with similar fingerprints to a query book.This program demonstrates vector similarity search using Neon's PostgreSQL database with the pgvector extension. Here's how it works:

**The Library Analogy**: Think of each vector as a unique "fingerprint" for a book in a library. When you have a new book (query vector), you want to find the 3 books with the most similar fingerprints.

**Key Components**:

1. **Vector Creation**: Like generating unique fingerprints for 100 books, each with 32 characteristics
2. **Database Setup**: Like preparing a special catalog system that can compare fingerprints
3. **Storage**: Like filing each book's fingerprint in the catalog
4. **Search**: Like finding books with fingerprints most similar to your query book

**Learning Points**:
- Uses `numpy.random.random()` to create vectors with values between 0 and 1
- Enables the `vector` extension in PostgreSQL for vector operations
- Uses cosine distance (`<=>` operator) to find similar vectors
- Creates an `ivfflat` index for faster similarity searches

**To run this**:
1. Make sure you have `psycopg2`, `numpy`, and `python-dotenv` installed
2. Set your Neon DATABASE_URL in a `.env` file in your home directory
3. Run the script

The program will show you the 3 most similar vectors to your query, along with their similarity scores (lower distance = more similar).

