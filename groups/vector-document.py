import json
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import boto3
import psycopg2
import numpy as np
from botocore.exceptions import ClientError


@dataclass
class VectorDocument:
    """Represents a document with its embedding and S3 location"""
    id: str
    s3_key: str
    embedding: List[float]
    metadata: Dict[str, Any]
    content_hash: Optional[str] = None
    created_at: Optional[datetime] = None


class S3VectorStore:
    """
    Hybrid storage system: embeddings + metadata in PostgreSQL, content in S3
    Think of it like having a smart index (PostgreSQL) pointing to a warehouse (S3)
    """
    
    def __init__(self, 
                 pg_connection_string: str,
                 s3_bucket: str,
                 s3_client: Optional[boto3.client] = None,
                 table_name: str = "vector_documents",
                 embedding_dim: int = 1536):
        
        self.bucket = s3_bucket
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        
        # Initialize S3 client
        self.s3 = s3_client or boto3.client('s3')
        
        # Initialize PostgreSQL connection
        self.conn = psycopg2.connect(pg_connection_string)
        self.conn.autocommit = True
        
        # Create table if it doesn't exist
        self._create_table_if_not_exists()
    
    def _create_table_if_not_exists(self):
        """Create the vector documents table with pgvector support"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    s3_key TEXT NOT NULL UNIQUE,
                    embedding vector({self.embedding_dim}) NOT NULL,
                    metadata JSONB,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create HNSW index for fast similarity search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} 
                USING hnsw (embedding vector_cosine_ops);
            """)
            
            # Create index on metadata for filtering
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx 
                ON {self.table_name} 
                USING gin (metadata);
            """)
    
    def _generate_s3_key(self, doc_id: str, file_extension: str = "") -> str:
        """Generate a structured S3 key for the document"""
        date_prefix = datetime.now().strftime("%Y/%m/%d")
        return f"documents/{date_prefix}/{doc_id}{file_extension}"
    
    def _calculate_content_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content for integrity checking"""
        return hashlib.sha256(content).hexdigest()
    
    def store_document(self, 
                      content: bytes,
                      embedding: List[float],
                      metadata: Dict[str, Any],
                      file_extension: str = "",
                      doc_id: Optional[str] = None) -> VectorDocument:
        """
        Store a document: content goes to S3, embedding + metadata to PostgreSQL
        
        Like putting a book in a warehouse and adding its info to a card catalog
        """
        
        # Generate unique ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Generate S3 key and calculate content hash
        s3_key = self._generate_s3_key(doc_id, file_extension)
        content_hash = self._calculate_content_hash(content)
        
        try:
            # Step 1: Upload content to S3
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=content,
                Metadata={
                    'content-hash': content_hash,
                    'document-id': doc_id
                }
            )
            
            # Step 2: Store embedding and metadata in PostgreSQL
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_name} 
                    (id, s3_key, embedding, metadata, content_hash, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        s3_key = EXCLUDED.s3_key,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        content_hash = EXCLUDED.content_hash,
                        created_at = EXCLUDED.created_at
                """, (
                    doc_id,
                    s3_key,
                    embedding,
                    json.dumps(metadata),
                    content_hash,
                    datetime.now()
                ))
            
            return VectorDocument(
                id=doc_id,
                s3_key=s3_key,
                embedding=embedding,
                metadata=metadata,
                content_hash=content_hash,
                created_at=datetime.now()
            )
            
        except Exception as e:
            # Clean up S3 if PostgreSQL insert failed
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            except:
                pass
            raise e
    
    def similarity_search(self, 
                         query_embedding: List[float],
                         limit: int = 5,
                         metadata_filter: Optional[Dict[str, Any]] = None,
                         distance_threshold: Optional[float] = None,
                         include_content: bool = False) -> List[Tuple[VectorDocument, float, Optional[bytes]]]:
        """
        Find similar documents using vector similarity
        
        Like asking a librarian: "Find me books similar to this one"
        """
        
        # Build the query
        where_clause = ""
        params = [query_embedding, limit]
        
        if metadata_filter:
            # Use PostgreSQL's JSONB operators for filtering
            where_conditions = []
            for key, value in metadata_filter.items():
                where_conditions.append(f"metadata @> %s")
                params.insert(-1, json.dumps({key: value}))
            
            if where_conditions:
                where_clause = f"WHERE {' AND '.join(where_conditions)}"
        
        distance_filter = ""
        if distance_threshold is not None:
            distance_filter = f"AND (embedding <=> %s) < %s"
            params.insert(-1, query_embedding)
            params.insert(-1, distance_threshold)
        
        query = f"""
            SELECT id, s3_key, embedding, metadata, content_hash, created_at,
                   (embedding <=> %s) as distance
            FROM {self.table_name}
            {where_clause}
            {distance_filter}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        
        results = []
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            
            for row in cur.fetchall():
                doc_id, s3_key, embedding, metadata, content_hash, created_at, distance = row
                
                doc = VectorDocument(
                    id=doc_id,
                    s3_key=s3_key,
                    embedding=embedding,
                    metadata=json.loads(metadata) if metadata else {},
                    content_hash=content_hash,
                    created_at=created_at
                )
                
                # Optionally fetch content from S3
                content = None
                if include_content:
                    content = self.get_document_content(s3_key)
                
                results.append((doc, distance, content))
        
        return results
    
    def get_document_content(self, s3_key: str) -> bytes:
        """Retrieve document content from S3"""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Document not found in S3: {s3_key}")
            raise e
    
    def get_document_by_id(self, doc_id: str, include_content: bool = False) -> Optional[Tuple[VectorDocument, Optional[bytes]]]:
        """Get a specific document by ID"""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, s3_key, embedding, metadata, content_hash, created_at
                FROM {self.table_name}
                WHERE id = %s
            """, (doc_id,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            doc_id, s3_key, embedding, metadata, content_hash, created_at = row
            
            doc = VectorDocument(
                id=doc_id,
                s3_key=s3_key,
                embedding=embedding,
                metadata=json.loads(metadata) if metadata else {},
                content_hash=content_hash,
                created_at=created_at
            )
            
            content = None
            if include_content:
                content = self.get_document_content(s3_key)
            
            return (doc, content)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from both PostgreSQL and S3"""
        # First get the S3 key
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT s3_key FROM {self.table_name} WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            
            if not row:
                return False
            
            s3_key = row[0]
            
            # Delete from PostgreSQL
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = %s", (doc_id,))
            
            # Delete from S3
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            except ClientError:
                # Log but don't fail if S3 delete fails
                pass
            
            return True
    
    def update_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Update document metadata without changing content or embedding"""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {self.table_name} 
                SET metadata = %s 
                WHERE id = %s
            """, (json.dumps(metadata), doc_id))
            
            return cur.rowcount > 0
    
    def bulk_similarity_search(self, 
                              query_embeddings: List[List[float]],
                              limit_per_query: int = 5) -> List[List[Tuple[VectorDocument, float]]]:
        """
        Perform multiple similarity searches efficiently
        
        Like asking: "For each of these books, find me 5 similar ones"
        """
        all_results = []
        
        for embedding in query_embeddings:
            results = self.similarity_search(embedding, limit=limit_per_query)
            # Remove content from bulk results to keep memory usage reasonable
            simplified_results = [(doc, distance) for doc, distance, _ in results]
            all_results.append(simplified_results)
        
        return all_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total_documents,
                    AVG(vector_dims(embedding)) as avg_dimensions,
                    MIN(created_at) as earliest_document,
                    MAX(created_at) as latest_document
                FROM {self.table_name}
            """)
            
            stats = cur.fetchone()
            
            return {
                'total_documents': stats[0],
                'avg_dimensions': stats[1],
                'earliest_document': stats[2],
                'latest_document': stats[3],
                'bucket': self.bucket,
                'table_name': self.table_name
            }
    
    def close(self):
        """Close the PostgreSQL connection"""
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == "__main__":
    # Initialize the vector store
    store = S3VectorStore(
        pg_connection_string="postgresql://user:pass@localhost/vectordb",
        s3_bucket="my-vector-documents",
        embedding_dim=1536  # OpenAI embedding size
    )
    
    try:
        # Store a document
        content = b"This is a sample document about machine learning and AI."
        embedding = [0.1] * 1536  # In practice, use OpenAI or other embedding model
        metadata = {
            "title": "ML Document", 
            "author": "Data Scientist",
            "category": "technology"
        }
        
        doc = store.store_document(
            content=content,
            embedding=embedding,
            metadata=metadata,
            file_extension=".txt"
        )
        print(f"Stored document: {doc.id}")
        
        # Search for similar documents
        query_embedding = [0.1] * 1536  # Your query embedding
        results = store.similarity_search(
            query_embedding=query_embedding,
            limit=5,
            metadata_filter={"category": "technology"},
            include_content=True
        )
        
        for doc, distance, content in results:
            print(f"Found document {doc.id} with distance {distance:.4f}")
            print(f"Content: {content[:100]}...")  # First 100 bytes
            
    finally:
        store.close()

