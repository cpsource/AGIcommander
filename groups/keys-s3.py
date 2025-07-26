class UUIDVectorStore:
    def store_document(self, content, embedding, metadata, 
                      doc_id=None, file_extension=""):
        """Store with consistent UUID across PostgreSQL and S3"""
        
        # Single UUID for everything
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Choose your S3 key strategy
        s3_key = self._create_s3_key(doc_id, file_extension)
        
        # Store content in S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=content,
            Metadata={
                'postgresql-id': doc_id,  # Reference back to PostgreSQL
                'vector-dimensions': str(len(embedding)),
                'stored-at': datetime.now().isoformat()
            }
        )
        
        # Store metadata + embedding in PostgreSQL  
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO vector_documents 
                (id, s3_key, embedding, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (doc_id, s3_key, embedding, json.dumps(metadata), datetime.now()))
        
        return doc_id  # Return the UUID that works for both systems
    
    def _create_s3_key(self, doc_id: str, file_extension: str = "") -> str:
        """Create S3 key from UUID - choose your strategy"""
        
        # Strategy 1: Direct UUID (recommended for simplicity)
        return f"{doc_id}{file_extension}"
        
        # Strategy 2: Hash partitioned (recommended for scale)
        # prefix = doc_id[:2]
        # return f"docs/{prefix}/{doc_id}{file_extension}"
        
    def get_by_uuid(self, doc_id: str, include_content: bool = False):
        """Get everything associated with a UUID"""
        
        # Get from PostgreSQL
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT s3_key, embedding, metadata, created_at
                FROM vector_documents WHERE id = %s
            """, (doc_id,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            s3_key, embedding, metadata, created_at = row
        
        # Get from S3 if requested
        content = None
        if include_content:
            content = self.s3.get_object(
                Bucket=self.bucket, 
                Key=s3_key
            )['Body'].read()
        
        return {
            'id': doc_id,
            'embedding': embedding,
            'metadata': json.loads(metadata),
            'content': content,
            'created_at': created_at,
            's3_key': s3_key
        }

# Usage
store = UUIDVectorStore(...)

# Store document
doc_id = store.store_document(
    content=b"Your document content here...",
    embedding=[1.2, 0.6, 0.8, 0.3],
    metadata={"concept": "bank-money", "version": "1.0"}
)

print(f"Document stored with UUID: {doc_id}")
# Both PostgreSQL and S3 use this same UUID!

# Retrieve everything
result = store.get_by_uuid(doc_id, include_content=True)
print(f"Retrieved: {result['metadata']}")
