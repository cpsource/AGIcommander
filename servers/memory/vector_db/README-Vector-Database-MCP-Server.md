# Vector Database MCP Server

A Model Context Protocol (MCP) server that provides semantic memory capabilities for AGIcommander using ChromaDB and sentence transformers. Think of this as the AI's **long-term semantic memory** - it doesn't just store information, it understands relationships and context.

## What Makes This Special?

Unlike traditional databases that store exact matches, this vector database understands **meaning**. Here's the analogy:

- **Traditional Database**: Like a filing cabinet where you need to know the exact folder name
- **Vector Database**: Like a smart librarian who understands what you're looking for even when you describe it vaguely

## Core Capabilities

### **Semantic Search**
```python
# Instead of exact keyword matching...
query = "error handling in Python functions"
# ...finds documents about exceptions, try/catch blocks, logging, etc.
```

### **Knowledge Organization**
- **Collections**: Organize knowledge by topic (code_patterns, research_notes, project_history)
- **Metadata**: Rich context for each piece of information
- **Embeddings**: Mathematical representations that capture meaning

### **Memory Operations**
- **Add**: Store new knowledge with automatic understanding
- **Search**: Find related information using natural language
- **Update**: Modify existing knowledge while preserving relationships
- **Backup**: Preserve knowledge across sessions

## Installation & Setup

The vector database is integrated into AGIcommander's main dependencies, but here are the key components:

```bash
# Core dependencies (already in your pyproject.toml)
pip install chromadb sentence-transformers

# Optional: GPU acceleration for embeddings
pip install torch  # If you have CUDA
```

## Usage Examples

### **Creating Knowledge Collections**

```json
{
  "tool": "create_collection",
  "arguments": {
    "collection_name": "code_patterns",
    "metadata": {
      "purpose": "Store successful coding patterns and solutions",
      "project": "agicommander"
    }
  }
}
```

### **Adding Knowledge**

```json
{
  "tool": "add_documents",
  "arguments": {
    "collection_name": "code_patterns",
    "documents": [
      "Always use try-except blocks when calling external APIs to handle network failures gracefully",
      "Implement async/await patterns for I/O operations to improve performance",
      "Use type hints in Python functions to improve code clarity and catch errors early"
    ],
    "metadata": [
      {"category": "error_handling", "language": "python", "confidence": "high"},
      {"category": "performance", "language": "python", "confidence": "high"}, 
      {"category": "code_quality", "language": "python", "confidence": "medium"}
    ]
  }
}
```

### **Semantic Search**

```json
{
  "tool": "search_similar",
  "arguments": {
    "collection_name": "code_patterns",
    "query": "How to handle failures when calling APIs?",
    "n_results": 3
  }
}
```

Response:
```json
{
  "results": [
    {
      "id": "doc_123",
      "document": "Always use try-except blocks when calling external APIs...",
      "similarity": 0.89,
      "metadata": {"category": "error_handling", "language": "python"}
    }
  ]
}
```

## AGIcommander Integration Patterns

### **Learning Memory**
```python
# After successful code generation
await vector_db.add_documents(
    collection_name="successful_solutions",
    documents=[generated_code],
    metadata={
        "task_type": "refactoring",
        "success_metrics": {"tests_passed": True, "performance_improved": True},
        "timestamp": datetime.now().isoformat()
    }
)
```

### **Knowledge Retrieval**
```python
# Before tackling a new task
similar_solutions = await vector_db.search_similar(
    collection_name="successful_solutions", 
    query="refactor large function into smaller components",
    n_results=5
)
```

### **Autonomous Learning**
```python
# During self-improvement cycles
research_findings = await vector_db.search_similar(
    collection_name="research_knowledge",
    query="best practices for AI code generation",
    n_results=10
)
```

## Collection Organization Strategy

Here's how to organize knowledge for maximum AGI effectiveness:

### **Core Collections**
- `code_patterns`: Successful coding solutions and patterns
- `project_context`: Information about current projects and their requirements
- `learning_history`: What the AI has learned and when
- `user_preferences`: User-specific preferences and feedback
- `research_findings`: External knowledge from web research

### **Specialized Collections**
- `error_solutions`: How to fix common errors and problems
- `performance_optimizations`: Techniques that improve code performance
- `security_practices`: Security-related knowledge and best practices
- `architecture_decisions`: High-level design decisions and their rationales

## Advanced Features

### **Metadata-Rich Storage**
Every piece of knowledge includes context:
```python
metadata = {
    "source": "user_feedback",
    "confidence_level": 0.85,
    "validation_status": "tested",
    "related_projects": ["agicommander", "mcp_servers"],
    "tags": ["python", "async", "performance"],
    "created_by": "research_agent",
    "last_used": "2025-07-13T10:30:00Z"
}
```

### **Backup and Recovery**
```json
{
  "tool": "backup_collection",
  "arguments": {
    "collection_name": "code_patterns",
    "backup_path": "./backups/code_patterns_2025_07_13.json"
  }
}
```

### **Similarity Tuning**
- **Distance < 0.3**: Very similar (near duplicates)
- **Distance 0.3-0.6**: Related concepts
- **Distance 0.6-0.8**: Loosely related
- **Distance > 0.8**: Different topics

## Performance Considerations

### **Embedding Models**
- **Default**: `all-MiniLM-L6-v2` (fast, good quality)
- **Better**: `all-mpnet-base-v2` (slower, higher quality)
- **Specialized**: `sentence-transformers/all-distilroberta-v1` (code-focused)

### **Memory Usage**
- Each document uses ~384 dimensions (for MiniLM)
- 10,000 documents ≈ 15MB of embeddings
- ChromaDB handles compression and indexing automatically

### **Search Speed**
- Sub-second search for up to 100,000 documents
- Automatic indexing for larger collections
- In-memory caching for frequently accessed collections

## Environment Configuration

```bash
# Optional environment variables
export VECTOR_DB_PATH="./memory/chromadb"
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export CHROMA_TELEMETRY="false"
```

## Integration with Other AGIcommander Components

### **With S3 Memory Server**
```python
# Store embeddings locally, backup to S3
await s3_server.upload_file(
    local_file_path="./memory/chromadb/backup.json",
    bucket_name="agicommander-memory",
    object_key="vector_db_backups/daily_backup.json"
)
```

### **With Code Server**
```python
# Before code modification, search for similar patterns
patterns = await vector_db.search_similar(
    collection_name="code_patterns",
    query=f"refactor {function_type} in {language}",
    n_results=5
)
```

### **With Research Server**
```python
# Store research findings for future reference
await vector_db.add_documents(
    collection_name="research_knowledge",
    documents=[research_summary],
    metadata={
        "source_url": url,
        "research_date": datetime.now().isoformat(),
        "confidence": research_confidence
    }
)
```

## Troubleshooting

**"Model download failed"**
- Check internet connection
- Try a different embedding model
- Use `sentence-transformers` cache: `~/.cache/torch/sentence_transformers/`

**"Collection already exists"**
- This is normal - the server will use the existing collection
- Use `get_collection_info` to inspect existing collections

**"Slow search performance"**
- Consider using a smaller, faster embedding model
- Reduce `n_results` in search queries
- Archive old collections to separate storage

**"High memory usage"**
- Monitor collection sizes with `count_documents`
- Consider splitting large collections by topic
- Use backup/restore to move old data to cold storage

## Security Notes

- Vector embeddings don't contain raw text (they're mathematical representations)
- Metadata can contain sensitive information - be careful what you store
- Backup files contain full document text - encrypt if necessary
- ChromaDB stores data locally by default (no cloud transmission)

This vector database forms the **semantic backbone** of AGIcommander's memory system, enabling true understanding-based knowledge retrieval rather than just keyword matching!
