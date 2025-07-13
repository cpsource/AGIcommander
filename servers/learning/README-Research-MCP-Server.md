# Research MCP Server

A Model Context Protocol (MCP) server that provides comprehensive research and knowledge acquisition capabilities for AGIcommander. Think of this as your AI's **research department** - capable of gathering, analyzing, and synthesizing information from multiple sources.

## Core Capabilities

### **Web Search & Information Gathering**
- **web_search**: Search the web for specific topics
- **fetch_webpage**: Extract content from individual web pages
- **research_topic**: Conduct comprehensive multi-source research

### **Analysis & Verification**
- **analyze_trends**: Analyze patterns and trends in research areas
- **fact_check**: Verify claims against multiple reliable sources
- **synthesize_research**: Combine findings from multiple research sessions

### **Knowledge Management**
- **monitor_topic**: Set up ongoing monitoring for topic changes
- **get_research_history**: Access previous research sessions
- **clear_cache**: Manage research cache and storage

## The Research Analogy

Think of this server like having a **brilliant research team**:

- **Research Assistant**: Gathers information from multiple sources
- **Fact Checker**: Verifies claims and cross-references sources
- **Trend Analyst**: Identifies patterns and emerging developments
- **Librarian**: Organizes and retrieves past research
- **Synthesizer**: Combines findings into coherent insights

## Usage Examples

### **Basic Web Search**
```json
{
  "tool": "web_search",
  "arguments": {
    "query": "artificial general intelligence development 2025",
    "num_results": 5
  }
}
```

### **Comprehensive Topic Research**
```json
{
  "tool": "research_topic",
  "arguments": {
    "topic": "Model Context Protocol implementation",
    "depth": "comprehensive",
    "focus_areas": ["best practices", "security", "performance"],
    "exclude_domains": ["social-media.com", "forums.example.com"]
  }
}
```

### **Content Extraction**
```json
{
  "tool": "fetch_webpage",
  "arguments": {
    "url": "https://example.com/research-article",
    "extract_main_content": true,
    "include_links": true
  }
}
```

### **Research History**
```json
{
  "tool": "get_research_history",
  "arguments": {
    "topic_filter": "machine learning",
    "limit": 10
  }
}
```

## Research Depth Levels

### **Basic Research**
- Simple overview queries
- 3-5 basic search terms
- Quick fact gathering

### **Detailed Research** (Default)
- Multiple search perspectives
- Best practices and challenges
- Latest developments
- Focus area deep-dives

### **Comprehensive Research**
- Extensive query coverage
- Case studies and implementations
- Trend analysis
- Future outlook assessment
- Implementation guides

## AGIcommander Integration

### **Learning Cycles**
```python
# During autonomous learning
research_results = await research_server.research_topic(
    topic="advanced code generation techniques",
    depth="comprehensive",
    focus_areas=["performance", "safety", "maintainability"]
)

# Store findings in vector database
await vector_db.add_documents(
    collection_name="research_knowledge",
    documents=[research_results['summary']],
    metadata={
        "research_id": research_results['research_id'],
        "topic": research_results['topic'],
        "confidence": "high"
    }
)
```

### **Real-time Knowledge Updates**
```python
# Monitor for changes in AGI development
await research_server.monitor_topic(
    topic="AGI safety developments",
    keywords=["alignment", "safety", "governance"],
    frequency="daily",
    alert_threshold=0.8
)
```

### **Fact Verification**
```python
# Before making claims or decisions
verification = await research_server.fact_check(
    claim="MCP servers communicate via stdin/stdout",
    sources_required=3,
    trusted_domains=["anthropic.com", "github.com", "docs.python.org"]
)
```

## Architecture Features

### **Respectful Web Crawling**
- Rate limiting per domain (2-second delays)
- Proper User-Agent headers
- Timeout handling
- Error recovery

### **Content Extraction**
- Smart main content detection
- Multiple extraction strategies
- Link extraction and processing
- Content relevance scoring

### **Caching System**
```
memory/research_cache/
├── webpage_123456.json       # Cached webpage content
├── research_topic_789.json   # Complete research sessions
├── monitor_abc123.json       # Topic monitoring configs
└── ...
```

### **Research History Tracking**
- Automatic session logging
- Query tracking and optimization
- Source reliability assessment
- Finding synthesis across sessions

## Configuration

### **Environment Variables**
```bash
export RESEARCH_CACHE_DIR="./memory/research_cache"
export RESEARCH_USER_AGENT="AGIcommander Research Bot 1.0"
export RESEARCH_TIMEOUT=30
export RESEARCH_RATE_LIMIT=2
```

### **Search Engine Integration**
The server includes a basic DuckDuckGo search implementation, but can be extended with:

- **Google Custom Search API**
- **Bing Search API** 
- **Academic search engines** (arXiv, Google Scholar)
- **Specialized databases** (GitHub, Stack Overflow)

## Advanced Features

### **Multi-Source Research**
```python
# Research combines multiple search strategies:
queries = [
    "topic overview",           # General understanding
    "topic latest developments", # Current state
    "topic best practices",     # Implementation guidance
    "topic challenges",         # Known issues
    "topic case studies"        # Real-world examples
]
```

### **Content Relevance Scoring**
```python
# Automatic relevance calculation
relevance_score = calculate_relevance(topic="machine learning", content=webpage_text)
# Returns 0.0-1.0 based on keyword matching and content analysis
```

### **Smart Content Extraction**
```python
# Tries multiple strategies to find main content:
selectors = [
    'main', 'article', '.content', '.post-content',
    '.entry-content', '.article-content', '#content'
]
```

## Future Enhancements

### **Planned Features**
- **Academic paper search** (arXiv, PubMed integration)
- **Real-time trend analysis** (Google Trends, social media APIs)
- **Advanced fact-checking** (multiple source cross-referencing)
- **Content summarization** (using LLM integration)
- **Research report generation** (structured output formats)

### **Integration Opportunities**
- **Vector database integration** for semantic search of research history
- **S3 storage** for large research datasets and multimedia content
- **Notification system** for monitoring alerts
- **API integrations** for specialized research sources

## Error Handling & Recovery

### **Network Issues**
- Automatic retry with exponential backoff
- Graceful degradation when sources unavailable
- Alternative search strategies

### **Rate Limiting**
- Per-domain request tracking
- Automatic delay insertion
- Respectful crawling practices

### **Content Issues**
- Multiple content extraction fallbacks
- Encoding detection and handling
- Malformed HTML recovery

## Security Considerations

### **Safe Browsing**
- URL validation and sanitization
- Content type checking
- Malware domain filtering (extendable)

### **Privacy Protection**
- No personal data collection
- Anonymized search queries
- Local cache storage only

### **Rate Limiting**
- Prevents overwhelming target servers
- Maintains good internet citizenship
- Configurable delay settings

This research server transforms AGIcommander from a **reactive AI** into a **proactive learning system** that can continuously gather knowledge, verify information, and build understanding over time - a crucial capability for any AGI system!

## Installation Dependencies

The server requires these additional packages:
```bash
pip install aiohttp beautifulsoup4 requests
```

These are included in the main AGIcommander dependencies via the `research` optional group in your pyproject.toml.

