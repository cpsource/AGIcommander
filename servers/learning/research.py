#!/usr/bin/env python3
"""
Research MCP Server for AGIcommander

This server provides research and knowledge acquisition capabilities.
Think of it as the AI's research assistant that can gather information
from various sources and synthesize knowledge for learning.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-mcp-server")

class ResearchMCPServer:
    """
    MCP Server for research and knowledge acquisition.
    
    Like having a brilliant research assistant who can explore the web,
    synthesize information, and present findings in a structured way.
    """
    
    def __init__(self, cache_dir: str = "./memory/research_cache"):
        self.server = Server("research-mcp-server")
        self.cache_dir = Path(cache_dir)
        self.session = None
        self.research_history = []
        self.rate_limits = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up all the MCP handlers - defining research capabilities"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return available research tools"""
            return [
                Tool(
                    name="web_search",
                    description="Search the web for information on a specific topic",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query or topic to research"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of search results to return (default: 5)",
                                "default": 5
                            },
                            "language": {
                                "type": "string",
                                "description": "Language code for search results (default: 'en')",
                                "default": "en"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="fetch_webpage",
                    description="Fetch and extract content from a specific webpage",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL of the webpage to fetch"
                            },
                            "extract_main_content": {
                                "type": "boolean",
                                "description": "Extract only main article content (default: true)",
                                "default": True
                            },
                            "include_links": {
                                "type": "boolean",
                                "description": "Include links found in the content (default: false)",
                                "default": False
                            }
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="research_topic",
                    description="Conduct comprehensive research on a topic using multiple sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to research comprehensively"
                            },
                            "depth": {
                                "type": "string",
                                "enum": ["basic", "detailed", "comprehensive"],
                                "description": "Depth of research to conduct",
                                "default": "detailed"
                            },
                            "focus_areas": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific areas to focus on during research"
                            },
                            "exclude_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Domains to exclude from research"
                            }
                        },
                        "required": ["topic"]
                    }
                ),
                Tool(
                    name="analyze_trends",
                    description="Analyze trends and patterns in a research area",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic area to analyze trends for"
                            },
                            "time_period": {
                                "type": "string",
                                "enum": ["week", "month", "quarter", "year"],
                                "description": "Time period to analyze trends over",
                                "default": "month"
                            },
                            "trend_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["popularity", "sentiment", "technology", "market"]
                                },
                                "description": "Types of trends to analyze"
                            }
                        },
                        "required": ["topic"]
                    }
                ),
                Tool(
                    name="fact_check",
                    description="Verify claims or statements against multiple reliable sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "Claim or statement to fact-check"
                            },
                            "sources_required": {
                                "type": "integer",
                                "description": "Minimum number of sources for verification (default: 3)",
                                "default": 3
                            },
                            "trusted_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of trusted domains to prioritize"
                            }
                        },
                        "required": ["claim"]
                    }
                ),
                Tool(
                    name="monitor_topic",
                    description="Set up monitoring for ongoing changes in a research topic",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to monitor for changes"
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific keywords to track"
                            },
                            "frequency": {
                                "type": "string",
                                "enum": ["hourly", "daily", "weekly"],
                                "description": "How often to check for updates",
                                "default": "daily"
                            },
                            "alert_threshold": {
                                "type": "number",
                                "description": "Significance threshold for alerts (0.0-1.0)",
                                "default": 0.7
                            }
                        },
                        "required": ["topic"]
                    }
                ),
                Tool(
                    name="synthesize_research",
                    description="Synthesize findings from multiple research sessions into a summary",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "research_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "IDs of research sessions to synthesize"
                            },
                            "synthesis_type": {
                                "type": "string",
                                "enum": ["summary", "comparison", "analysis", "report"],
                                "description": "Type of synthesis to perform",
                                "default": "summary"
                            },
                            "output_format": {
                                "type": "string",
                                "enum": ["structured", "narrative", "bullet_points"],
                                "description": "Format for the synthesized output",
                                "default": "structured"
                            }
                        },
                        "required": ["research_ids"]
                    }
                ),
                Tool(
                    name="get_research_history",
                    description="Retrieve history of research sessions and findings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic_filter": {
                                "type": "string",
                                "description": "Filter results by topic"
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Start date for history (YYYY-MM-DD)"
                            },
                            "date_to": {
                                "type": "string",
                                "description": "End date for history (YYYY-MM-DD)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 20
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="clear_cache",
                    description="Clear research cache and temporary files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "older_than_days": {
                                "type": "integer",
                                "description": "Clear cache entries older than this many days",
                                "default": 7
                            },
                            "topic_filter": {
                                "type": "string",
                                "description": "Only clear cache for specific topic"
                            }
                        },
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls - this is where the research work happens"""
            
            try:
                # Initialize session if not already done
                if not self.session:
                    await self._initialize_session()
                
                # Route to appropriate handler
                if name == "web_search":
                    return await self._web_search(**arguments)
                elif name == "fetch_webpage":
                    return await self._fetch_webpage(**arguments)
                elif name == "research_topic":
                    return await self._research_topic(**arguments)
                elif name == "analyze_trends":
                    return await self._analyze_trends(**arguments)
                elif name == "fact_check":
                    return await self._fact_check(**arguments)
                elif name == "monitor_topic":
                    return await self._monitor_topic(**arguments)
                elif name == "synthesize_research":
                    return await self._synthesize_research(**arguments)
                elif name == "get_research_history":
                    return await self._get_research_history(**arguments)
                elif name == "clear_cache":
                    return await self._clear_cache(**arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
    
    async def _initialize_session(self):
        """Initialize HTTP session and cache directory"""
        try:
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize aiohttp session with proper headers
            self.session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'AGIcommander Research Bot 1.0 (Educational/Research Purpose)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                },
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            logger.info("Research session initialized successfully")
            
        except Exception as e:
            raise Exception(f"Failed to initialize research session: {str(e)}")
    
    async def _web_search(self, query: str, num_results: int = 5, language: str = "en") -> List[TextContent]:
        """Perform web search using a search engine"""
        try:
            # For this implementation, we'll use a simple approach
            # In production, you'd integrate with proper search APIs like Google Custom Search, Bing, etc.
            
            # Simple DuckDuckGo search as fallback
            search_url = "https://duckduckgo.com/html"
            params = {
                'q': query,
                'l': language,
            }
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    result_elements = soup.find_all('a', {'class': 'result__a'})[:num_results]
                    
                    for element in result_elements:
                        title = element.get_text(strip=True)
                        url = element.get('href', '')
                        
                        # Extract snippet if available
                        snippet = ""
                        parent = element.find_parent('div', {'class': 'result'})
                        if parent:
                            snippet_elem = parent.find('a', {'class': 'result__snippet'})
                            if snippet_elem:
                                snippet = snippet_elem.get_text(strip=True)
                        
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                    
                    # Store in research history
                    research_entry = {
                        'id': f"search_{int(time.time())}",
                        'type': 'web_search',
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'results': results,
                        'num_results': len(results)
                    }
                    self.research_history.append(research_entry)
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            'search_query': query,
                            'results': results,
                            'total_found': len(results),
                            'research_id': research_entry['id']
                        }, indent=2)
                    )]
                else:
                    raise Exception(f"Search request failed with status {response.status}")
                    
        except Exception as e:
            raise Exception(f"Web search failed: {str(e)}")
    
    async def _fetch_webpage(self, url: str, extract_main_content: bool = True, include_links: bool = False) -> List[TextContent]:
        """Fetch and extract content from a webpage"""
        try:
            # Check rate limiting
            domain = urlparse(url).netloc
            if self._is_rate_limited(domain):
                raise Exception(f"Rate limited for domain {domain}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.get_text(strip=True) if title else "No title"
                    
                    # Extract main content
                    content = ""
                    if extract_main_content:
                        # Try to find main content areas
                        main_selectors = [
                            'main', 'article', '.content', '.post-content',
                            '.entry-content', '.article-content', '#content'
                        ]
                        
                        for selector in main_selectors:
                            main_element = soup.select_one(selector)
                            if main_element:
                                content = main_element.get_text(strip=True, separator='\n')
                                break
                        
                        if not content:
                            # Fallback to body content, excluding nav, footer, etc.
                            for tag in soup(['nav', 'footer', 'aside', 'script', 'style']):
                                tag.decompose()
                            body = soup.find('body')
                            content = body.get_text(strip=True, separator='\n') if body else ""
                    else:
                        content = soup.get_text(strip=True, separator='\n')
                    
                    # Extract links if requested
                    links = []
                    if include_links:
                        for link in soup.find_all('a', href=True):
                            href = urljoin(url, link['href'])
                            link_text = link.get_text(strip=True)
                            if link_text and href.startswith('http'):
                                links.append({
                                    'text': link_text,
                                    'url': href
                                })
                    
                    # Update rate limiting
                    self._update_rate_limit(domain)
                    
                    # Cache the content
                    cache_data = {
                        'url': url,
                        'title': title_text,
                        'content': content,
                        'links': links,
                        'timestamp': datetime.now().isoformat(),
                        'content_length': len(content)
                    }
                    
                    cache_file = self.cache_dir / f"webpage_{hash(url)}.json"
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            'url': url,
                            'title': title_text,
                            'content': content[:2000] + "..." if len(content) > 2000 else content,
                            'content_length': len(content),
                            'links_found': len(links),
                            'links': links if include_links else [],
                            'cached': True
                        }, indent=2)
                    )]
                else:
                    raise Exception(f"Failed to fetch webpage: HTTP {response.status}")
                    
        except Exception as e:
            raise Exception(f"Failed to fetch webpage: {str(e)}")
    
    async def _research_topic(self, topic: str, depth: str = "detailed", 
                            focus_areas: Optional[List[str]] = None,
                            exclude_domains: Optional[List[str]] = None) -> List[TextContent]:
        """Conduct comprehensive research on a topic"""
        try:
            research_id = f"topic_{int(time.time())}"
            logger.info(f"Starting {depth} research on topic: {topic}")
            
            # Generate search queries based on depth and focus areas
            queries = self._generate_research_queries(topic, depth, focus_areas)
            
            research_results = {
                'research_id': research_id,
                'topic': topic,
                'depth': depth,
                'focus_areas': focus_areas or [],
                'queries_used': queries,
                'sources': [],
                'key_findings': [],
                'summary': "",
                'timestamp': datetime.now().isoformat()
            }
            
            # Conduct searches for each query
            for query in queries:
                try:
                    search_results = await self._web_search(query, num_results=3)
                    search_data = json.loads(search_results[0].text)
                    
                    # Fetch content from top results
                    for result in search_data['results'][:2]:  # Limit to top 2 per query
                        if exclude_domains and any(domain in result['url'] for domain in exclude_domains):
                            continue
                            
                        try:
                            webpage_content = await self._fetch_webpage(result['url'])
                            webpage_data = json.loads(webpage_content[0].text)
                            
                            source = {
                                'url': result['url'],
                                'title': result['title'],
                                'query_used': query,
                                'content_snippet': webpage_data['content'][:500] + "...",
                                'relevance_score': self._calculate_relevance(topic, webpage_data['content'])
                            }
                            research_results['sources'].append(source)
                            
                        except Exception as e:
                            logger.warning(f"Failed to fetch {result['url']}: {e}")
                            continue
                    
                    # Brief pause between queries to be respectful
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue
            
            # Analyze and extract key findings
            research_results['key_findings'] = self._extract_key_findings(
                research_results['sources'], topic
            )
            
            # Generate summary
            research_results['summary'] = self._generate_research_summary(
                research_results, depth
            )
            
            # Store research results
            self.research_history.append(research_results)
            
            # Save to cache
            cache_file = self.cache_dir / f"research_{research_id}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(research_results, f, indent=2, ensure_ascii=False)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'research_id': research_id,
                    'topic': topic,
                    'sources_found': len(research_results['sources']),
                    'key_findings': research_results['key_findings'],
                    'summary': research_results['summary'],
                    'research_depth': depth,
                    'completed_at': research_results['timestamp']
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Topic research failed: {str(e)}")
    
    def _generate_research_queries(self, topic: str, depth: str, focus_areas: Optional[List[str]]) -> List[str]:
        """Generate search queries for comprehensive research"""
        base_queries = [topic]
        
        if depth == "basic":
            return [topic, f"{topic} overview", f"what is {topic}"]
        
        elif depth == "detailed":
            queries = [
                topic,
                f"{topic} overview",
                f"{topic} latest developments",
                f"{topic} best practices",
                f"{topic} challenges"
            ]
            
            if focus_areas:
                for area in focus_areas:
                    queries.append(f"{topic} {area}")
            
            return queries
        
        elif depth == "comprehensive":
            queries = [
                topic,
                f"{topic} comprehensive guide",
                f"{topic} latest research",
                f"{topic} trends 2024 2025",
                f"{topic} case studies",
                f"{topic} implementation",
                f"{topic} advantages disadvantages",
                f"{topic} future outlook"
            ]
            
            if focus_areas:
                for area in focus_areas:
                    queries.extend([
                        f"{topic} {area}",
                        f"{topic} {area} best practices",
                        f"{topic} {area} case study"
                    ])
            
            return queries
        
        return base_queries
    
    def _calculate_relevance(self, topic: str, content: str) -> float:
        """Calculate relevance score between topic and content"""
        # Simple relevance calculation based on keyword presence
        topic_words = topic.lower().split()
        content_lower = content.lower()
        
        matches = sum(1 for word in topic_words if word in content_lower)
        return matches / len(topic_words) if topic_words else 0.0
    
    def _extract_key_findings(self, sources: List[Dict], topic: str) -> List[str]:
        """Extract key findings from research sources"""
        findings = []
        
        # This is a simplified implementation
        # In a real system, you'd use NLP techniques for better extraction
        for source in sources[:5]:  # Top 5 sources
            content = source.get('content_snippet', '')
            sentences = content.split('.')
            
            # Look for sentences containing the topic
            topic_words = topic.lower().split()
            for sentence in sentences:
                if any(word in sentence.lower() for word in topic_words):
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 20 and clean_sentence not in findings:
                        findings.append(clean_sentence)
                        if len(findings) >= 10:  # Limit findings
                            break
        
        return findings[:10]
    
    def _generate_research_summary(self, research_data: Dict, depth: str) -> str:
        """Generate a summary of research findings"""
        topic = research_data['topic']
        sources_count = len(research_data['sources'])
        findings_count = len(research_data['key_findings'])
        
        summary = f"Research Summary for: {topic}\n\n"
        summary += f"Depth: {depth.title()}\n"
        summary += f"Sources analyzed: {sources_count}\n"
        summary += f"Key findings identified: {findings_count}\n\n"
        
        if research_data['key_findings']:
            summary += "Key Findings:\n"
            for i, finding in enumerate(research_data['key_findings'][:5], 1):
                summary += f"{i}. {finding}\n"
        
        summary += f"\nResearch completed: {research_data['timestamp']}"
        
        return summary
    
    def _is_rate_limited(self, domain: str) -> bool:
        """Check if domain is rate limited"""
        if domain not in self.rate_limits:
            return False
        
        last_request = self.rate_limits[domain]
        return (time.time() - last_request) < 2  # 2 second delay between requests
    
    def _update_rate_limit(self, domain: str):
        """Update rate limit tracking for domain"""
        self.rate_limits[domain] = time.time()
    
    async def _analyze_trends(self, topic: str, time_period: str = "month", 
                            trend_types: Optional[List[str]] = None) -> List[TextContent]:
        """Analyze trends in a research area"""
        # Placeholder implementation - would integrate with trend analysis APIs
        return [TextContent(
            type="text",
            text=json.dumps({
                'topic': topic,
                'time_period': time_period,
                'trend_analysis': f"Trend analysis for {topic} over {time_period} - Feature coming soon",
                'note': "This feature requires integration with trend analysis APIs"
            }, indent=2)
        )]
    
    async def _fact_check(self, claim: str, sources_required: int = 3, 
                         trusted_domains: Optional[List[str]] = None) -> List[TextContent]:
        """Fact-check a claim against multiple sources"""
        # Placeholder implementation - would implement proper fact-checking logic
        return [TextContent(
            type="text",
            text=json.dumps({
                'claim': claim,
                'verification_status': 'pending',
                'sources_checked': 0,
                'sources_required': sources_required,
                'note': "Fact-checking implementation coming soon"
            }, indent=2)
        )]
    
    async def _monitor_topic(self, topic: str, keywords: Optional[List[str]] = None,
                           frequency: str = "daily", alert_threshold: float = 0.7) -> List[TextContent]:
        """Set up monitoring for a topic"""
        monitor_id = f"monitor_{int(time.time())}"
        
        monitor_config = {
            'monitor_id': monitor_id,
            'topic': topic,
            'keywords': keywords or [],
            'frequency': frequency,
            'alert_threshold': alert_threshold,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        # Save monitor configuration
        monitor_file = self.cache_dir / f"monitor_{monitor_id}.json"
        with open(monitor_file, 'w', encoding='utf-8') as f:
            json.dump(monitor_config, f, indent=2)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                'action': 'monitor_created',
                'monitor_id': monitor_id,
                'topic': topic,
                'frequency': frequency,
                'status': 'monitoring_active'
            }, indent=2)
        )]
    
    async def _synthesize_research(self, research_ids: List[str], 
                                 synthesis_type: str = "summary",
                                 output_format: str = "structured") -> List[TextContent]:
        """Synthesize findings from multiple research sessions"""
        # Placeholder implementation
        return [TextContent(
            type="text",
            text=json.dumps({
                'research_ids': research_ids,
                'synthesis_type': synthesis_type,
                'output_format': output_format,
                'note': "Research synthesis implementation coming soon"
            }, indent=2)
        )]
    
    async def _get_research_history(self, topic_filter: Optional[str] = None,
                                  date_from: Optional[str] = None,
                                  date_to: Optional[str] = None,
                                  limit: int = 20) -> List[TextContent]:
        """Get research history"""
        filtered_history = self.research_history.copy()
        
        # Apply filters
        if topic_filter:
            filtered_history = [r for r in filtered_history 
                              if topic_filter.lower() in r.get('topic', '').lower()]
        
        # Apply date filters (simplified)
        if date_from or date_to:
            # Would implement proper date filtering here
            pass
        
        # Apply limit
        filtered_history = filtered_history[-limit:]
        
        return [TextContent(
            type="text",
            text=json.dumps({
                'research_history': filtered_history,
                'total_entries': len(filtered_history),
                'filters_applied': {
                    'topic': topic_filter,
                    'date_from': date_from,
                    'date_to': date_to,
                    'limit': limit
                }
            }, indent=2)
        )]
    
    async def _clear_cache(self, older_than_days: int = 7, 
                          topic_filter: Optional[str] = None) -> List[TextContent]:
        """Clear research cache"""
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            files_removed = 0
            
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.stat().st_mtime < cutoff_date.timestamp():
                    if topic_filter:
                        # Check if file relates to topic (simplified)
                        if topic_filter.lower() not in cache_file.name.lower():
                            continue
                    
                    cache_file.unlink()
                    files_removed += 1
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'action': 'cache_cleared',
                    'files_removed': files_removed,
                    'older_than_days': older_than_days,
                    'topic_filter': topic_filter
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to clear cache: {str(e)}")

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Research MCP Server...")
        
        # Import and setup stdio transport
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="research-mcp-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
            )

async def main():
    """Main entry point"""
    # Allow configuration via environment variables
    cache_dir = os.getenv("RESEARCH_CACHE_DIR", "./memory/research_cache")
    
    server = ResearchMCPServer(cache_dir=cache_dir)
    
    try:
        await server.run()
    finally:
        # Cleanup
        if server.session:
            await server.session.close()

if __name__ == "__main__":
    asyncio.run(main()
