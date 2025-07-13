#!/usr/bin/env python3
"""
Vector Database MCP Server for AGIcommander

This server provides semantic memory capabilities through ChromaDB.
Think of it as the AI's long-term memory that can understand context
and relationships between different pieces of information.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
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
logger = logging.getLogger("vector-db-mcp-server")

class VectorDBMCPServer:
    """
    MCP Server for Vector Database operations using ChromaDB.
    
    This is like having a librarian who not only knows where every book is,
    but also understands what each book is about and can find related
    information even when you don't know exactly what you're looking for.
    """
    
    def __init__(self, db_path: str = "./memory/chromadb", model_name: str = "all-MiniLM-L6-v2"):
        self.server = Server("vector-db-mcp-server")
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.client = None
        self.embedding_model = None
        self.collections = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up all the MCP handlers - defining what operations are available"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return available vector database tools"""
            return [
                Tool(
                    name="create_collection",
                    description="Create a new collection for organizing related documents/knowledge",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection to create"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata for the collection"
                            }
                        },
                        "required": ["collection_name"]
                    }
                ),
                Tool(
                    name="list_collections",
                    description="List all available collections in the vector database",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="add_documents",
                    description="Add documents/knowledge to a collection with automatic embedding",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection to add to"
                            },
                            "documents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of text documents to add"
                            },
                            "metadata": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "Optional metadata for each document"
                            },
                            "ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional custom IDs for documents (auto-generated if not provided)"
                            }
                        },
                        "required": ["collection_name", "documents"]
                    }
                ),
                Tool(
                    name="search_similar",
                    description="Search for semantically similar documents using natural language queries",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection to search in"
                            },
                            "query": {
                                "type": "string",
                                "description": "Natural language search query"
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            },
                            "include_metadata": {
                                "type": "boolean",
                                "description": "Include metadata in results (default: true)",
                                "default": True
                            },
                            "include_distances": {
                                "type": "boolean", 
                                "description": "Include similarity distances (default: true)",
                                "default": True
                            }
                        },
                        "required": ["collection_name", "query"]
                    }
                ),
                Tool(
                    name="get_documents",
                    description="Retrieve specific documents by their IDs",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection"
                            },
                            "ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of document IDs to retrieve"
                            },
                            "include_metadata": {
                                "type": "boolean",
                                "description": "Include metadata in results (default: true)",
                                "default": True
                            }
                        },
                        "required": ["collection_name", "ids"]
                    }
                ),
                Tool(
                    name="update_documents",
                    description="Update existing documents in a collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection"
                            },
                            "ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of document IDs to update"
                            },
                            "documents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Updated document content"
                            },
                            "metadata": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "Updated metadata for documents"
                            }
                        },
                        "required": ["collection_name", "ids", "documents"]
                    }
                ),
                Tool(
                    name="delete_documents",
                    description="Delete documents from a collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection"
                            },
                            "ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of document IDs to delete"
                            }
                        },
                        "required": ["collection_name", "ids"]
                    }
                ),
                Tool(
                    name="count_documents",
                    description="Get the number of documents in a collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection"
                            }
                        },
                        "required": ["collection_name"]
                    }
                ),
                Tool(
                    name="get_collection_info",
                    description="Get detailed information about a collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection"
                            }
                        },
                        "required": ["collection_name"]
                    }
                ),
                Tool(
                    name="delete_collection",
                    description="Delete an entire collection and all its documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection to delete"
                            }
                        },
                        "required": ["collection_name"]
                    }
                ),
                Tool(
                    name="backup_collection",
                    description="Create a backup of a collection's data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection to backup"
                            },
                            "backup_path": {
                                "type": "string",
                                "description": "Path where backup should be saved"
                            }
                        },
                        "required": ["collection_name", "backup_path"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls - this is where the actual vector DB work happens"""
            
            try:
                # Initialize clients if not already done
                if not self.client:
                    await self._initialize_clients()
                
                # Route to appropriate handler
                if name == "create_collection":
                    return await self._create_collection(**arguments)
                elif name == "list_collections":
                    return await self._list_collections()
                elif name == "add_documents":
                    return await self._add_documents(**arguments)
                elif name == "search_similar":
                    return await self._search_similar(**arguments)
                elif name == "get_documents":
                    return await self._get_documents(**arguments)
                elif name == "update_documents":
                    return await self._update_documents(**arguments)
                elif name == "delete_documents":
                    return await self._delete_documents(**arguments)
                elif name == "count_documents":
                    return await self._count_documents(**arguments)
                elif name == "get_collection_info":
                    return await self._get_collection_info(**arguments)
                elif name == "delete_collection":
                    return await self._delete_collection(**arguments)
                elif name == "backup_collection":
                    return await self._backup_collection(**arguments)
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
    
    async def _initialize_clients(self):
        """Initialize ChromaDB client and embedding model"""
        try:
            # Create database directory if it doesn't exist
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = await asyncio.get_event_loop().run_in_executor(
                None, lambda: SentenceTransformer(self.model_name)
            )
            
            logger.info("Vector database clients initialized successfully")
            
        except Exception as e:
            raise Exception(f"Failed to initialize vector database: {str(e)}")
    
    async def _create_collection(self, collection_name: str, metadata: Optional[Dict] = None) -> List[TextContent]:
        """Create a new collection"""
        try:
            # Add timestamp to metadata
            if metadata is None:
                metadata = {}
            metadata["created_at"] = datetime.now().isoformat()
            metadata["embedding_model"] = self.model_name
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata
            )
            
            # Cache the collection reference
            self.collections[collection_name] = collection
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "action": "create_collection",
                    "collection_name": collection_name,
                    "metadata": metadata,
                    "success": True
                }, indent=2)
            )]
            
        except Exception as e:
            if "already exists" in str(e).lower():
                # Collection already exists, get it instead
                collection = self.client.get_collection(collection_name)
                self.collections[collection_name] = collection
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "action": "create_collection",
                        "collection_name": collection_name,
                        "message": "Collection already exists",
                        "success": True
                    }, indent=2)
                )]
            else:
                raise Exception(f"Failed to create collection: {str(e)}")
    
    async def _list_collections(self) -> List[TextContent]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            
            collection_info = []
            for collection in collections:
                info = {
                    "name": collection.name,
                    "id": collection.id,
                    "metadata": collection.metadata,
                    "count": collection.count()
                }
                collection_info.append(info)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "collections": collection_info,
                    "total_count": len(collection_info)
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to list collections: {str(e)}")
    
    async def _add_documents(self, collection_name: str, documents: List[str], 
                           metadata: Optional[List[Dict]] = None, 
                           ids: Optional[List[str]] = None) -> List[TextContent]:
        """Add documents to a collection"""
        try:
            # Get or create collection
            collection = self._get_collection(collection_name)
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"added_at": datetime.now().isoformat()} for _ in documents]
            else:
                # Add timestamp to existing metadata
                for meta in metadata:
                    meta["added_at"] = datetime.now().isoformat()
            
            # Generate embeddings
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.embedding_model.encode(documents).tolist()
            )
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "action": "add_documents",
                    "collection_name": collection_name,
                    "document_count": len(documents),
                    "ids": ids,
                    "success": True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to add documents: {str(e)}")
    
    async def _search_similar(self, collection_name: str, query: str, 
                            n_results: int = 5, include_metadata: bool = True,
                            include_distances: bool = True) -> List[TextContent]:
        """Search for similar documents"""
        try:
            collection = self._get_collection(collection_name)
            
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.embedding_model.encode([query]).tolist()[0]
            )
            
            # Build include list
            include = ["documents"]
            if include_metadata:
                include.append("metadatas")
            if include_distances:
                include.append("distances")
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=include
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i]
                }
                
                if include_metadata and "metadatas" in results:
                    result["metadata"] = results["metadatas"][0][i]
                
                if include_distances and "distances" in results:
                    result["distance"] = results["distances"][0][i]
                    result["similarity"] = 1 - results["distances"][0][i]  # Convert distance to similarity
                
                formatted_results.append(result)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "collection_name": collection_name,
                    "results": formatted_results,
                    "result_count": len(formatted_results)
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to search documents: {str(e)}")
    
    async def _get_documents(self, collection_name: str, ids: List[str], 
                           include_metadata: bool = True) -> List[TextContent]:
        """Get specific documents by ID"""
        try:
            collection = self._get_collection(collection_name)
            
            include = ["documents"]
            if include_metadata:
                include.append("metadatas")
            
            results = collection.get(
                ids=ids,
                include=include
            )
            
            # Format results
            documents = []
            for i, doc_id in enumerate(results["ids"]):
                doc = {
                    "id": doc_id,
                    "document": results["documents"][i]
                }
                
                if include_metadata and "metadatas" in results:
                    doc["metadata"] = results["metadatas"][i]
                
                documents.append(doc)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "collection_name": collection_name,
                    "documents": documents,
                    "document_count": len(documents)
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to get documents: {str(e)}")
    
    async def _update_documents(self, collection_name: str, ids: List[str], 
                              documents: List[str], metadata: Optional[List[Dict]] = None) -> List[TextContent]:
        """Update existing documents"""
        try:
            collection = self._get_collection(collection_name)
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"updated_at": datetime.now().isoformat()} for _ in documents]
            else:
                for meta in metadata:
                    meta["updated_at"] = datetime.now().isoformat()
            
            # Generate new embeddings
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.embedding_model.encode(documents).tolist()
            )
            
            # Update documents
            collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "action": "update_documents",
                    "collection_name": collection_name,
                    "updated_ids": ids,
                    "success": True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to update documents: {str(e)}")
    
    async def _delete_documents(self, collection_name: str, ids: List[str]) -> List[TextContent]:
        """Delete documents from collection"""
        try:
            collection = self._get_collection(collection_name)
            
            collection.delete(ids=ids)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "action": "delete_documents",
                    "collection_name": collection_name,
                    "deleted_ids": ids,
                    "success": True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to delete documents: {str(e)}")
    
    async def _count_documents(self, collection_name: str) -> List[TextContent]:
        """Count documents in collection"""
        try:
            collection = self._get_collection(collection_name)
            count = collection.count()
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "collection_name": collection_name,
                    "document_count": count
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to count documents: {str(e)}")
    
    async def _get_collection_info(self, collection_name: str) -> List[TextContent]:
        """Get detailed collection information"""
        try:
            collection = self._get_collection(collection_name)
            
            info = {
                "name": collection.name,
                "id": collection.id,
                "metadata": collection.metadata,
                "document_count": collection.count()
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(info, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to get collection info: {str(e)}")
    
    async def _delete_collection(self, collection_name: str) -> List[TextContent]:
        """Delete an entire collection"""
        try:
            self.client.delete_collection(collection_name)
            
            # Remove from cache
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "action": "delete_collection",
                    "collection_name": collection_name,
                    "success": True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to delete collection: {str(e)}")
    
    async def _backup_collection(self, collection_name: str, backup_path: str) -> List[TextContent]:
        """Create a backup of collection data"""
        try:
            collection = self._get_collection(collection_name)
            
            # Get all documents
            results = collection.get(include=["documents", "metadatas"])
            
            # Prepare backup data
            backup_data = {
                "collection_name": collection_name,
                "collection_metadata": collection.metadata,
                "backup_timestamp": datetime.now().isoformat(),
                "document_count": len(results["ids"]),
                "documents": []
            }
            
            # Add all documents to backup
            for i, doc_id in enumerate(results["ids"]):
                doc_data = {
                    "id": doc_id,
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else None
                }
                backup_data["documents"].append(doc_data)
            
            # Save backup file
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "action": "backup_collection",
                    "collection_name": collection_name,
                    "backup_path": str(backup_file),
                    "document_count": backup_data["document_count"],
                    "success": True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to backup collection: {str(e)}")
    
    def _get_collection(self, collection_name: str):
        """Get collection, creating it if it doesn't exist"""
        if collection_name not in self.collections:
            try:
                collection = self.client.get_collection(collection_name)
                self.collections[collection_name] = collection
            except Exception:
                # Collection doesn't exist, create it
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "created_at": datetime.now().isoformat(),
                        "embedding_model": self.model_name
                    }
                )
                self.collections[collection_name] = collection
        
        return self.collections[collection_name]

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Vector Database MCP Server...")
        
        # Import and setup stdio transport
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="vector-db-mcp-server",
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
    db_path = os.getenv("VECTOR_DB_PATH", "./memory/chromadb")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    server = VectorDBMCPServer(db_path=db_path, model_name=model_name)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())

