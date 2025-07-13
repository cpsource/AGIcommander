#!/usr/bin/env python3
"""
AWS S3 MCP Server

This server provides S3 operations through the Model Context Protocol.
It's like having a smart assistant that can interact with your S3 buckets
using natural language commands.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
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
logger = logging.getLogger("s3-mcp-server")

class S3MCPServer:
    """
    MCP Server for AWS S3 operations.
    
    Think of this as a smart wrapper around S3 that understands
    structured requests and provides structured responses.
    """
    
    def __init__(self):
        self.server = Server("s3-mcp-server")
        self.s3_client = None
        self.s3_resource = None
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up all the MCP handlers - like defining the server's vocabulary"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Return available S3 tools"""
            return [
                Tool(
                    name="list_buckets",
                    description="List all S3 buckets in the account",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="list_objects",
                    description="List objects in an S3 bucket with optional prefix filter",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "bucket_name": {
                                "type": "string",
                                "description": "Name of the S3 bucket"
                            },
                            "prefix": {
                                "type": "string",
                                "description": "Optional prefix to filter objects"
                            },
                            "max_keys": {
                                "type": "integer",
                                "description": "Maximum number of objects to return (default: 100)",
                                "default": 100
                            }
                        },
                        "required": ["bucket_name"]
                    }
                ),
                Tool(
                    name="check_object_exists",
                    description="Check if an object exists in an S3 bucket",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "bucket_name": {
                                "type": "string",
                                "description": "Name of the S3 bucket"
                            },
                            "object_key": {
                                "type": "string",
                                "description": "S3 object key (file path)"
                            }
                        },
                        "required": ["bucket_name", "object_key"]
                    }
                ),
                Tool(
                    name="upload_file",
                    description="Upload a local file to S3",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "local_file_path": {
                                "type": "string",
                                "description": "Path to the local file to upload"
                            },
                            "bucket_name": {
                                "type": "string",
                                "description": "Name of the S3 bucket"
                            },
                            "object_key": {
                                "type": "string",
                                "description": "S3 object key (destination path)"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata to attach to the object"
                            }
                        },
                        "required": ["local_file_path", "bucket_name", "object_key"]
                    }
                ),
                Tool(
                    name="download_file",
                    description="Download a file from S3 to local filesystem",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "bucket_name": {
                                "type": "string",
                                "description": "Name of the S3 bucket"
                            },
                            "object_key": {
                                "type": "string",
                                "description": "S3 object key (source path)"
                            },
                            "local_file_path": {
                                "type": "string",
                                "description": "Local path where file should be saved"
                            }
                        },
                        "required": ["bucket_name", "object_key", "local_file_path"]
                    }
                ),
                Tool(
                    name="delete_object",
                    description="Delete an object from S3",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "bucket_name": {
                                "type": "string",
                                "description": "Name of the S3 bucket"
                            },
                            "object_key": {
                                "type": "string",
                                "description": "S3 object key to delete"
                            }
                        },
                        "required": ["bucket_name", "object_key"]
                    }
                ),
                Tool(
                    name="get_object_metadata",
                    description="Get metadata and properties of an S3 object",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "bucket_name": {
                                "type": "string",
                                "description": "Name of the S3 bucket"
                            },
                            "object_key": {
                                "type": "string",
                                "description": "S3 object key"
                            }
                        },
                        "required": ["bucket_name", "object_key"]
                    }
                ),
                Tool(
                    name="create_presigned_url",
                    description="Generate a presigned URL for temporary access to an S3 object",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "bucket_name": {
                                "type": "string",
                                "description": "Name of the S3 bucket"
                            },
                            "object_key": {
                                "type": "string",
                                "description": "S3 object key"
                            },
                            "expiration": {
                                "type": "integer",
                                "description": "URL expiration time in seconds (default: 3600)",
                                "default": 3600
                            },
                            "http_method": {
                                "type": "string",
                                "description": "HTTP method for the URL (GET, PUT, etc.)",
                                "default": "GET"
                            }
                        },
                        "required": ["bucket_name", "object_key"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls - this is where the actual S3 work happens"""
            
            try:
                # Initialize S3 clients if not already done
                if not self.s3_client:
                    await self._initialize_s3_clients()
                
                # Route to appropriate handler based on tool name
                if name == "list_buckets":
                    return await self._list_buckets()
                elif name == "list_objects":
                    return await self._list_objects(**arguments)
                elif name == "check_object_exists":
                    return await self._check_object_exists(**arguments)
                elif name == "upload_file":
                    return await self._upload_file(**arguments)
                elif name == "download_file":
                    return await self._download_file(**arguments)
                elif name == "delete_object":
                    return await self._delete_object(**arguments)
                elif name == "get_object_metadata":
                    return await self._get_object_metadata(**arguments)
                elif name == "create_presigned_url":
                    return await self._create_presigned_url(**arguments)
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
    
    async def _initialize_s3_clients(self):
        """Initialize S3 clients - like setting up your connection to AWS"""
        try:
            self.s3_client = boto3.client('s3')
            self.s3_resource = boto3.resource('s3')
            
            # Test the connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.list_buckets
            )
            logger.info("S3 clients initialized successfully")
            
        except NoCredentialsError:
            raise Exception("AWS credentials not found. Please configure your credentials.")
        except Exception as e:
            raise Exception(f"Failed to initialize S3 clients: {str(e)}")
    
    async def _list_buckets(self) -> List[TextContent]:
        """List all S3 buckets"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.list_buckets
            )
            
            buckets = []
            for bucket in response['Buckets']:
                buckets.append({
                    'name': bucket['Name'],
                    'creation_date': bucket['CreationDate'].isoformat()
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'buckets': buckets,
                    'count': len(buckets)
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to list buckets: {str(e)}")
    
    async def _list_objects(self, bucket_name: str, prefix: str = "", max_keys: int = 100) -> List[TextContent]:
        """List objects in a bucket"""
        try:
            kwargs = {
                'Bucket': bucket_name,
                'MaxKeys': max_keys
            }
            if prefix:
                kwargs['Prefix'] = prefix
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.list_objects_v2(**kwargs)
            )
            
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag'].strip('"')
                    })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'bucket': bucket_name,
                    'prefix': prefix,
                    'objects': objects,
                    'count': len(objects),
                    'is_truncated': response.get('IsTruncated', False)
                }, indent=2)
            )]
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                raise Exception(f"Bucket '{bucket_name}' does not exist")
            else:
                raise Exception(f"Failed to list objects: {str(e)}")
    
    async def _check_object_exists(self, bucket_name: str, object_key: str) -> List[TextContent]:
        """Check if an object exists"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'bucket': bucket_name,
                    'object_key': object_key,
                    'exists': True
                }, indent=2)
            )]
            
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        'bucket': bucket_name,
                        'object_key': object_key,
                        'exists': False
                    }, indent=2)
                )]
            else:
                raise Exception(f"Error checking object existence: {str(e)}")
    
    async def _upload_file(self, local_file_path: str, bucket_name: str, object_key: str, metadata: Optional[Dict] = None) -> List[TextContent]:
        """Upload a file to S3"""
        try:
            # Check if local file exists
            if not Path(local_file_path).exists():
                raise Exception(f"Local file '{local_file_path}' does not exist")
            
            # Prepare upload arguments
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Upload the file
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.s3_client.upload_file(
                    local_file_path, bucket_name, object_key, ExtraArgs=extra_args
                )
            )
            
            # Get file size for confirmation
            file_size = Path(local_file_path).stat().st_size
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'action': 'upload',
                    'local_file': local_file_path,
                    'bucket': bucket_name,
                    'object_key': object_key,
                    'file_size': file_size,
                    'success': True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")
    
    async def _download_file(self, bucket_name: str, object_key: str, local_file_path: str) -> List[TextContent]:
        """Download a file from S3"""
        try:
            # Create directory if it doesn't exist
            Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.s3_client.download_file(bucket_name, object_key, local_file_path)
            )
            
            # Get downloaded file size for confirmation
            file_size = Path(local_file_path).stat().st_size
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'action': 'download',
                    'bucket': bucket_name,
                    'object_key': object_key,
                    'local_file': local_file_path,
                    'file_size': file_size,
                    'success': True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")
    
    async def _delete_object(self, bucket_name: str, object_key: str) -> List[TextContent]:
        """Delete an object from S3"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.s3_client.delete_object(Bucket=bucket_name, Key=object_key)
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'action': 'delete',
                    'bucket': bucket_name,
                    'object_key': object_key,
                    'success': True
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to delete object: {str(e)}")
    
    async def _get_object_metadata(self, bucket_name: str, object_key: str) -> List[TextContent]:
        """Get object metadata"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
            )
            
            metadata = {
                'bucket': bucket_name,
                'object_key': object_key,
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified').isoformat() if response.get('LastModified') else None,
                'etag': response.get('ETag', '').strip('"'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {}),
                'storage_class': response.get('StorageClass'),
                'version_id': response.get('VersionId')
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(metadata, indent=2)
            )]
            
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                raise Exception(f"Object '{object_key}' not found in bucket '{bucket_name}'")
            else:
                raise Exception(f"Failed to get object metadata: {str(e)}")
    
    async def _create_presigned_url(self, bucket_name: str, object_key: str, expiration: int = 3600, http_method: str = "GET") -> List[TextContent]:
        """Generate a presigned URL"""
        try:
            url = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': object_key},
                    ExpiresIn=expiration
                )
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    'bucket': bucket_name,
                    'object_key': object_key,
                    'presigned_url': url,
                    'expiration_seconds': expiration,
                    'http_method': http_method
                }, indent=2)
            )]
            
        except Exception as e:
            raise Exception(f"Failed to create presigned URL: {str(e)}")

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting S3 MCP Server...")
        
        # Import and setup stdio transport
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="s3-mcp-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
            )

async def main():
    """Main entry point"""
    server = S3MCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())

