# S3 MCP Server

A Model Context Protocol (MCP) server that provides AWS S3 operations. Think of this as a smart bridge that lets AI assistants interact with your S3 buckets using natural language.

## What is MCP?

MCP (Model Context Protocol) is like having a universal translator between AI assistants and external services. Instead of the AI needing to know the specifics of every API, MCP servers provide a standardized way to expose capabilities.

## Features

This S3 MCP server provides these operations:

- **List buckets**: See all your S3 buckets
- **List objects**: Browse files in a bucket (with optional filtering)
- **Check existence**: Verify if a file exists
- **Upload files**: Send local files to S3
- **Download files**: Retrieve files from S3
- **Delete objects**: Remove files from S3
- **Get metadata**: Inspect file properties
- **Create presigned URLs**: Generate temporary access links

## Installation

1. **Set up the environment**:
   ```bash
   mkdir -p servers/memory/S3
   cd servers/memory/S3
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials** (one of these methods):
   
   **Option A: AWS CLI**
   ```bash
   aws configure
   ```
   
   **Option B: Environment variables**
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```
   
   **Option C: IAM roles** (if running on EC2)
   - Attach an IAM role with S3 permissions to your instance

## Usage

### Running the Server

```bash
python s3_mcp_server.py
```

The server communicates via stdin/stdout, so it's designed to be launched by MCP clients rather than run directly.

### Example Operations

Here are some examples of what you can do through an MCP client:

**List all buckets:**
```json
{
  "tool": "list_buckets"
}
```

**List files in a bucket:**
```json
{
  "tool": "list_objects",
  "arguments": {
    "bucket_name": "my-bucket",
    "prefix": "photos/",
    "max_keys": 50
  }
}
```

**Upload a file:**
```json
{
  "tool": "upload_file",
  "arguments": {
    "local_file_path": "./document.pdf",
    "bucket_name": "my-bucket",
    "object_key": "documents/document.pdf",
    "metadata": {
      "author": "John Doe",
      "department": "Engineering"
    }
  }
}
```

**Download a file:**
```json
{
  "tool": "download_file",
  "arguments": {
    "bucket_name": "my-bucket",
    "object_key": "documents/report.pdf",
    "local_file_path": "./downloads/report.pdf"
  }
}
```

## Understanding the Code Structure

The server is built like a restaurant with specialized stations:

- **Server Setup**: The main kitchen that handles incoming orders (MCP requests)
- **Tool Definitions**: The menu that describes what operations are available
- **Handler Functions**: Individual cooking stations that perform specific S3 operations
- **Error Handling**: Quality control that catches and reports problems

### Key Components

1. **S3MCPServer Class**: The main orchestrator
2. **Tool Handlers**: Functions that map to specific S3 operations
3. **Async Operations**: Non-blocking execution for better performance
4. **Error Management**: Graceful handling of AWS and file system errors

### Example Integration

To use this with Claude Desktop or another MCP client, you'd add it to your configuration:

```json
{
  "mcpServers": {
    "s3": {
      "command": "python",
      "args": ["/path/to/servers/memory/S3/s3_mcp_server.py"]
    }
  }
}
```

## AWS Permissions

Your AWS credentials need these S3 permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListAllMyBuckets",
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:GetObjectMetadata"
      ],
      "Resource": [
        "arn:aws:s3:::*",
        "arn:aws:s3:::*/*"
      ]
    }
  ]
}
```

## Troubleshooting

**"AWS credentials not found"**
- Check your AWS configuration using `aws configure list`
- Verify environment variables are set correctly

**"Bucket does not exist"**
- Ensure the bucket name is correct and you have access
- Check that you're in the right AWS region

**"Permission denied"**
- Verify your AWS user/role has the necessary S3 permissions
- Check bucket policies that might restrict access

## Security Notes

- This server inherits your AWS credentials' permissions
- Presigned URLs expire automatically (default: 1 hour)
- Always use least-privilege IAM policies
- Be cautious with delete operations - they're permanent

## Extension Ideas

Want to extend this server? Here are some ideas:

- Add batch operations for multiple files
- Implement S3 bucket creation/deletion
- Add support for S3 lifecycle policies
- Include CloudFront distribution management
- Add file compression/decompression
- Implement progress tracking for large uploads

The beauty of MCP is that you can add new tools without changing how clients interact with the server!

