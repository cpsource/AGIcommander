import json
import os
import pytest
from pathlib import Path
from typing import Dict, Any

# Assuming the S3 MCPServer is in this relative location
from servers.memory.S3.s3_mcp_server import S3MCPServer

# Load environment variables for AWS credentials
from dotenv import load_dotenv
load_dotenv()

# Check if AWS credentials are set
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# Skip tests if AWS credentials are not configured
requires_aws_auth = pytest.mark.skipif(
    not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_REGION),
    reason="AWS credentials not configured"
)

@pytest.fixture
def s3_server():
    """Fixture to create and initialize the S3 MCPServer"""
    server = S3MCPServer()
    asyncio.run(server._initialize_s3_clients())  # Initialize clients
    return server

@pytest.fixture
def test_bucket_name():
    """Fixture to provide a test bucket name"""
    # Replace with your test bucket name
    return "your-test-bucket-name"

@pytest.fixture
def test_file_path():
    """Fixture to provide a test file path"""
    # Create a dummy test file
    test_file = Path("test_file.txt")
    test_file.write_text("This is a test file for S3 MCP Server.")
    yield str(test_file)
    # Cleanup after the test
    test_file.unlink()

@pytest.mark.asyncio
@requires_aws_auth
async def test_list_buckets(s3_server):
    """Test listing S3 buckets"""
    result = await s3_server._list_buckets()
    assert isinstance(result, list)
    assert len(result) > 0
    data = json.loads(result[0].text)
    assert "buckets" in data
    assert "count" in data

@pytest.mark.asyncio
@requires_aws_auth
async def test_list_objects(s3_server, test_bucket_name):
    """Test listing objects in a bucket"""
    result = await s3_server._list_objects(bucket_name=test_bucket_name)
    assert isinstance(result, list)
    data = json.loads(result[0].text)
    assert "bucket" in data
    assert "objects" in data
    assert "count" in data

@pytest.mark.asyncio
@requires_aws_auth
async def test_check_object_exists(s3_server, test_bucket_name):
    """Test checking if an object exists in a bucket"""
    # Upload a test object
    object_key = "test_object.txt"
    await s3_server._upload_file(
        local_file_path="test_file.txt",  # Dummy file, content doesn't matter
        bucket_name=test_bucket_name,
        object_key=object_key
    )

    # Check if the object exists
    result = await s3_server._check_object_exists(bucket_name=test_bucket_name, object_key=object_key)
    assert isinstance(result, list)
    data = json.loads(result[0].text)
    assert "exists" in data
    assert data["exists"] is True

    # Delete the test object
    await s3_server._delete_object(bucket_name=test_bucket_name, object_key=object_key)

@pytest.mark.asyncio
@requires_aws_auth
async def test_upload_and_download_file(s3_server, test_bucket_name, test_file_path):
    """Test uploading and downloading a file"""
    object_key = "test_upload.txt"
    local_download_path = "test_download.txt"

    # Upload the file
    upload_result = await s3_server._upload_file(
        local_file_path=test_file_path,
        bucket_name=test_bucket_name,
        object_key=object_key
    )
    assert isinstance(upload_result, list)
    upload_data = json.loads(upload_result[0].text)
    assert upload_data["success"] is True

    # Download the file
    download_result = await s3_server._download_file(
        bucket_name=test_bucket_name,
        object_key=object_key,
        local_file_path=local_download_path
    )
    assert isinstance(download_result, list)
    download_data = json.loads(download_result[0].text)
    assert download_data["success"] is True

    # Verify the downloaded file exists
    assert Path(local_download_path).exists()

    # Delete the test object and downloaded file
    await s3_server._delete_object(bucket_name=test_bucket_name, object_key=object_key)
    Path(local_download_path).unlink()

@pytest.mark.asyncio
@requires_aws_auth
async def test_delete_object(s3_server, test_bucket_name):
    """Test deleting an object"""
    # Upload a test object
    object_key = "test_delete.txt"
    await s3_server._upload_file(
        local_file_path="test_file.txt",  # Dummy file, content doesn't matter
        bucket_name=test_bucket_name,
        object_key=object_key
    )

    # Delete the object
    result = await s3_server._delete_object(bucket_name=test_bucket_name, object_key=object_key)
    assert isinstance(result, list)
    data = json.loads(result[0].text)
    assert "success" in data
    assert data["success"] is True

    # Verify the object no longer exists
    check_result = await s3_server._check_object_exists(bucket_name=test_bucket_name, object_key=object_key)
    check_data = json.loads(check_result[0].text)
    assert "exists" in check_data
    assert check_data["exists"] is False

@pytest.mark.asyncio
@requires_aws_auth
async def test_get_object_metadata(s3_server, test_bucket_name):
    """Test getting object metadata"""
    # Upload a test object
    object_key = "test_metadata.txt"
    await s3_server._upload_file(
        local_file_path="test_file.txt",  # Dummy file, content doesn't matter
        bucket_name=test_bucket_name,
        object_key=object_key
    )

    # Get object metadata
    result = await s3_server._get_object_metadata(bucket_name=test_bucket_name, object_key=object_key)
    assert isinstance(result, list)
    data = json.loads(result[0].text)
    assert "bucket" in data
    assert "object_key" in data
    assert "size" in data

    # Delete the test object
    await s3_server._delete_object(bucket_name=test_bucket_name, object_key=object_key)

@pytest.mark.asyncio
@requires_aws_auth
async def test_create_presigned_url(s3_server, test_bucket_name):
    """Test creating a presigned URL"""
    # Upload a test object
    object_key = "test_presigned.txt"
    await s3_server._upload_file(
        local_file_path="test_file.txt",  # Dummy file, content doesn't matter
        bucket_name=test_bucket_name,
        object_key=object_key
    )

    # Create a presigned URL
    result = await s3_server._create_presigned_url(bucket_name=test_bucket_name, object_key=object_key)
    assert isinstance(result, list)
    data = json.loads(result[0].text)
    assert "presigned_url" in data
    assert "expiration_seconds" in data

    # Delete the test object
    await s3_server._delete_object(bucket_name=test_bucket_name, object_key=object_key)
