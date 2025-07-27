AWS Vector Buckets are a brand new feature (just announced in July 2025) that adds native vector storage and querying capabilities to S3. Here's what you need to know:

## What are AWS Vector Buckets?

Vector buckets are a new bucket type that's purpose-built to store and query vectors, allowing you to store, access, and perform similarity queries on vector data without provisioning any infrastructure.

## Key API Operations

The vector bucket API includes these main operations:
- **PutVectors** - Insert vector embeddings into a vector index
- **QueryVectors** - Perform similarity searches  
- **ListVectors** - List vectors in an index
- **DeleteVectors** - Remove vectors
- **GetVectorBucket** - Get bucket information

## Where to Find Documentation

**Main Documentation:**
- Amazon S3 Vectors User Guide: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html
- Getting Started Tutorial: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html
- AWS CLI Reference: https://docs.aws.amazon.com/cli/latest/reference/s3vectors/

**Key Features:**
- Up to 90% cost reduction compared to traditional vector databases
- Sub-second query performance with strongly consistent writes
- Up to 10,000 indexes per bucket, 50 million vectors per index
- Native integration with Amazon Bedrock Knowledge Bases and OpenSearch

**Current Availability:**
Currently in preview in US East (N. Virginia), US East (Ohio), US West (Oregon), Asia Pacific (Sydney), and Europe (Frankfurt) regions.

This is perfect for RAG applications, semantic search, and AI workloads where you need cost-effective vector storage with reasonable query performance!

