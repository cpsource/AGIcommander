import boto3
print(boto3.__version__)

# Test if s3vectors service is available
try:
    s3vectors = boto3.client('s3vectors', region_name='us-east-1')
    print("✅ S3 Vectors support is available!")
except Exception as e:
    print(f"❌ S3 Vectors not supported: {e}")
