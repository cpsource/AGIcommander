import boto3
from botocore.exceptions import ClientError

# Create an S3 client object
s3_client = boto3.client('s3')

# Specify the bucket name and S3 object key
bucket_name = 'bucket-investigate-001'
object_name = 'my_uploaded_file.txt'

try:
    s3_client.head_object(Bucket=bucket_name, Key=object_name)
    print(f"Object '{object_name}' exists in bucket '{bucket_name}'")
except ClientError as e:
    if e.response['Error']['Code'] == "404":
        print(f"Object '{object_name}' does not exist in bucket '{bucket_name}'")
    else:
        print(f"Error checking object existence: {e}")


