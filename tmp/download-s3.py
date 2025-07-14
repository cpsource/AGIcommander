import boto3

# Create an S3 client object
s3_client = boto3.client('s3')

# Specify the bucket name and S3 object key
bucket_name = 'bucket-investigate-001'
object_name = 'my_uploaded_file.txt'

# Local file path to save the downloaded file
local_file_name = 'downloaded_file.txt'

try:
    s3_client.download_file(bucket_name, object_name, local_file_name)
    print(f"File '{object_name}' downloaded from '{bucket_name}' to '{local_file_name}'")
except Exception as e:
    print(f"Error downloading file: {e}")


