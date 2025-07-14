import boto3

# Create an S3 client object
s3_client = boto3.client('s3')

# Specify the bucket name
bucket_name = 'bucket-investigate-001'

# Local file path and desired S3 object key
file_name = 'my_local_file.txt' 
object_name = 'my_uploaded_file.txt' 

try:
    s3_client.upload_file(file_name, bucket_name, object_name)
    print(f"File '{file_name}' uploaded to '{bucket_name}/{object_name}'")
except Exception as e:
    print(f"Error uploading file: {e}")

