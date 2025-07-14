import boto3

# Create an S3 resource object
s3 = boto3.resource('s3')

# Specify the bucket name
bucket_name = 'bucket-investigate-001' 

# Get the S3 bucket object
bucket = s3.Bucket(bucket_name)

# List objects in the bucket
print(f"Objects in bucket '{bucket_name}':")
for obj in bucket.objects.all():
    print(obj.key)


