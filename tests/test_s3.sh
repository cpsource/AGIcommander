
# tests/test_s3.sh

# Navigate to the project root directory (assuming the script is in the tests directory)
PROJECT_ROOT=$(dirname "$(dirname "$0")")
cd "$PROJECT_ROOT"

# Activate the virtual environment (if you have one)
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# Run the pytest command
pytest tests/test_s3.py

```

---tests/README-test-s3.md---
```markdown
# How to Run the S3 Integration Tests

This document explains how to run the S3 integration tests for the AGIcommander project.

## Prerequisites

1.  **AWS Account and Credentials:** You need an AWS account and properly configured credentials. The tests use your default AWS configuration, so ensure you have set up your credentials using one of the following methods:

    *   **AWS CLI:** Configure your credentials using the AWS CLI:

        ```bash
        aws configure
        ```

    *   **Environment Variables:** Set the following environment variables:

        ```bash
        export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
        export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
        export AWS_DEFAULT_REGION=YOUR_REGION
        ```

    *   **IAM Role:** If running on an EC2 instance, ensure the instance has an IAM role with the necessary S3 permissions.

2.  **Test Bucket:** You need an existing S3 bucket to run the tests against.  Replace `"your-test-bucket-name"` in `tests/test_s3.py` with the name of your test bucket.  Make sure the bucket is in the region specified in your AWS configuration.

3.  **Python Environment:** You need a Python environment with the required dependencies installed.  It is highly recommended to use a virtual environment.

4.  **AGIcommander Project:** You need to have the AGIcommander project cloned and set up.

## Installation

1.  **Clone the AGIcommander repository:**

    ```bash
    git clone <repository_url>
    cd agicommander
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install pytest
    pip install python-dotenv
    ```

## Configuration

1.  **Edit `tests/test_s3.py`:**

    *   Replace `"your-test-bucket-name"` with the name of your test S3 bucket.

2.  **Ensure AWS credentials are set:**

    *   Verify that your AWS credentials are set up correctly using one of the methods described in the Prerequisites section.

## Running the Tests

1.  **Navigate to the project root directory:**

    ```bash
    cd agicommander
    ```

2.  **Run the tests using the provided shell script:**

    ```bash
    ./tests/test_s3.sh
    ```

    Alternatively, you can run the tests directly using `pytest`:

    ```bash
    pytest tests/test_s3.py
    ```

## Expected Output

The tests should run and pass, indicating that the S3 MCP server is functioning correctly.  If any tests fail, review the error messages and ensure that your AWS credentials and bucket name are configured correctly.

## Troubleshooting

*   **"AWS credentials not found"**: Verify that your AWS credentials are set up correctly using one of the methods described in the Prerequisites section.
*   **"Bucket does not exist"**: Ensure that the bucket name in `tests/test_s3.py` is correct and that the bucket exists in your AWS account.
*   **"An error occurred (AccessDenied) when calling the ListBuckets operation: Access Denied"**: Ensure that your AWS credentials have the necessary permissions to list buckets and perform other S3 operations.
*   **Tests are skipped**: If the tests are skipped, it means the AWS credentials check in `tests/test_s3.py` failed. Double-check your environment variables or AWS CLI configuration.

## Cleanup

The tests create temporary files in your S3 bucket. The tests attempt to clean up these files, but it's a good idea to verify that all test objects have been deleted from your bucket after running the tests.
```