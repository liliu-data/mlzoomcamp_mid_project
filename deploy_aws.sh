#!/bin/bash
# AWS deployment script using AWS Elastic Beanstalk or ECS

set -e

echo "=========================================="
echo "Deploying Stroke Prediction API to AWS"
echo "=========================================="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed."
    echo "Please install it from: https://aws.amazon.com/cli/"
    exit 1
fi

# Set variables
AWS_REGION=${AWS_REGION:-"us-east-1"}
ECR_REPO_NAME="stroke-prediction"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: Could not get AWS account ID. Please configure AWS credentials."
    exit 1
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "Region: ${AWS_REGION}"
echo "ECR Repository: ${ECR_REPO_NAME}"

# Create ECR repository if it doesn't exist
echo "Creating ECR repository if needed..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} || \
    aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}

# Get login token and login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# Build and push Docker image
echo "Building Docker image..."
docker build -t ${ECR_REPO_NAME}:latest .

echo "Tagging image..."
docker tag ${ECR_REPO_NAME}:latest ${ECR_URI}:latest

echo "Pushing Docker image to ECR..."
docker push ${ECR_URI}:latest

echo "=========================================="
echo "Docker image pushed successfully!"
echo "=========================================="
echo "ECR URI: ${ECR_URI}:latest"
echo ""
echo "Next steps:"
echo "1. Create an ECS task definition or Elastic Beanstalk application"
echo "2. Deploy using the ECR image URI above"
echo "3. Or use AWS App Runner, Lambda (with container), or ECS Fargate"

