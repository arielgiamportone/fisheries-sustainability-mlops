#!/bin/bash
# ==============================================
# ECR Repository Setup Script for DL_Bayesian
# ==============================================
#
# Usage:
#   chmod +x infrastructure/ecr/setup-ecr.sh
#   ./infrastructure/ecr/setup-ecr.sh
#
# Prerequisites:
#   - AWS CLI installed and configured
#   - Appropriate IAM permissions

set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="dl-bayesian"

echo "=============================================="
echo "Setting up ECR repositories for DL_Bayesian"
echo "Region: $AWS_REGION"
echo "=============================================="

# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $AWS_ACCOUNT_ID"

# Repository names
API_REPO="${PROJECT_NAME}-api"
MLFLOW_REPO="${PROJECT_NAME}-mlflow"

# Create API repository
echo ""
echo "Creating repository: $API_REPO"
aws ecr create-repository \
    --repository-name "$API_REPO" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    --region "$AWS_REGION" \
    2>/dev/null || echo "Repository $API_REPO already exists"

# Create MLFlow repository
echo "Creating repository: $MLFLOW_REPO"
aws ecr create-repository \
    --repository-name "$MLFLOW_REPO" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    --region "$AWS_REGION" \
    2>/dev/null || echo "Repository $MLFLOW_REPO already exists"

# Create lifecycle policy to keep only last 10 images
echo ""
echo "Setting lifecycle policies..."

LIFECYCLE_POLICY='{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep last 10 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 10
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}'

aws ecr put-lifecycle-policy \
    --repository-name "$API_REPO" \
    --lifecycle-policy-text "$LIFECYCLE_POLICY" \
    --region "$AWS_REGION"

aws ecr put-lifecycle-policy \
    --repository-name "$MLFLOW_REPO" \
    --lifecycle-policy-text "$LIFECYCLE_POLICY" \
    --region "$AWS_REGION"

# Output repository URIs
echo ""
echo "=============================================="
echo "ECR Repositories Created Successfully!"
echo "=============================================="
echo ""
echo "API Repository:"
echo "  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${API_REPO}"
echo ""
echo "MLFlow Repository:"
echo "  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${MLFLOW_REPO}"
echo ""
echo "To push images, first authenticate:"
echo "  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo ""
