# AWS Infrastructure for DL_Bayesian

This directory contains Infrastructure as Code (IaC) for deploying the DL_Bayesian application to AWS.

## Architecture Overview

```
                    ┌─────────────────┐
                    │   Route 53      │
                    │   (DNS)         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   CloudFront    │
                    │   (CDN)         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   ALB           │
                    │   (Load Balancer)│
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│   ECS Fargate   │ │   ECS Fargate   │ │   ECS Fargate   │
│   (API)         │ │   (API)         │ │   (MLFlow)      │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │   EFS           │
                    │   (MLFlow Data) │
                    └─────────────────┘
```

## Prerequisites

1. AWS CLI installed and configured
2. Docker installed
3. Appropriate IAM permissions
4. VPC with public and private subnets

## Deployment Steps

### 1. Create ECR Repositories

```bash
chmod +x infrastructure/ecr/setup-ecr.sh
./infrastructure/ecr/setup-ecr.sh
```

### 2. Build and Push Docker Images

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push API image
docker build -t dl-bayesian-api -f docker/api/Dockerfile .
docker tag dl-bayesian-api:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/dl-bayesian-api:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/dl-bayesian-api:latest

# Build and push MLFlow image
docker build -t dl-bayesian-mlflow -f docker/mlflow/Dockerfile docker/mlflow/
docker tag dl-bayesian-mlflow:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/dl-bayesian-mlflow:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/dl-bayesian-mlflow:latest
```

### 3. Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name dl-bayesian-cluster --region us-east-1
```

### 4. Register Task Definitions

```bash
# Update ACCOUNT_ID and REGION in task definitions first
aws ecs register-task-definition --cli-input-json file://infrastructure/ecs/task-definition-api.json
aws ecs register-task-definition --cli-input-json file://infrastructure/ecs/task-definition-mlflow.json
```

### 5. Create ECS Services

```bash
# Create MLFlow service first
aws ecs create-service \
    --cluster dl-bayesian-cluster \
    --service-name dl-bayesian-mlflow \
    --task-definition dl-bayesian-mlflow \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"

# Create API service
aws ecs create-service \
    --cluster dl-bayesian-cluster \
    --service-name dl-bayesian-api \
    --task-definition dl-bayesian-api \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}" \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=dl-bayesian-api,containerPort=8000"
```

## Security Groups

### API Security Group
- Inbound: 8000 from ALB Security Group
- Outbound: 5000 to MLFlow Security Group, 443 to ECR

### MLFlow Security Group
- Inbound: 5000 from API Security Group
- Outbound: 443 to S3 (artifacts), 2049 to EFS

### ALB Security Group
- Inbound: 80, 443 from 0.0.0.0/0
- Outbound: 8000 to API Security Group

## Estimated Costs

| Resource | Configuration | Est. Monthly Cost |
|----------|--------------|-------------------|
| ECS Fargate (API) | 2 x 0.5 vCPU, 1GB | ~$30 |
| ECS Fargate (MLFlow) | 1 x 0.25 vCPU, 0.5GB | ~$10 |
| ALB | Standard | ~$20 |
| EFS | 10GB | ~$3 |
| CloudWatch Logs | 10GB | ~$5 |
| **Total** | | **~$68/month** |

## Monitoring

- CloudWatch Logs: `/ecs/dl-bayesian-api`, `/ecs/dl-bayesian-mlflow`
- CloudWatch Metrics: ECS Service metrics
- Health checks: `/health` endpoint

## Troubleshooting

### Check service status
```bash
aws ecs describe-services --cluster dl-bayesian-cluster --services dl-bayesian-api
```

### View logs
```bash
aws logs tail /ecs/dl-bayesian-api --follow
```

### Force new deployment
```bash
aws ecs update-service --cluster dl-bayesian-cluster --service dl-bayesian-api --force-new-deployment
```
