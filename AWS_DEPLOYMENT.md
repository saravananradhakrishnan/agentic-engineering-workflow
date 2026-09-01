# AWS Deployment Guide for Multi-Agent Builder

This document describes how to build, containerize, and deploy the `multi-agent-builder` FastAPI application on AWS using Amazon ECR and AWS App Runner or AWS ECS (Fargate).

---

## 1. Prerequisites

1. **AWS CLI** installed and configured (`aws configure`).
2. **Docker** installed and running locally.
3. API keys available:
   - `GROQ_API_KEY` (if using Groq LLMs)
   - `GOOGLE_API_KEY` (if using Gemini LLMs)

---

## 2. Step 1: Push Docker Image to AWS ECR (Elastic Container Registry)

### 1. Set environment variables
```bash
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_NAME="multi-agent-builder"
IMAGE_TAG="latest"
```

### 2. Create ECR Repository (if not already created)
```bash
aws ecr create-repository \
    --repository-name ${ECR_REPO_NAME} \
    --region ${AWS_REGION}
```

### 3. Authenticate Docker to your AWS ECR registry
```bash
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

### 4. Build and tag the Docker image
```bash
docker build -t ${ECR_REPO_NAME}:${IMAGE_TAG} .

docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}
```

### 5. Push the image to AWS ECR
```bash
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}
```

---

## 3. Option A: Deploy to AWS App Runner (Recommended - Simple & Serverless)

AWS App Runner is the easiest way to deploy containerized web applications on AWS with automatic scaling, SSL, and health checks.

### 1. Create App Runner Service via AWS Console or CLI:
- **Source**: ECR Repository
- **Container image**: `${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/multi-agent-builder:latest`
- **Port**: `8000`
- **Environment Variables**:
  - `GROQ_API_KEY`: `<your_groq_api_key>`
  - `GOOGLE_API_KEY`: `<your_google_api_key>`
  - `LLM_PROVIDER`: `auto` (or `groq` / `gemini`)
- **Health Check Configuration**:
  - **Protocol**: HTTP
  - **Path**: `/health`
  - **Interval**: 10 seconds
  - **Timeout**: 5 seconds
  - **Healthy threshold**: 2
  - **Unhealthy threshold**: 5

### 2. Deploy via AWS CLI:
```bash
aws apprunner create-service \
    --service-name multi-agent-builder-service \
    --source-configuration '{
        "AuthenticationConfiguration": {
            "AccessRoleArn": "arn:aws:iam::'${AWS_ACCOUNT_ID}':role/AppRunnerECRAccessRole"
        },
        "ImageRepository": {
            "ImageIdentifier": "'${AWS_ACCOUNT_ID}'.dkr.ecr.'${AWS_REGION}'.amazonaws.com/multi-agent-builder:latest",
            "ImageConfiguration": {
                "Port": "8000",
                "RuntimeEnvironmentVariables": {
                    "GROQ_API_KEY": "'${GROQ_API_KEY}'",
                    "GOOGLE_API_KEY": "'${GOOGLE_API_KEY}'",
                    "LLM_PROVIDER": "auto"
                }
            },
            "ImageRepositoryType": "ECR"
        }
    }'
```

---

## 4. Option B: Deploy to AWS ECS with Fargate

For production environments requiring custom VPC network topologies, AWS ECS Fargate is ideal.

### 1. Create Task Definition (`task-definition.json`)
```json
{
  "family": "multi-agent-builder-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "multi-agent-builder-container",
      "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/multi-agent-builder:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        { "name": "LLM_PROVIDER", "value": "auto" }
      ],
      "secrets": [
        { "name": "GROQ_API_KEY", "valueFrom": "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:GROQ_API_KEY" },
        { "name": "GOOGLE_API_KEY", "valueFrom": "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:GOOGLE_API_KEY" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 10
      }
    }
  ]
}
```

### 2. Target Group Health Check Configuration (for AWS ALB)
- **Target Group Protocol**: HTTP
- **Target Group Port**: 8000
- **Health check path**: `/health`
- **Success code**: `200`

---

## 5. Verification

Once deployed, test your endpoints:

```bash
# 1. Health check URL (returns {"status": "healthy"})
curl https://<YOUR-AWS-SERVICE-URL>/health

# Response:
# {"status":"healthy"}

# 2. Trigger build API
curl -X POST https://<YOUR-AWS-SERVICE-URL>/api/v1/build \
  -H "Content-Type: application/json" \
  -d '{
    "user_requirement": "Create a Python utility module for prime number checks."
  }'
```
