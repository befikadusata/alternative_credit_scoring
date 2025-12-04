# Deployment Guide

This document outlines the process for deploying the credit scoring API to an AWS environment. The deployment is managed using Docker for containerization and Terraform for infrastructure as code.

## 1. Prerequisites

Before you begin, ensure you have the following installed and configured:

*   [Docker](https://www.docker.com/get-started)
*   [Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
*   [AWS CLI](https://aws.amazon.com/cli/), configured with your AWS credentials.

## 2. Deployment Steps

The deployment process consists of two main stages:
1.  Building the Docker image and pushing it to a container registry (AWS ECR).
2.  Using Terraform to provision the AWS infrastructure and deploy the application.

### Step 1: Build and Push the Docker Image

1.  **Log in to AWS ECR:**
    First, you need to authenticate your Docker client with your AWS ECR registry.

    ```bash
    aws ecr get-login-password --region <your-aws-region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<your-aws-region>.amazonaws.com
    ```

2.  **Create the ECR Repository (if it doesn't exist):**
    The Terraform plan will create this, but if you want to push the image before running Terraform, you can create it manually.

    ```bash
    aws ecr create-repository --repository-name credit-scoring-api --image-scanning-configuration scanOnPush=true --region <your-aws-region>
    ```

3.  **Build the Docker Image:**
    From the root of the project, build the Docker image. Replace `<tag>` with a specific version, like `v1.0.0` or a git commit hash.

    ```bash
    docker build -t credit-scoring-api:<tag> .
    ```

4.  **Tag and Push the Image to ECR:**
    Tag the image with the full ECR repository URI and then push it.

    ```bash
    # Tag the image
    docker tag credit-scoring-api:<tag> <your-aws-account-id>.dkr.ecr.<your-aws-region>.amazonaws.com/credit-scoring-api:<tag>

    # Push the image
    docker push <your-aws-account-id>.dkr.ecr.<your-aws-region>.amazonaws.com/credit-scoring-api:<tag>
    ```

### Step 2: Deploy with Terraform

The Terraform scripts in the `terraform/` directory define the necessary infrastructure to run the application on AWS ECS with Fargate.

1.  **Navigate to the Terraform directory:**

    ```bash
    cd terraform
    ```

2.  **Initialize Terraform:**
    This will download the necessary provider plugins.

    ```bash
    terraform init
    ```

3.  **Review the Execution Plan:**
    It's a best practice to review the resources Terraform will create before applying the changes. You will need to provide the image tag you pushed to ECR.

    ```bash
    terraform plan -var="docker_image_tag=<tag>"
    ```

4.  **Apply the Configuration:**
    This will provision the infrastructure on AWS and deploy the application.

    ```bash
    terraform apply -var="docker_image_tag=<tag>" -auto-approve
    ```

    After the apply is complete, Terraform will output the URL of the API endpoint.

### Step 3: Destroying the Infrastructure

To tear down all the resources created by Terraform, use the `destroy` command:

```bash
terraform destroy -auto-approve
```

## 3. Provisioned Infrastructure

The Terraform scripts create the following resources on AWS:

*   **Amazon ECR (Elastic Container Registry):** A private Docker container registry to store the application image.
*   **Amazon ECS (Elastic Container Service) Cluster:** A logical grouping for the application services.
*   **AWS Fargate Task Definition:** A blueprint for the application task, specifying the Docker image, CPU/memory, IAM roles, and port mappings.
*   **AWS Fargate Service:** Manages running and maintaining the specified number of instances of the task definition.
*   **Application Load Balancer (ALB):** Exposes the service to the internet and distributes traffic to the running tasks.
*   **Target Group and Listener:** Connects the ALB to the ECS service.
*   **IAM Roles:** An execution role for the ECS tasks to allow them to pull images from ECR and send logs to CloudWatch.
*   **Security Groups:** Firewall rules to control traffic to the ALB and from the ALB to the ECS service.
*   **CloudWatch Log Group:** To collect and store logs from the running application containers.
