variable "aws_region" {
  description = "The AWS region to deploy the resources in."
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "The name of the application."
  type        = string
  default     = "credit-scoring-api"
}

variable "docker_image_tag" {
  description = "The tag of the Docker image to deploy."
  type        = string
  default     = "latest"
}

variable "vpc_id" {
  description = "The ID of the VPC to deploy the resources in. If not provided, the default VPC will be used."
  type        = string
  default     = null
}

variable "subnet_ids" {
  description = "A list of subnet IDs to deploy the resources in. If not provided, default subnets will be used."
  type        = list(string)
  default     = []
}
