output "api_endpoint_url" {
  description = "The URL of the deployed API endpoint."
  value       = "http://${aws_lb.main.dns_name}"
}
