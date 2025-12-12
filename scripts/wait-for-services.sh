#!/bin/bash
# wait-for-services.sh

set -e

# List of services to check (service_name:port)
SERVICES="api:8000 mlflow:5000"
TIMEOUT=120
SLEEP_INTERVAL=5

echo "Waiting for services to be healthy..."

for service in $SERVICES; do
    name="${service%%:*}"
    port="${service##*:}"
    
    echo "Waiting for $name on port $port..."
    
    start_time=$(date +%s)
    while true; do
        if nc -z localhost $port; then
            echo "$name is up!"
            break
        fi
        
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        if [ $elapsed_time -ge $TIMEOUT ]; then
            echo "Timeout waiting for $name."
            exit 1
        fi
        
        sleep $SLEEP_INTERVAL
    done
done

echo "All services are healthy."
exit 0
