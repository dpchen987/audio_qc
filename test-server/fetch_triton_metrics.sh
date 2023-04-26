#!/bin/bash

if [[ $1 == "--help" || $1 == "-h" ]]; then
    echo "Usage: $0 [TRITON_METRIC_URL]"
    echo "Fetch triton metrics from the TRITON_METRIC_URL every second and save them to gpu_logs/<timestamped>.dat files."
    echo "If no TRITON_METRIC_URL is provided, the default TRITON_METRIC_URL is http://localhost:8002/metrics."
    exit 0
fi

triton_metric_url=${1:-"http://localhost:8002/metrics"}
echo "TRITON_METRIC_URL: $triton_metric_url"

mkdir -p gpu_logs
while true; do
  timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
  filename="${timestamp}.dat"
  curl -s $triton_metric_url > gpu_logs/$filename
  sleep 1
done
