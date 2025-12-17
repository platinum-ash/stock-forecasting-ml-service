#!/bin/bash
set -e

echo " Building forecasting service..."
podman build -f Dockerfile -t forecasting-service:latest .

echo "Stopping and removing old container..."
podman stop forecasting 2>/dev/null || true
podman rm forecasting 2>/dev/null || true

echo "Starting forecasting service..."
podman run -d \
  --name forecasting \
  --network app-net \
  -e DATABASE_URL="postgresql+psycopg2://tsuser:ts_password@timescaledb:5432/timeseries" \
  -e PREPROCESSING_SERVICE_URL="http://preprocessing:8000" \
  -p 8001:8001 \
  forecasting-service:latest

echo "✅ Forecasting service started!"

podman logs -f forecasting
