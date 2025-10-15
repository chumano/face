#!/bin/bash

# Face Embedding API Deployment Script

set -e

echo "Face Embedding API Deployment"
echo "============================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists docker; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if model files exist
echo "Checking model files..."
if [ ! -d "./models" ]; then
    echo "Warning: ./models directory not found. Creating it..."
    mkdir -p ./models
    echo "Please copy your model files (face_encoder_symbol.json, face_encoder.params) to ./models/"
fi

if [ ! -f "./models/face_encoder_symbol.json" ] || [ ! -f "./models/face_encoder.params" ]; then
    echo "Warning: Model files not found in ./models/"
    echo "Please copy the following files to ./models/:"
    echo "  - face_encoder_symbol.json"
    echo "  - face_encoder.params"
fi

# Build and start services
echo "Building and starting services..."
docker-compose down --remove-orphans
docker-compose build
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check service health
echo "Checking service health..."

# Check Qdrant
if curl -f http://localhost:6333/health >/dev/null 2>&1; then
    echo "✓ Qdrant is healthy"
else
    echo "✗ Qdrant health check failed"
fi

# Check Face API
if curl -f http://localhost:5000/health >/dev/null 2>&1; then
    echo "✓ Face API is healthy"
else
    echo "✗ Face API health check failed"
fi

echo ""
echo "Deployment complete!"
echo ""
echo "Service URLs:"
echo "- Face API: http://localhost:5000"
echo "- Qdrant: http://localhost:6333"
echo "- Nginx (if enabled): http://localhost:80"
echo ""
echo "To test the API:"
echo "python client_test.py"
echo ""
echo "To start in development mode:"
echo "./start.sh -m development"
echo ""
echo "To start in production mode:"
echo "./start.sh -m production -w 4"
echo ""
echo "To view logs:"
echo "docker-compose logs -f face-api"
echo ""
echo "To stop services:"
echo "docker-compose down"
