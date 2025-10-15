#!/bin/bash

# Test script for Face Embedding API with Gunicorn

echo "Testing Face Embedding API with Gunicorn..."

# Set test environment variables
export FLASK_ENV=production
export GUNICORN_WORKERS=2
export PORT=5000

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "Error: Gunicorn is not installed. Installing..."
    pip install gunicorn
fi

# Start the server in the background
echo "Starting Gunicorn server..."
gunicorn --config gunicorn_config.py app:app &
GUNICORN_PID=$!

# Wait for server to start
sleep 5

# Test health endpoint
echo "Testing health endpoint..."
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✓ Health endpoint working"
else
    echo "✗ Health endpoint failed"
fi

# Test with a sample request (if you have images)
echo "Test completed. Server is running with PID: $GUNICORN_PID"
echo "To test manually:"
echo "curl http://localhost:5000/health"
echo ""
echo "To stop the server:"
echo "kill $GUNICORN_PID"

# Keep script running to show logs
echo "Press Ctrl+C to stop the server and exit..."
wait $GUNICORN_PID
