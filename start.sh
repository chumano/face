#!/bin/bash

# Face Embedding API Startup Script

set -e

# Default values
MODE="development"
PORT=5000
WORKERS=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -m, --mode MODE     Set mode (development|production) [default: development]"
            echo "  -p, --port PORT     Set port number [default: 5000]"
            echo "  -w, --workers NUM   Set number of workers for production [default: 4]"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Face Embedding API..."
echo "Mode: $MODE"
echo "Port: $PORT"

# Set environment variables
export FLASK_ENV=$MODE
export FLASK_PORT=$PORT

if [ "$MODE" = "development" ]; then
    echo "Starting in development mode with Flask dev server..."
    python app.py
elif [ "$MODE" = "production" ]; then
    echo "Starting in production mode with Gunicorn ($WORKERS workers)..."
    gunicorn --bind 0.0.0.0:$PORT --workers $WORKERS --config gunicorn_config.py app:app
else
    echo "Error: Invalid mode. Use 'development' or 'production'"
    exit 1
fi
