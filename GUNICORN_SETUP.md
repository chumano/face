# Face Embedding API - Gunicorn Production Setup

This document describes how to run the Face Embedding API with Gunicorn for production deployment.

## Quick Start

### Development Mode
```bash
# Linux/Mac
./start.sh -m development

# Windows
.\start.ps1 -Mode development

# Direct Python (development only)
python app.py
```

### Production Mode
```bash
# Linux/Mac
./start.sh -m production -w 4

# Windows  
.\start.ps1 -Mode production -Workers 4

# Direct Gunicorn
gunicorn --config gunicorn_config.py app:app
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Application mode (development/production) |
| `PORT` | `5000` | Server port |
| `GUNICORN_WORKERS` | `auto` | Number of Gunicorn workers |
| `MODEL_SYMBOL_PATH` | `/five/none-symbol.json` | Path to model symbol file |
| `MODEL_PARAMS_PATH` | `/five/none-0000.params` | Path to model params file |
| `QDRANT_URL` | `http://qdrant:6333/...` | Qdrant search URL |
| `USE_GPU` | `true` | Enable GPU acceleration |
| `GPU_ID` | `0` | GPU device ID |
| `BATCH_SIZE` | `1` | Processing batch size |
| `MAX_SEARCH_RESULTS` | `100` | Maximum search results |
| `LOG_LEVEL` | `INFO` | Logging level |

### Gunicorn Configuration

The `gunicorn_config.py` file contains production settings:

- **Workers**: Automatically calculated based on CPU cores
- **Worker Class**: `sync` (suitable for CPU-intensive face processing)
- **Timeout**: 30 seconds per request
- **Max Requests**: 1000 per worker before restart
- **Preload App**: Enabled for better memory usage

### Application Configuration

The `config.py` file provides environment-specific settings:

- **Development**: Debug mode, detailed logging
- **Production**: Optimized settings, structured logging

## Docker Deployment

### Build and Run
```bash
# Build the image
docker build -t face-embedding-api .

# Run with Docker
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e GUNICORN_WORKERS=4 \
  -v /path/to/models:/five \
  face-embedding-api

# Run with Docker Compose
docker-compose up -d
```

### Docker Environment Variables
```yaml
environment:
  - FLASK_ENV=production
  - GUNICORN_WORKERS=4
  - USE_GPU=false  # Set to false if no GPU in container
  - LOG_LEVEL=INFO
```

## Performance Tuning

### Worker Configuration

```bash
# Calculate optimal workers
workers = (2 x CPU_cores) + 1

# For face processing workload
workers = CPU_cores  # Due to GIL and CPU-intensive operations
```

### Memory Considerations

- Each worker loads the full model (~500MB)
- Monitor memory usage: `docker stats` or `htop`
- Adjust workers based on available memory

### GPU Usage

```bash
# Enable GPU
export USE_GPU=true
export GPU_ID=0

# Multiple GPUs (requires code modification)
# Run multiple instances on different ports/GPUs
```

## Monitoring

### Health Checks
```bash
# Application health
curl http://localhost:5000/health

# Docker health check (automatic)
# Configured in Dockerfile with 30s intervals
```

### Logging
```bash
# View live logs
docker-compose logs -f face-api

# Application logs (structured in production)
tail -f /var/log/face-api.log
```

### Metrics
```bash
# Gunicorn stats
curl http://localhost:5000/stats  # If stats endpoint added

# System metrics
docker stats face-embedding-api
```

## Scaling

### Horizontal Scaling
```bash
# Multiple instances with load balancer
# Instance 1
PORT=5001 gunicorn --config gunicorn_config.py app:app

# Instance 2  
PORT=5002 gunicorn --config gunicorn_config.py app:app

# Nginx load balancer configuration in nginx.conf
```

### Vertical Scaling
```bash
# Increase workers
export GUNICORN_WORKERS=8
gunicorn --config gunicorn_config.py app:app

# Increase memory/CPU limits in docker-compose.yml
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce workers or increase memory
   export GUNICORN_WORKERS=2
   ```

2. **GPU Not Available**
   ```bash
   # Fallback to CPU
   export USE_GPU=false
   ```

3. **Model Loading Errors**
   ```bash
   # Check model paths
   export MODEL_SYMBOL_PATH=/path/to/symbol.json
   export MODEL_PARAMS_PATH=/path/to/params.params
   ```

4. **Connection Timeouts**
   ```bash
   # Increase timeout in gunicorn_config.py
   timeout = 60
   ```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export FLASK_ENV=development

# Run with single worker for debugging
gunicorn --workers 1 --timeout 0 app:app
```

## Security Considerations

### Production Security
- Remove debug mode in production
- Use environment variables for secrets
- Enable HTTPS (configure in nginx.conf)
- Implement rate limiting
- Add authentication if needed

### Docker Security
```bash
# Run as non-root user
USER app
WORKDIR /app

# Limit container resources
docker run --memory=2g --cpus=2 face-embedding-api
```

## Testing

### Load Testing
```bash
# Install wrk
sudo apt-get install wrk

# Test embedding endpoint
wrk -t12 -c400 -d30s --timeout 30s \
  -s test_embed.lua \
  http://localhost:5000/embed

# Test search endpoint  
wrk -t12 -c400 -d30s \
  -s test_search.lua \
  http://localhost:5000/search
```

### Integration Testing
```bash
# Run test suite
python -m pytest tests/

# Manual testing
python client_test.py

# Smoke test
./test_gunicorn.sh
```

## Best Practices

1. **Always use Gunicorn in production**
2. **Monitor memory usage per worker**  
3. **Use GPU when available for better performance**
4. **Implement proper logging and monitoring**
5. **Regular health checks and auto-restart**
6. **Load balance across multiple instances for high availability**
7. **Use environment-specific configuration**
8. **Regular model updates and graceful restarts**
