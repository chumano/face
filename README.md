# Face Embedding API

A Flask-based REST API for face embedding extraction and similarity search using MXNet face recognition models.

## Features

- **Multiple Input Methods**: Support for file uploads, local file paths, and FTP URLs
- **Face Embedding**: Extract 512-dimensional face embeddings using MXNet models
- **Similarity Search**: Search for similar faces in Qdrant vector database
- **Combined Operations**: Single endpoint for both embedding extraction and search
- **Error Handling**: Comprehensive error handling with meaningful messages

## API Endpoints

### 1. Health Check
```
GET /health
```
Returns the API health status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Face embedding API is running"
}
```

### 2. Extract Embedding
```
POST /embed
```
Extract face embedding from an image.

**Input Methods:**

#### File Upload (multipart/form-data)
```bash
curl -X POST http://localhost:5000/embed \
  -F "image=@./images/face1.jpg"

curl -X POST http://192.168.1.72:5000/embed \
  -F "image=@./images/face1.jpg"
```

#### Local File Path (JSON)
```bash
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.jpg"}'
```

#### FTP URL (JSON)
```bash
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "ftp_url": "ftp://server.com/path/image.jpg",
    "username": "user",
    "password": "pass"
  }'
```

**Response:**
```json
{
  "success": true,
  "source_type": "file_upload",
  "source_info": {"filename": "image.jpg"},
  "embedding": [0.1, 0.2, ...], // 512-dimensional vector
  "embedding_shape": [512]
}
```

### 3. Search Similar Faces
```
POST /search
```
Search for similar faces in the vector database.

**Parameters:**
- Same input methods as `/embed`
- `top` (optional): Number of results to return (1-100, default: 5)

#### File Upload with Top Parameter
```bash
# face1: customer_00019 Ha Le
# face2: customer_00022 Linh Huynh
# face3: customer_00032 Thien Pham
# face4: customer_00023 Thao Nguyen
curl -X POST http://localhost:5000/search \
  -F "image=@./images/face1.jpg" \
  -F "top=3"

curl -X POST http://192.168.1.72:5000/search \
  -F "image=@./images/face1.jpg" \
  -F "top=3"
```

#### JSON with Top Parameter
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "top": 10,
    "embedding": false
  }'
```

**Response:**
```json
{
  "success": true,
  "source_type": "file_path",
  "source_info": {"path": "/path/to/image.jpg"},
  "embedding_shape": [512],
  "top": 5,
  "results": {
    "result": [
      {
        "id": "point_id",
        "score": 0.95,
        "payload": {...}
      }
    ]
  }
}
```

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure Model Files**:
Make sure the MXNet model files are available:
- `./models/`

3. **Start Qdrant**:
Ensure Qdrant vector database is running and accessible at `http://qdrant:6333`

## Usage

### Start the API Server
```bash
python app.py
```
The API will be available at `http://localhost:5000`

### Test the API
```bash
python client_test.py
```

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` for debug mode
- `FLASK_PORT`: Change default port (default: 5000)

### Model Configuration
Update the model paths in `app.py`:
```python
symbol_file = '/path/to/your/symbol.json'
params_file = '/path/to/your/params.params'
```

### Qdrant Configuration
Update the Qdrant URL in `app.py`:
```python
self.qdrant_url = 'http://your-qdrant-host:6333/collections/your-collection/points/search'
```

## Error Handling

The API provides detailed error messages for common issues:

- **400 Bad Request**: Invalid input data, unsupported file types, missing parameters
- **404 Not Found**: File not found for local file paths
- **500 Internal Server Error**: Model errors, Qdrant connection issues

## Supported Image Formats

- PNG
- JPEG/JPG
- GIF
- BMP
- TIFF

## File Size Limits

- Maximum file upload size: 16MB
- Recommended image size: Any size (images are automatically resized to 112x112)

## Security Considerations

1. **File Validation**: Only allowed image formats are accepted
2. **Path Validation**: Use secure filename handling for uploads
3. **FTP Security**: Store FTP credentials securely (consider environment variables)
4. **Input Sanitization**: All inputs are validated before processing

## Development

### Project Structure
```
face/
├── app.py              # Main Flask API
├── face.py             # Original face processing code
├── client_test.py      # API client test script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── images/            # Sample images for testing
```

### Running in Development Mode
```bash
export FLASK_ENV=development
python app.py
```

## Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t face-embedding-api .
docker run -p 5000:5000 face-embedding-api
```

## API Client Examples

### Python Client
```python
import requests

# File upload
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/embed', files={'image': f})

# File path
response = requests.post(
    'http://localhost:5000/search',
    json={'image_path': '/path/to/image.jpg', 'top': 5}
)
```

### cURL Examples
```bash
# Health check
curl http://localhost:5000/health

# File upload embedding
curl -X POST http://localhost:5000/embed -F "image=@image.jpg"

# File path search
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/image.jpg", "top": 3}'
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure model files exist and paths are correct
2. **GPU Issues**: Modify `mx.gpu(0)` to `mx.cpu()` if no GPU available
3. **Qdrant Connection**: Verify Qdrant service is running and accessible
4. **Memory Issues**: Reduce batch size or image resolution for large images

### Test qrant Connection
```bash
curl http://localhost:6333/collections

# in docker
curl http://qdrant:6333/collections
```
```python
import requests
response = requests.get('http://qdrant:6333/collections')
print(response.json())
```

### Logs
Enable debug logging by setting `debug=True` in `app.run()`
