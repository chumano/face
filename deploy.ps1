# Face Embedding API Deployment Script for Windows
# PowerShell version

Write-Host "Face Embedding API Deployment" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-Command "docker")) {
    Write-Host "Error: Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "docker-compose")) {
    Write-Host "Error: Docker Compose is not installed. Please install Docker Compose first." -ForegroundColor Red
    exit 1
}

# Check if model files exist
Write-Host "Checking model files..." -ForegroundColor Yellow
if (-not (Test-Path "./models")) {
    Write-Host "Warning: ./models directory not found. Creating it..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "./models" -Force
    Write-Host "Please copy your model files (face_encoder_symbol.json, face_encoder.params) to ./models/" -ForegroundColor Yellow
}

if ((-not (Test-Path "./models/face_encoder_symbol.json")) -or (-not (Test-Path "./models/face_encoder.params"))) {
    Write-Host "Warning: Model files not found in ./models/" -ForegroundColor Yellow
    Write-Host "Please copy the following files to ./models/:" -ForegroundColor Yellow
    Write-Host "  - face_encoder_symbol.json" -ForegroundColor Yellow
    Write-Host "  - face_encoder.params" -ForegroundColor Yellow
}

# Build and start services
Write-Host "Building and starting services..." -ForegroundColor Yellow
docker-compose down --remove-orphans
docker-compose build
docker-compose up -d

# Wait for services to start
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service health
Write-Host "Checking service health..." -ForegroundColor Yellow

# Check Qdrant
try {
    $response = Invoke-WebRequest -Uri "http://localhost:6333/health" -Method GET -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Qdrant is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Qdrant health check failed" -ForegroundColor Red
}

# Check Face API
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -Method GET -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Face API is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Face API health check failed" -ForegroundColor Red
}

Write-Host ""
Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "- Face API: http://localhost:5000" -ForegroundColor White
Write-Host "- Qdrant: http://localhost:6333" -ForegroundColor White
Write-Host "- Nginx (if enabled): http://localhost:80" -ForegroundColor White
Write-Host ""
Write-Host "To test the API:" -ForegroundColor Cyan
Write-Host "python client_test.py" -ForegroundColor White
Write-Host ""
Write-Host "To start in development mode:" -ForegroundColor Cyan
Write-Host ".\start.ps1 -Mode development" -ForegroundColor White
Write-Host ""
Write-Host "To start in production mode:" -ForegroundColor Cyan
Write-Host ".\start.ps1 -Mode production -Workers 4" -ForegroundColor White
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "docker-compose logs -f face-api" -ForegroundColor White
Write-Host ""
Write-Host "To stop services:" -ForegroundColor Cyan
Write-Host "docker-compose down" -ForegroundColor White
