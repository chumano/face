# Face Embedding API Startup Script for Windows
# PowerShell version

param(
    [Parameter()]
    [ValidateSet("development", "production")]
    [string]$Mode = "development",
    
    [Parameter()]
    [int]$Port = 5000,
    
    [Parameter()]
    [int]$Workers = 4,
    
    [Parameter()]
    [switch]$Help
)

if ($Help) {
    Write-Host "Face Embedding API Startup Script"
    Write-Host "Usage: .\start.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Port PORT         Set port number [default: 5000]"
    Write-Host "  -Workers NUM       Set number of workers for production [default: 4]"
    Write-Host "  -Help              Show this help message"
    exit 0
}

Write-Host "Starting Face Embedding API..." -ForegroundColor Green
Write-Host "Mode: $Mode" -ForegroundColor Cyan
Write-Host "Port: $Port" -ForegroundColor Cyan

# Set environment variables
$env:FLASK_ENV = $Mode
$env:FLASK_PORT = $Port

if ($Mode -eq "development") {
    Write-Host "Starting in development mode with Flask dev server..." -ForegroundColor Yellow
    python src/app.py
}else {
    Write-Host "Error: Invalid mode. Use 'development' or 'production'" -ForegroundColor Red
    exit 1
}
