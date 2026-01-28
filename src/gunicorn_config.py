# Gunicorn configuration file
import multiprocessing

# CUDA: Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading: initialization error"
# https://github.com/apache/mxnet/issues/17826


# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
import os
workers = int(os.getenv('GUNICORN_WORKERS', 2))
worker_class = "sync"

worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Restart workers after this many requests, with up to max_requests_jitter added
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "face-embedding-api"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Application
# The Python module and application callable
# This should point to your Flask app instance
# Format: module_name:variable_name
wsgi_module = "app:app"

# Worker warmup hook
def post_worker_init(worker):
    """
    Called just after a worker has been initialized.
    Use this to warm up the face service (load model into memory/GPU).
    """
    from app import get_face_service
    worker.log.info(f"Worker {worker.pid}: Warming up face service...")
    try:
        face_service = get_face_service()
        worker.log.info(f"Worker {worker.pid}: Face service initialized successfully")
        
       
    except Exception as e:
        worker.log.error(f"Worker {worker.pid}: Failed to warm up face service: {e}")