import requests
import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:5000/embed"
IMAGE_PATH = "./images/face1.jpg"
NUM_REQUESTS = 1000
MAX_WORKERS = 8

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n\nShutdown requested (Ctrl+C)... stopping gracefully...")
    shutdown_requested = True

def send_request(image_bytes):
    files = {"image": ("face1.jpg", image_bytes, "image/jpeg")}
    start = time.time()
    response = requests.post(URL, files=files)
    elapsed = time.time() - start
    return elapsed, response.status_code

def benchmark():
    global shutdown_requested
    times = []
    with open(IMAGE_PATH, "rb") as img_file:
        image_bytes = img_file.read()
    total_start = time.time()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(NUM_REQUESTS):
            if shutdown_requested:
                break
            futures.append(executor.submit(send_request, image_bytes))
        
        completed = 0
        for future in as_completed(futures):
            if shutdown_requested:
                break
            elapsed, status = future.result()
            times.append(elapsed)
            completed += 1
            print(f"Request {completed}: {elapsed:.4f}s, Status: {status}")
    
    total_elapsed = time.time() - total_start
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nCompleted {len(times)} requests")
        print(f"Average response time: {avg_time:.4f}s")
        print(f"Total time: {total_elapsed:.4f}s")
    else:
        print("\nNo requests completed")

if __name__ == "__main__":
    try:
        # get URL and NUM_REQUESTS from args if provided
        # python benchmark.py http://100.64.0.4:5000/embed 1000
        if len(sys.argv) > 1:
            URL = sys.argv[1]
        if len(sys.argv) > 2:
            NUM_REQUESTS = int(sys.argv[2])
        benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(0)