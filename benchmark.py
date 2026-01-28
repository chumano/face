import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:5000/embed"
IMAGE_PATH = "./images/face1.jpg"
NUM_REQUESTS = 1000
MAX_WORKERS = 10

def send_request(image_bytes):
    files = {"image": ("face1.jpg", image_bytes, "image/jpeg")}
    start = time.time()
    response = requests.post(URL, files=files)
    elapsed = time.time() - start
    return elapsed, response.status_code

def benchmark():
    times = []
    with open(IMAGE_PATH, "rb") as img_file:
        image_bytes = img_file.read()
    total_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(send_request, image_bytes) for _ in range(NUM_REQUESTS)]
        for i, future in enumerate(as_completed(futures), 1):
            elapsed, status = future.result()
            times.append(elapsed)
            print(f"Request {i}: {elapsed:.4f}s, Status: {status}")
    total_elapsed = time.time() - total_start
    avg_time = sum(times) / len(times)
    print(f"\nAverage response time over {NUM_REQUESTS} requests: {avg_time:.4f}s")
    print(f"Total time for {NUM_REQUESTS} requests: {total_elapsed:.4f}s")

if __name__ == "__main__":
    benchmark()