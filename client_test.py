import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_embed_file_upload():
    """Test embedding with file upload"""
    print("Testing file upload embedding...")
    
    # Replace with actual image path
    image_path = "images/face1.jpg"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/embed", files=files)
        
        print("File Upload Embedding Result:")
        result = response.json()
        if 'embedding' in result:
            result['embedding'] = f"[{len(result['embedding'])} values]"  # Don't print full embedding
        print(json.dumps(result, indent=2))
        print()
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print()

def test_embed_file_path():
    """Test embedding with file path"""
    print("Testing file path embedding...")
    
    data = {
        "image_path": "/app/f4r/images/thao.jpg"
    }
    
    response = requests.post(
        f"{BASE_URL}/embed", 
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("File Path Embedding Result:")
    result = response.json()
    if 'embedding' in result:
        result['embedding'] = f"[{len(result['embedding'])} values]"  # Don't print full embedding
    print(json.dumps(result, indent=2))
    print()

def test_embed_ftp():
    """Test embedding with FTP URL"""
    print("Testing FTP embedding...")
    
    data = {
        "ftp_url": "ftp://example.com/path/to/image.jpg",
        "username": "user",  # optional
        "password": "pass"   # optional
    }
    
    response = requests.post(
        f"{BASE_URL}/embed", 
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("FTP Embedding Result:")
    result = response.json()
    print(json.dumps(result, indent=2))
    print()

def test_search_file_upload():
    """Test search with file upload"""
    print("Testing file upload search...")
    
    # Replace with actual image path
    image_path = "images/face2.jpg"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'top': '3'}  # Return top 3 results
            response = requests.post(f"{BASE_URL}/search", files=files, data=data)
        
        print("File Upload Search Result:")
        result = response.json()
        print(json.dumps(result, indent=2))
        print()
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print()

def test_search_file_path():
    """Test search with file path"""
    print("Testing file path search...")
    
    data = {
        "image_path": "/app/f4r/images/thao_2.jpg",
        "top": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/search", 
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    
    print("File Path Search Result:")
    result = response.json()
    print(json.dumps(result, indent=2))
    print()

if __name__ == "__main__":
    print("Face Embedding API Client Test")
    print("=" * 40)
    
    # Test all endpoints
    test_health()
    test_embed_file_path()
    test_search_file_path()
    
    # Uncomment these if you have local images to test
    # test_embed_file_upload()
    # test_search_file_upload()
    
    # Uncomment this if you have FTP access to test
    # test_embed_ftp()
