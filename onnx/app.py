import onnxruntime as ort
import cv2
import numpy as np

model_file = '../models/face_encoder.onnx'
image_file = '../images/thao.jpg'

# Load the ONNX model
# onnx_model = onnx.load(model_file)
# onnx.checker.check_model(onnx_model)

# Create ONNX Runtime session
session = ort.InferenceSession(model_file)

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Model input name: {input_name}")
print(f"Model input shape: {input_shape}")

# Load and preprocess the image
def preprocess_image(image_path, target_size=(112, 112)):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to target size
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0, 1] or [-1, 1] depending on model
    img = img.astype(np.float32) / 255.0
    
    # Some models expect values in [-1, 1] range
    img = (img - 0.5) / 0.5
    
    # Transpose to CHW format (channels, height, width)
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    
    # Model expects batch size of 2, so duplicate the image
    batch_data = np.stack([img, img], axis=0)  # Shape: [2, 3, 112, 112]
    
    return batch_data

# Generate face embedding
def get_face_embedding(image_path):
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    # Extract embedding (usually the first output)
    # Since we duplicated the image, take the first embedding from the batch
    embedding = outputs[0][0]  # Take first item from batch of 2
    
    # Flatten if needed and normalize
    embedding = embedding.flatten()
    
    # L2 normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

# Generate embedding for the input image
try:
    embedding = get_face_embedding(image_file)
    print(f"Face embedding shape: {embedding.shape}")
    print(f"Face embedding (first 10 values): {embedding[:10]}")
    print(f"Embedding norm: {np.linalg.norm(embedding)}")
    
    # Save embedding to file
    embedding_file = image_file.replace('.jpg', '_embedding.npy')
    np.save(embedding_file, embedding)
    print(f"Embedding saved to: {embedding_file}")
    
except Exception as e:
    print(f"Error generating embedding: {e}")

