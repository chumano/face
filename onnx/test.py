import onnxruntime as ort
import numpy as np

# Load the ONNX model
model_file = "../models/face_encoder.onnx"
model_file = "../models/onnxruntime.onnx"
session = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])

# Prepare input data (replace with your actual image data)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Model input name: {input_name}")
print(f"Model input shape: {input_shape}")
# Example: random input, replace with preprocessed image
#input_data = np.random.rand(*[d if d else 1 for d in input_shape]).astype(np.float32)
input_data =  np.random.randint(0, 256, size=(1, 3, 112, 112)).astype(np.float32)

raw_image = np.random.randint(0, 256, size=(112, 112, 3)).astype(np.float32)  # HWC format
# Preprocess: normalize to [-1, 1] and transpose to NCHW
normalized_image = (raw_image - 127.5) / 128.0
input_data = np.transpose(normalized_image, (2, 0, 1))[np.newaxis, ...]  # Add batch dim: (1, 3, 112, 112)


input = {input_name: input_data}
print("Input keys:", input.keys())
# Run inference
print(f"Input data shape: {input_data.shape}")
print(f"Input data range: [{input_data.min()}, {input_data.max()}]")

print("Running inference...")
# Run inference
try:
    outputs = session.run(None, input)
    
    # Print output
    print("Output shape:", outputs[0].shape)
    print("Output (first 10 values):", outputs[0][0][:10])
except Exception as e:
    print(f"Error during inference: {e}")

