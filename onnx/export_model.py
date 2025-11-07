import mxnet as mx
import numpy as np
import onnx
import sys
from packaging import version

print("mxnet version:", mx.__version__)
print("onnx version:", onnx.__version__)
print("numpy version:", np.__version__)

# Check ONNX version compatibility
if version.parse(onnx.__version__) >= version.parse("1.13.0"):
    print("Error: ONNX version >= 1.13.0 is not compatible with MXNet's ONNX exporter.")
    print("Please install onnx==1.12.0 for export to work:")
    print("    pip install onnx==1.12.0")
    sys.exit(1)

if not hasattr(mx, "onnx"):
    print("Error: mxnet.onnx module is not available in your MXNet installation.")
    print("Or see: https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/deploy/export/onnx.html")
    exit(1)


#help(mx.onnx.export_model)

onnx_file = 'face_encoder.onnx'

symbol_file = '../models/face_encoder_symbol.json'
params_file = '../models/face_encoder.params'

# load model
sym = mx.sym.load(symbol_file)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3,112, 112))],
                    label_shapes=None, force_rebind=True)
mod.load_params(params_file)
  
arg_params, aux_params = mod.get_params()

# Export the model to ONNX format
input_shape = (1, 3, 112, 112)  # Batch size of 1, 3 color channels (RGB), 112x112 image size
mx.onnx.export_model(sym, [arg_params,aux_params] , [input_shape], 
                     np.float32, 
                     onnx_file, 
                     dynamic=False,
                     run_shape_inference=True,
                     verbose=True)

print(f"Model exported to {onnx_file}")
# Verify the ONNX model
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# print opset version
print("ONNX model opset version:", onnx_model.opset_import[0].version)


def remove_initializer_from_input(onnx_path,save_new_path):
    """
        @param onnx_path:           exported onnx model path by pytroch
        @param save_new_path:       clean onnx model path after remove input initialzer
    """
    model = onnx.load(onnx_path)
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initilizer in graph input")
        return
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(model, save_new_path)

#remove_initializer_from_input(onnx_file, onnx_file)