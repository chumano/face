
# ONNX runtime Face Embedding API

NOTWORKING CURRENTLY. PLEASE USE MXNET VERSION INSTEAD.

## Prerequisites

- python 3.10+
```bash
pip install mxnet==1.9.1

pip install onnx==1.12.0
pip install numpy==1.23.5

# inference engine
pip install onnxruntime==1.19.2
1.12.1
# cv2
pip install opencv-python==4.12.0
```

onnx==1.8.0
onnxruntime==1.7.0
protobuf==3.14.0

apt update
apt-get install -y libgl1


https://github.com/apache/mxnet/blob/v1.9.1/python/mxnet/onnx/mx2onnx/_export_model.py

## Fix ONNX Model for MXNet Exporter Issues
https://github.com/microsoft/onnxruntime/issues/3205
```bash
python onnx/fix_mx_onnx.py
```

## Check verion
python -c "import onnx; print(onnx.__version__)"
python -c "import onnxruntime; print(onnxruntime.__version__)"
python -c "import mxnet; print(mxnet.__version__)"
python -c "import numpy; print(numpy.__version__)"
