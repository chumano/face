To install cuDNN for CUDA 11.2 in your Dockerfile, add these lines after your CUDA setup:
# Install cuDNN for CUDA 11.2
apt-get update && \
    apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/cudnn-11.2-linux-x64-v8.1.1.33.tgz && \
    tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz && \
    cp cuda/include/cudnn*.h /usr/local/cuda/include && \
    cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
    rm -rf cuda cudnn-11.2-linux-x64-v8.1.1.33.tgz

import ctypes
ctypes.CDLL('libcudnn.so')

python3 -c "import ctypes; ctypes.CDLL('libcudnn.so')"

ls /usr/local/cuda/lib64 | grep libcudnn
ls /usr/local/cuda/lib64 | grep libcusolver

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

https://pypi.org/project/mxnet-cu112/