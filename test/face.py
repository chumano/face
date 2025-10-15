
import cv2
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.io import DataBatch

class MyEncoder:
    def __init__(self, mod, batch_size=2):
        self.mod = mod # is mx.mod.Module
        self.batch_size = batch_size
        self.ctx = mx.gpu(0)
    def _preprocess_input(self, image):
        if isinstance(image, np.ndarray):
            image = image.astype('float')
            #image = (image - 127.5) / 128.0
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))
        return image
    def __preprocess_input(self, images):
        """Batch preprocessing"""
        batch = []
        for img in images:
            preprocessed = self._preprocess_input(img)
            batch.append(preprocessed)
        return np.array(batch)
    def compute_embedding_images(self, list_aligned_face_images, flip=True):
        embeddings = []
        # Process in batches
        for i in range(0, len(list_aligned_face_images), self.batch_size):
            batch_img = list_aligned_face_images[i:i + self.batch_size]
            # Preprocess
            batch_data = self.__preprocess_input(batch_img)
            # Convert to MXNet array
            data = nd.array(batch_data, ctx=self.ctx)
            # Create data batch
            db = mx.io.DataBatch(data=[data])
            # Forward pass
            self.mod.forward(db, is_train=False)
            # Get output
            net_out = self.mod.get_outputs()
            # Extract embeddings (typically fc1_output or similar)
            embedding = net_out[0].asnumpy()
            if flip:
                # Apply flip augmentation
                flipped_data = nd.flip(data, axis=3)
                db_flip = mx.io.DataBatch(data=[flipped_data])
                self.mod.forward(db_flip, is_train=False)
                net_out_flip = self.mod.get_outputs()
                embedding_flip = net_out_flip[0].asnumpy()
                # Average original and flipped embeddings
                embedding = (embedding + embedding_flip) #/ 2.0
            embeddings.append(embedding)
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


symbol_file = '/five/none-symbol.json'
params_file = '/five/none-0000.params'

# load model
sym = mx.sym.load(symbol_file)
model = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
model.bind(for_training=False, data_shapes=[('data', (1, 3,112, 112))],
                    label_shapes=None, force_rebind=True)

model.load_params(params_file)

e = MyEncoder(model, batch_size=1)

#view model info
dir(model) # mxnet model
model.data_shapes # [('data', (1, 3, 112, 112))]
model.output_shapes # {'fc1_output': (1, 512)}



img_file = '/app/f4r/images/thao.jpg'
img_file = '/app/f4r/images/thao_2.jpg'
img_file = '/app/f4r/images/hien.jpg'
oimg = cv2.imread(img_file)
img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))

# direct compute
rs = e.compute_embedding_images([img])


rs[0].shape # (512,)
rs[0][:20] # first 5 elements

def embed_image(image_path):
    oimg = cv2.imread(image_path)
    img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    rs = e.compute_embedding_images([img])
    return rs

# qdrant search with rs[0] embedding using requests
import requests
import json
url = 'http://qdrant:6333/collections/f4r/points/search'
headers = {'Content-Type': 'application/json'}

rs = embed_image('/app/f4r/images/face1.jpg')
data = {
    "vector": rs[0].tolist(),
    "top": 5,
    "with_payload": True
}
response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()
result

def search_image(image_path, top=5):
    rs = embed_image(image_path)
    data = {
        "vector": rs[0].tolist(),
        "top": top,
        "with_payload": True
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    # pretty print
    print(json.dumps(result, indent=4))
    #return result

search_image('/app/f4r/images/face1.jpg', top=5)
search_image('/app/f4r/images/face2.jpg', top=5)

search_image('/app/f4r/images/thao_2.jpg', top=5)
search_image('/app/f4r/images/loc.jpg', top=5)