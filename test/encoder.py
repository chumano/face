

import cv2
import e_service
import mxnet as mx
import numpy as np
from processing.face_aligner import FaceAligner


e = e_service.Encoder(mx ,0, '/app/f4r/model/encoder/model')

#view e info
e.image_size # 112x112
dir(e.model) # mxnet model
e.model.data_shapes # [('data', (1, 3, 112, 112))]
e.model.output_shapes # {'fc1_output': (1, 512)}

#e.model.save_checkpoint('/five/none', 0)

fa = FaceAligner()

img_file = '/app/f4r/images/thao.jpg'
img_file = '/app/f4r/images/thao_2.jpg'
img_file = '/app/f4r/images/hien.jpg'
oimg = cv2.imread(img_file)
img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))

# direct compute
rs = e.compute_embedding_images([img])

# align face then compute
img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
aimg = fa.preprocess(img)
rs = e.compute_embedding_images([aimg])

rs[0].shape # (512,)
rs[0][:20] # first 5 elements

# qdrant search with rs[0] embedding using requests
import requests
import json
url = 'http://qdrant:6333/collections/f4r/points/search'
headers = {'Content-Type': 'application/json'}

data = {
    "vector": rs[0].tolist(),
    "top": 5,
    "with_payload": True
}
response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()
result
