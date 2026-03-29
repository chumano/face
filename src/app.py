from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import json
import os
import logging
import ftplib
import io
from urllib.parse import urlparse
from werkzeug.utils import secure_filename


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

class ImageLoader:
    """Handles loading and preprocessing images from various sources."""

    IMAGE_SIZE = (112, 112)

    @staticmethod
    def _postprocess(bgr_image):
        """Convert BGR→RGB and resize to model input size."""
        img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, ImageLoader.IMAGE_SIZE)

    @staticmethod
    def _decode_bytes(raw_bytes):
        """Decode raw bytes to a BGR image array."""
        img_array = np.frombuffer(raw_bytes, np.uint8)
        oimg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if oimg is None:
            raise ValueError("Unable to decode image data")
        return oimg

    @staticmethod
    def from_path(image_path):
        """Load image from a local file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        oimg = cv2.imread(image_path)
        if oimg is None:
            raise ValueError(f"Unable to load image from: {image_path}")
        return ImageLoader._postprocess(oimg)

    @staticmethod
    def from_ftp(ftp_url, username=None, password=None):
        """Load image from an FTP URL."""
        try:
            parsed = urlparse(ftp_url)
            if parsed.scheme != 'ftp':
                raise ValueError("Invalid FTP URL: scheme must be 'ftp'")

            ftp = ftplib.FTP()
            ftp.connect(parsed.hostname, parsed.port or 21)
            ftp.login(username or '', password or '')

            buf = io.BytesIO()
            ftp.retrbinary(f'RETR {parsed.path}', buf.write)
            ftp.quit()

            return ImageLoader._postprocess(ImageLoader._decode_bytes(buf.getvalue()))
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise ValueError(f"Error loading image from FTP: {e}")

    @staticmethod
    def from_file_upload(file):
        """Load image from a Werkzeug file upload object."""
        try:
            return ImageLoader._postprocess(ImageLoader._decode_bytes(file.read()))
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error processing uploaded image: {e}")


# ---------------------------------------------------------------------------
# Request parsing helper
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_image_request(req, default_top=5):
    """
    Parse an incoming Flask request and return (img, source_type, source_info, top).

    Raises ValueError / FileNotFoundError on bad input so callers can handle
    them uniformly.
    """
    top = default_top

    if 'image' in req.files:
        file = req.files['image']
        if not file.filename:
            raise ValueError("No file selected")
        if not _allowed_file(file.filename):
            raise ValueError("Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff")

        img = ImageLoader.from_file_upload(file)
        source_type = "file_upload"
        source_info = {"filename": secure_filename(file.filename)}

        if 'top' in req.form:
            try:
                top = int(req.form['top'])
            except ValueError:
                raise ValueError("Invalid 'top' parameter. Must be an integer")

    elif req.is_json:
        data = req.get_json()

        if 'top' in data:
            try:
                top = int(data['top'])
            except (ValueError, TypeError):
                raise ValueError("Invalid 'top' parameter. Must be an integer")

        if 'image_path' in data:
            img = ImageLoader.from_path(data['image_path'])
            source_type = "file_path"
            source_info = {"path": data['image_path']}

        elif 'ftp_url' in data:
            img = ImageLoader.from_ftp(
                data['ftp_url'],
                username=data.get('username'),
                password=data.get('password'),
            )
            source_type = "ftp_url"
            source_info = {"url": data['ftp_url']}

        else:
            raise ValueError("Missing 'image_path' or 'ftp_url' in JSON data")

    else:
        raise ValueError(
            "No image data provided. Use file upload or JSON with image_path/ftp_url"
        )

    return img, source_type, source_info, top


# ---------------------------------------------------------------------------
# Model / embedding
# ---------------------------------------------------------------------------

class FaceEncoder:
    def __init__(self, mod, batch_size=2, context=None):
        # Lazy import mxnet
        global mx, nd
        import mxnet as mx
        from mxnet import nd
        self.mod = mod # is mx.mod.Module
        self.batch_size = batch_size
        self.ctx = context or mx.gpu(0)

    def _preprocess_input(self, image):
        if isinstance(image, np.ndarray):
            image = image.astype('float')
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))
        return image

    def __preprocess_batch(self, images):
        return np.array([self._preprocess_input(img) for img in images])

    def compute_embedding_images(self, list_aligned_face_images, flip=True):
        global nd
        from mxnet import nd
        embeddings = []
        for i in range(0, len(list_aligned_face_images), self.batch_size):
            batch_img = list_aligned_face_images[i:i + self.batch_size]
            data = nd.array(self.__preprocess_batch(batch_img), ctx=self.ctx)
            db = mx.io.DataBatch(data=[data])
            self.mod.forward(db, is_train=False)
            embedding = self.mod.get_outputs()[0].asnumpy()
            if flip:
                db_flip = mx.io.DataBatch(data=[nd.flip(data, axis=3)])
                self.mod.forward(db_flip, is_train=False)
                embedding = embedding + self.mod.get_outputs()[0].asnumpy()
            embeddings.append(embedding)
        return np.concatenate(embeddings, axis=0)


class FaceEmbeddingService:
    def __init__(self, config):
        global mx, nd
        import mxnet as mx
        from mxnet import nd

        symbol_file = config.get('MODEL_SYMBOL_PATH')
        params_file = config.get('MODEL_PARAMS_PATH')
        use_gpu = config.get('USE_GPU', True)
        gpu_id = config.get('GPU_ID', 0)
        print("Using GPU:", use_gpu, "GPU ID:", gpu_id)
        context = mx.gpu(gpu_id) if use_gpu else mx.cpu()

        sym = mx.sym.load(symbol_file)
        model = mx.mod.Module(symbol=sym, context=context, label_names=None)
        model.bind(
            for_training=False,
            data_shapes=[('data', (1, 3, 112, 112))],
            label_shapes=None,
            force_rebind=True,
        )
        model.load_params(params_file)
        print(f"Process {os.getpid()}: Model loaded successfully.")

        batch_size = config.get('BATCH_SIZE', 1)
        self.encoder = FaceEncoder(model, batch_size=batch_size, context=context)
        self.qdrant_url = config.get('QDRANT_URL', 'http://qdrant:6333/collections/f4r/points/search')
        self.headers = {'Content-Type': 'application/json'}
        self.max_search_results = config.get('MAX_SEARCH_RESULTS', 10)

    def compute_embedding(self, img):
        """Compute embedding for a single image."""
        return self.encoder.compute_embedding_images([img])[0]

    def search_similar_faces(self, embedding, top=5):
        """Search for similar faces in Qdrant."""
        payload = {
            "vector": embedding.tolist(),
            "top": top,
            "with_payload": True,
        }
        response = requests.post(self.qdrant_url, headers=self.headers, data=json.dumps(payload))
        return response.json()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

from config import config as app_config
config_name = os.getenv('FLASK_ENV', 'development')
app.config.from_object(app_config[config_name])

log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
logging.basicConfig(
    level=log_level,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
)
app.logger.setLevel(log_level)
app.logger.info(f'Face embedding API startup in {config_name} mode')


def get_face_service():
    """Lazily create and cache the FaceEmbeddingService instance."""
    if not hasattr(get_face_service, '_instance'):
        get_face_service._instance = FaceEmbeddingService(app.config)
    return get_face_service._instance


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Face embedding API is running"})


@app.route('/embed', methods=['POST'])
def embed_image():
    """
    Compute embedding for an image.
    Accepts: file upload (multipart 'image'), JSON {'image_path'}, or JSON {'ftp_url'}.
    """
    try:
        img, source_type, source_info, _ = parse_image_request(request)
        app.logger.info(f"Computing embedding for source type: {source_type}")
        embedding = get_face_service().compute_embedding(img)
        return jsonify({
            "success": True,
            "source_type": source_type,
            "source_info": source_info,
            "embedding": embedding.tolist(),
            "embedding_shape": embedding.shape,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Compute embedding and return similar faces from Qdrant.
    Accepts an optional 'top' parameter (default 5).
    """
    try:
        face_service = get_face_service()
        img, source_type, source_info, top = parse_image_request(request)

        if not (1 <= top <= face_service.max_search_results):
            return jsonify({
                "error": f"Parameter 'top' must be between 1 and {face_service.max_search_results}"
            }), 400

        embedding = face_service.compute_embedding(img)
        search_results = face_service.search_similar_faces(embedding, top)

        return jsonify({
            "success": True,
            "source_type": source_type,
            "source_info": source_info,
            "top": top,
            "search_results": search_results,
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)