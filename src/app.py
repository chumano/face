from flask import Flask, request, jsonify
import cv2
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.io import DataBatch
import requests
import json
import os
import tempfile
import logging
from urllib.parse import urlparse
import ftplib
import io
from werkzeug.utils import secure_filename

class MyEncoder:
    def __init__(self, mod, batch_size=2, context=None):
        self.mod = mod # is mx.mod.Module
        self.batch_size = batch_size
        self.ctx = context or mx.gpu(0)
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

class FaceEmbeddingService:
    def __init__(self, config):
        # Initialize model
        symbol_file = config.get('MODEL_SYMBOL_PATH', './models/face_encoder_symbol.json')
        params_file = config.get('MODEL_PARAMS_PATH', './models/face_encoder.params')

        # Determine context (GPU or CPU)
        use_gpu = config.get('USE_GPU', True)
        gpu_id = config.get('GPU_ID', 0)
        context = mx.gpu(gpu_id) if use_gpu else mx.cpu()
        
        # load model
        sym = mx.sym.load(symbol_file)
        model = mx.mod.Module(symbol=sym, context=context, label_names=None)
        model.bind(for_training=False, data_shapes=[('data', (1, 3,112, 112))],
                          label_shapes=None, force_rebind=True)
        model.load_params(params_file)
        
        batch_size = config.get('BATCH_SIZE', 1)
        self.encoder = MyEncoder(model, batch_size=batch_size, context=context)
        self.qdrant_url = config.get('QDRANT_URL', 'http://qdrant:6333/collections/f4r/points/search')
        self.headers = {'Content-Type': 'application/json'}
        self.max_search_results = config.get('MAX_SEARCH_RESULTS', 100)
    
    def load_image_from_path(self, image_path):
        """Load image from local file path"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        oimg = cv2.imread(image_path)
        if oimg is None:
            raise ValueError(f"Unable to load image from: {image_path}")
        
        img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        return img
    
    def load_image_from_ftp(self, ftp_url, username=None, password=None):
        """Load image from FTP URL"""
        try:
            parsed_url = urlparse(ftp_url)
            if parsed_url.scheme != 'ftp':
                raise ValueError("Invalid FTP URL")
            
            ftp = ftplib.FTP()
            ftp.connect(parsed_url.hostname, parsed_url.port or 21)
            
            if username and password:
                ftp.login(username, password)
            else:
                ftp.login()  # Anonymous login
            
            # Download file to memory
            bio = io.BytesIO()
            ftp.retrbinary(f'RETR {parsed_url.path}', bio.write)
            ftp.quit()
            
            # Convert to image
            bio.seek(0)
            img_array = np.frombuffer(bio.getvalue(), np.uint8)
            oimg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if oimg is None:
                raise ValueError("Unable to decode image from FTP")
            
            img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            return img
            
        except Exception as e:
            raise ValueError(f"Error loading image from FTP: {str(e)}")
    
    def load_image_from_file_upload(self, file):
        """Load image from uploaded file"""
        try:
            # Read file content
            file_content = file.read()
            
            # Convert to numpy array and decode
            img_array = np.frombuffer(file_content, np.uint8)
            oimg = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if oimg is None:
                raise ValueError("Unable to decode uploaded image")
            
            img = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            return img
            
        except Exception as e:
            raise ValueError(f"Error processing uploaded image: {str(e)}")
    
    def compute_embedding(self, img):
        """Compute embedding for a single image"""
        rs = self.encoder.compute_embedding_images([img])
        return rs[0]
    
    def search_similar_faces(self, embedding, top=5):
        """Search for similar faces in Qdrant"""
        data = {
            "vector": embedding.tolist(),
            "top": top,
            "with_payload": True
        }
        response = requests.post(self.qdrant_url, headers=self.headers, data=json.dumps(data))
        return response.json()

# Initialize Flask app
app = Flask(__name__)

# Load configuration
from config import config
config_name = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Configure logging
log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
logging.basicConfig(
    level=log_level,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
app.logger.setLevel(log_level)
app.logger.info(f'Face embedding API startup in {config_name} mode')

# Initialize face embedding service
face_service = FaceEmbeddingService(app.config)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Face embedding API is running"})

@app.route('/embed', methods=['POST'])
def embed_image():
    """
    Compute embedding for an image
    Supports:
    - File upload (multipart/form-data with 'image' field)
    - Local file path (JSON with 'image_path' field)
    - FTP URL (JSON with 'ftp_url' field and optional 'username', 'password')
    """
    try:
        # Check if it's a file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            if not allowed_file(file.filename):
                return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff"}), 400
            
            img = face_service.load_image_from_file_upload(file)
            source_type = "file_upload"
            source_info = {"filename": secure_filename(file.filename)}
        
        # Check for JSON data with image_path or ftp_url
        elif request.is_json:
            data = request.get_json()
            
            if 'image_path' in data:
                img = face_service.load_image_from_path(data['image_path'])
                source_type = "file_path"
                source_info = {"path": data['image_path']}
            
            elif 'ftp_url' in data:
                username = data.get('username')
                password = data.get('password')
                img = face_service.load_image_from_ftp(data['ftp_url'], username, password)
                source_type = "ftp_url"
                source_info = {"url": data['ftp_url']}
            
            else:
                return jsonify({"error": "Missing 'image_path' or 'ftp_url' in JSON data"}), 400
        
        else:
            return jsonify({"error": "No image data provided. Use file upload or JSON with image_path/ftp_url"}), 400
        
        # Compute embedding
        embedding = face_service.compute_embedding(img)
        
        return jsonify({
            "success": True,
            "source_type": source_type,
            "source_info": source_info,
            "embedding": embedding.tolist(),
            "embedding_shape": embedding.shape
        })
    
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def handle_embed_and_search(request):
    """
    Unified handler for /search and /embed_and_search endpoints
    """
    top = 5  # default value
    try:
        # Check if it's a file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return None, None, None, None, jsonify({"error": "No file selected"}), 400
            if not allowed_file(file.filename):
                return None, None, None, None, jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff"}), 400
            img = face_service.load_image_from_file_upload(file)
            source_type = "file_upload"
            source_info = {"filename": secure_filename(file.filename)}
            # Get top parameter from form data if available
            if 'top' in request.form:
                try:
                    top = int(request.form['top'])
                except ValueError:
                    return None, None, None, None, jsonify({"error": "Invalid 'top' parameter. Must be an integer"}), 400
        # Check for JSON data
        elif request.is_json:
            data = request.get_json()
            # Get top parameter
            if 'top' in data:
                try:
                    top = int(data['top'])
                except (ValueError, TypeError):
                    return None, None, None, None, jsonify({"error": "Invalid 'top' parameter. Must be an integer"}), 400
            if 'image_path' in data:
                img = face_service.load_image_from_path(data['image_path'])
                source_type = "file_path"
                source_info = {"path": data['image_path']}
            elif 'ftp_url' in data:
                username = data.get('username')
                password = data.get('password')
                img = face_service.load_image_from_ftp(data['ftp_url'], username, password)
                source_type = "ftp_url"
                source_info = {"url": data['ftp_url']}
            else:
                return None, None, None, None, jsonify({"error": "Missing 'image_path' or 'ftp_url' in JSON data"}), 400
        else:
            return None, None, None, None, jsonify({"error": "No image data provided. Use file upload or JSON with image_path/ftp_url"}), 400
        # Validate top parameter
        max_results = face_service.max_search_results
        if top < 1 or top > max_results:
            return None, None, None, None, jsonify({"error": f"Parameter 'top' must be between 1 and {max_results}"}), 400
        # Compute embedding and search
        embedding = face_service.compute_embedding(img)
        search_results = face_service.search_similar_faces(embedding, top)
        return embedding, img, source_type, source_info, search_results, top
    except FileNotFoundError as e:
        return None, None, None, None, jsonify({"error": str(e)}), 404
    except ValueError as e:
        return None, None, None, None, jsonify({"error": str(e)}), 400
    except Exception as e:
        return None, None, None, None, jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/search', methods=['POST'])
def search_similar():
    """
    Combined endpoint that returns search results, and optionally embedding if 'embedding' param is true
    """
    embedding_param = False
    # Check for embedding param in form or JSON
    if request.is_json:
        data = request.get_json()
        embedding_param = bool(data.get('embedding', False))
    elif 'embedding' in request.form:
        embedding_param = request.form.get('embedding', 'false').lower() == 'true'
    result = handle_embed_and_search(request)

    if isinstance(result[4], dict):
        # error response
        return result[4], result[5]
    embedding, img, source_type, source_info, search_results, top = result
    response = {
        "success": True,
        "source_type": source_type,
        "source_info": source_info,
        "top": top,
        "search_results": search_results
    }
    if embedding_param:
        response["embedding"] = embedding.tolist()
        response["embedding_shape"] = embedding.shape
    return jsonify(response)
    
if __name__ == '__main__':
    # For development only - use gunicorn for production
    import os
    debug_mode = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
