# Production Configuration for Face Embedding API

import os

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Model Configuration
    MODEL_SYMBOL_PATH = os.environ.get('MODEL_SYMBOL_PATH', '/app/models/face_encoder_symbol.json')
    MODEL_PARAMS_PATH = os.environ.get('MODEL_PARAMS_PATH', '/app/models/face_encoder.params')

    # Qdrant Configuration
    QDRANT_URL = os.environ.get('QDRANT_URL', 'http://qdrant:6333/collections/f4r/points/search')
    
    # MXNet Configuration
    USE_GPU = os.environ.get('USE_GPU', 'false').lower() == 'true'
    GPU_ID = int(os.environ.get('GPU_ID', '0'))
    
    # API Configuration
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))
    MAX_SEARCH_RESULTS = int(os.environ.get('MAX_SEARCH_RESULTS', '100'))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'INFO'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
