import logging
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEncoder:
    """Encodes lyrics text into feature vectors"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text: str) -> np.ndarray:
        """Extract features from lyrics text"""
        self.logger.info(f"Processing lyrics text of length {len(text)}")
        
        # Generate embedding
        features = self.model.encode(text)
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        self.logger.info(f"Generated lyrics features with shape: {features.shape}")
        return features 