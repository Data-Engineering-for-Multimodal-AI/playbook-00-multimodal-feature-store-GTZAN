import logging
from ..models.pipeline import Modality
from .audio import AudioEncoder
from .image import ImageEncoder
from .text import TextEncoder

class EncoderFactory:
    """Factory for creating encoders based on modality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encoders = {}
    
    def create_encoder(self, modality: Modality):
        """Create an encoder for a specific modality"""
        if modality not in self.encoders:
            if modality == Modality.LYRICS:
                self.logger.info(f"Creating encoder for {modality.value}")
                self.encoders[modality] = TextEncoder()
            elif modality == Modality.AUDIO:
                self.logger.info(f"Creating encoder for {modality.value}")
                self.encoders[modality] = AudioEncoder()
            elif modality == Modality.IMAGE:
                self.logger.info(f"Creating encoder for {modality.value}")
                self.encoders[modality] = ImageEncoder()
            else:
                raise ValueError(f"Unsupported modality: {modality}")
        
        return self.encoders[modality] 