import logging

from ..models.pipeline import Modality


class EncoderFactory:
    """One encoder per modality, created on first use (the lyrics model loads only if lyrics are processed)."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encoders = {}

    def create_encoder(self, modality: Modality):
        if modality not in self.encoders:
            if modality == Modality.AUDIO:
                from .audio import AudioEncoder
                self.encoders[modality] = AudioEncoder()
            elif modality == Modality.IMAGE:
                from .image import ImageEncoder
                self.encoders[modality] = ImageEncoder()
            elif modality == Modality.LYRICS:
                from .text import TextEncoder
                self.encoders[modality] = TextEncoder()
            else:
                raise ValueError(f"Unsupported modality: {modality}")
        return self.encoders[modality]
