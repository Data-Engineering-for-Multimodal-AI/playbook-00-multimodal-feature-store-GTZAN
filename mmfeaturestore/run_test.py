#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
from PIL import Image
import torch
import librosa
from sentence_transformers import SentenceTransformer

# Setup Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.pipeline import Modality
from src.encoders.audio import AudioEncoder
from src.encoders.image import ImageEncoder
from src.encoders.text import TextEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test-runner")

def test_audio_encoder():
    """Test the audio encoder with sample file"""
    sample_path = "./sample_data/genres_original/rock/rock.00000.wav"
    if not os.path.exists(sample_path):
        logger.error(f"Sample file not found: {sample_path}")
        return False
    
    try:
        encoder = AudioEncoder()
        features = encoder.encode(sample_path)
        logger.info(f"Audio encoder test successful: shape={features.shape}")
        return True
    except Exception as e:
        logger.error(f"Audio encoder test failed: {e}")
        return False

def test_image_encoder():
    """Test the image encoder with sample file"""
    sample_path = "./sample_data/images_original/rock/rock00000.png"
    if not os.path.exists(sample_path):
        logger.error(f"Sample file not found: {sample_path}")
        return False
    
    try:
        encoder = ImageEncoder()
        features = encoder.encode(sample_path)
        logger.info(f"Image encoder test successful: shape={features.shape}")
        return True
    except Exception as e:
        logger.error(f"Image encoder test failed: {e}")
        return False

def test_text_encoder():
    """Test the text encoder with sample lyrics"""
    sample_lyrics = """
    I'm living on the edge of a lifetime
    Baby push me over
    Take me to the other side
    I've been dying to never know the ending
    Baby watch me go
    Watch me go
    """
    
    try:
        encoder = TextEncoder()
        features = encoder.encode(sample_lyrics)
        logger.info(f"Text encoder test successful: shape={features.shape}")
        return True
    except Exception as e:
        logger.error(f"Text encoder test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting tests with sample data...")
    
    # Run tests
    audio_test = test_audio_encoder()
    image_test = test_image_encoder()
    text_test = test_text_encoder()
    
    # Print results
    logger.info(f"Audio encoder test: {'PASSED' if audio_test else 'FAILED'}")
    logger.info(f"Image encoder test: {'PASSED' if image_test else 'FAILED'}")
    logger.info(f"Text encoder test: {'PASSED' if text_test else 'FAILED'}")
    
    if audio_test and image_test and text_test:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 