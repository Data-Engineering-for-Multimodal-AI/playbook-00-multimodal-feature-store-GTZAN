#!/usr/bin/env python3
import os
import sys
import uuid
import logging
import asyncio
import numpy as np
import json
from typing import Dict, List, Optional
from datetime import datetime

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
logger = logging.getLogger("sample-test")

class FeatureStore:
    """Simple in-memory feature store for testing"""
    
    def __init__(self):
        self.features = {}
    
    def store_features(self, song_id: str, modality: Modality, features: np.ndarray) -> None:
        """Store features for a song"""
        key = f"{song_id}/{modality.value}"
        self.features[key] = features
        logger.info(f"Stored features for {key}: shape={features.shape}")
    
    def get_features(self, song_id: str, modality: Modality) -> np.ndarray:
        """Get features for a song"""
        key = f"{song_id}/{modality.value}"
        features = self.features.get(key)
        if features is not None:
            logger.info(f"Retrieved {modality.value} features: shape={features.shape}, norm={np.linalg.norm(features):.4f}")
        return features

def process_audio_sample(audio_file: str) -> np.ndarray:
    """Process an audio sample and return the features"""
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return None
    
    logger.info(f"Processing audio file: {audio_file}")
    encoder = AudioEncoder()
    features = encoder.encode(audio_file)
    
    logger.info(f"Generated audio features with shape: {features.shape}")
    return features

def process_image_sample(image_file: str) -> np.ndarray:
    """Process an image sample and return the features"""
    if not os.path.exists(image_file):
        logger.error(f"Image file not found: {image_file}")
        return None
    
    logger.info(f"Processing image file: {image_file}")
    encoder = ImageEncoder()
    features = encoder.encode(image_file)
    
    logger.info(f"Generated image features with shape: {features.shape}")
    return features

def process_lyrics_sample(text: str) -> np.ndarray:
    """Process a lyrics sample and return the features"""
    encoder = TextEncoder()
    features = encoder.encode(text)
    
    logger.info(f"Generated lyrics features with shape: {features.shape}")
    return features

def main():
    """Main function"""
    logger.info("Starting test with sample data...")
    
    # Sample rock song lyrics
    rock_lyrics = """
    I'm living on the edge of a lifetime
    Baby push me over
    Take me to the other side
    I've been dying to never know the ending
    Baby watch me go
    Watch me go
    
    I'm living on the edge of a lifetime
    Baby push me over
    Take me to the other side
    I've been dying to never know the ending
    Baby watch me go
    Watch me go
    
    It's getting closer now
    My time is running out
    But it's feeling so good
    It's feeling so good
    I'm living on the edge of a lifetime
    """
    
    # Sample file paths
    audio_file = "./genres_original/rock/rock.00000.wav"
    image_file = "./images_original/rock/rock00000.png"
    
    # Create feature store
    feature_store = FeatureStore()
    
    # Generate song ID
    song_id = "rock-sample-001"
    
    # Process audio
    audio_features = process_audio_sample(audio_file)
    if audio_features is not None:
        feature_store.store_features(song_id, Modality.AUDIO, audio_features)
    
    # Process image
    image_features = process_image_sample(image_file)
    if image_features is not None:
        feature_store.store_features(song_id, Modality.IMAGE, image_features)
    
    # Process lyrics
    lyrics_features = process_lyrics_sample(rock_lyrics)
    if lyrics_features is not None:
        feature_store.store_features(song_id, Modality.LYRICS, lyrics_features)
    
    # Retrieve features
    audio_features = feature_store.get_features(song_id, Modality.AUDIO)
    image_features = feature_store.get_features(song_id, Modality.IMAGE)
    lyrics_features = feature_store.get_features(song_id, Modality.LYRICS)
    
    # Check if all features were retrieved
    success = all(f is not None for f in [audio_features, image_features, lyrics_features])
    
    if success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 