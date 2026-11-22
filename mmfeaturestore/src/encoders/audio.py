"""Audio features: 28 numbers per clip (13 MFCC means, spectral centroid, 12 chroma means, zero-crossing
rate, spectral rolloff), standardized with the statistics in models/audio_scaler.json, then L2-normalized.

Without standardizing, the spectral centroid and rolloff (thousands of Hz) outweigh everything else in a
cosine similarity: every clip scores about 1.0 against every other one.
"""
import json
import logging
from pathlib import Path

import numpy as np
import librosa

# Mean and standard deviation of each feature over 999 GTZAN clips. Only the statistics, no audio.
SCALER = Path(__file__).resolve().parents[2] / "models" / "audio_scaler.json"

class AudioEncoder:
    """Encodes audio into feature vectors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mean = self.std = None
        if SCALER.exists():
            saved = json.loads(SCALER.read_text())
            self.mean, self.std = np.array(saved["mean"]), np.array(saved["std"])
        else:
            self.logger.warning(f"{SCALER} not found: audio features are not standardized")
    
    def encode(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file"""
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract features
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
        
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr, axis=1)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean, 
            spectral_centroid_mean, 
            chroma_mean, 
            zcr_mean, 
            rolloff_mean
        ])
        
        # Standardize each feature, then normalize to length 1
        if self.mean is not None:
            features = (features - self.mean) / self.std
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        self.logger.info(f"Generated audio features with shape: {features.shape}")
        return features 