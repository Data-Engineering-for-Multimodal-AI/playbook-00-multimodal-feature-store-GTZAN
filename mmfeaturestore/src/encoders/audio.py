import logging
import numpy as np
import librosa

class AudioEncoder:
    """Encodes audio into feature vectors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
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
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        self.logger.info(f"Generated audio features with shape: {features.shape}")
        return features 