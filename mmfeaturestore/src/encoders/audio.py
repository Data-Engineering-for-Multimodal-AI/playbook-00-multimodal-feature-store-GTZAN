"""
Audio features: 28 numbers per clip (13 MFCC means, spectral centroid, 12 chroma means, zero-crossing
rate, spectral rolloff), standardized with the statistics in models/audio_scaler.json, then L2-normalized.

Without standardizing, the centroid and rolloff (thousands of Hz) outweigh everything else in a cosine
similarity, and the vector mostly measures brightness.
"""
import json
import logging
from typing import Optional

import librosa
import numpy as np

from .. import config

FEATURE_NAMES = ([f"mfcc_{i}" for i in range(1, 14)] + ["spectral_centroid"] + [f"chroma_{i}" for i in range(1, 13)]
                 + ["zero_crossing_rate", "spectral_rolloff"])
SCALER = config.MODELS / "audio_scaler.json"


def zero_crossing_rate(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    """librosa's zero-crossing rate, in NumPy: librosa's version (numba) can crash the process on macOS
    when PyTorch is loaded in it too, as it is in the worker."""
    frames = librosa.util.frame(np.pad(y, frame_length // 2, mode="edge"), frame_length=frame_length,
                                hop_length=hop_length)
    return float(np.mean(np.abs(np.diff(np.signbit(frames), axis=0)), axis=0).mean())


class AudioEncoder:
    def __init__(self, standardize: bool = True):
        self.logger = logging.getLogger(__name__)
        self.mean = self.std = None
        if standardize:
            if not SCALER.exists():
                raise FileNotFoundError(f"{SCALER} is missing: run python ingest_gtzan.py --fit-scaler")
            saved = json.loads(SCALER.read_text())
            self.mean, self.std = np.array(saved["mean"]), np.array(saved["std"])

    @staticmethod
    def raw_features(audio_path: str) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=22050)
        return np.concatenate([
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),
            librosa.feature.spectral_centroid(y=y, sr=sr).mean(axis=1),
            librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
            [zero_crossing_rate(y)],
            librosa.feature.spectral_rolloff(y=y, sr=sr).mean(axis=1),
        ])

    def normalize(self, features: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            features = (features - self.mean) / self.std
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

    def encode(self, audio_path: str) -> np.ndarray:
        return self.normalize(self.raw_features(audio_path))


def fit_scaler(features: np.ndarray, source: str, path: Optional[str] = None) -> None:
    std = features.std(axis=0)
    with open(path or SCALER, "w") as f:
        json.dump({"source": source, "features": FEATURE_NAMES, "count": len(features),
                   "mean": features.mean(axis=0).round(8).tolist(), "std": np.where(std > 0, std, 1).round(8).tolist()},
                  f, indent=1)
