"""Smoke test without services: synthetic clips through both encoders. Needs no Docker and no GTZAN."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mmfeaturestore"))

import quickstart  # noqa: E402
from src.encoders.audio import AudioEncoder  # noqa: E402
from src.encoders.image import ImageEncoder  # noqa: E402


def test_encoders_on_synthetic_clips(tmp_path):
    quickstart.make_synthetic_data(tmp_path, per_genre=2)
    songs = quickstart.find_songs(tmp_path, per_genre=2)
    assert len(songs) == 6 and all(s["image"] for s in songs)

    audio = AudioEncoder().encode(str(songs[0]["audio"]))
    assert audio.shape == (28,) and abs(np.linalg.norm(audio) - 1) < 1e-5

    encoder = ImageEncoder()
    image = encoder.encode(str(songs[0]["image"]))
    assert image.shape == (512,) and abs(np.linalg.norm(image) - 1) < 1e-5


def test_image_encoder_uses_pretrained_weights():
    import torchvision.models as models
    pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).conv1.weight
    assert ImageEncoder().model[0].weight.detach().equal(pretrained.detach())


def test_audio_features_are_standardized_with_the_scaler(tmp_path):
    quickstart.make_synthetic_data(tmp_path, per_genre=2)
    clips = [str(s["audio"]) for s in quickstart.find_songs(tmp_path, per_genre=2)]
    encoder = AudioEncoder()
    assert encoder.mean is not None, "models/audio_scaler.json was not loaded"
    vectors = np.stack([encoder.encode(c) for c in clips])
    # Without the scaler, the centroid and rolloff (thousands of Hz) dominate and every pair scores ~1.0
    sims = vectors @ vectors.T
    assert sims[np.triu_indices(len(clips), 1)].min() < 0.9
