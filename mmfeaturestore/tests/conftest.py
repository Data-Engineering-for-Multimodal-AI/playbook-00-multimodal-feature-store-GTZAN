"""Tests use synthetic clips and images in a temporary data folder, their own Milvus collections and
song IDs starting with "pytest-" (removed afterwards), so they never touch GTZAN or its vectors."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

DATA = Path(tempfile.mkdtemp(prefix="featurestore-tests-"))
os.environ["FEATURESTORE_DATA"] = str(DATA)
os.environ["MILVUS_COLLECTION_PREFIX"] = "pytest"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GENRES = {"low": 220.0, "mid": 880.0, "high": 3520.0}


def make_data() -> None:
    rng = np.random.default_rng(0)
    sr = 22050
    t = np.arange(sr * 3) / sr
    for genre, freq in GENRES.items():
        (DATA / "genres_original" / genre).mkdir(parents=True, exist_ok=True)
        (DATA / "images_original" / genre).mkdir(parents=True, exist_ok=True)
        for i in range(3):
            tone = 0.5 * np.sin(2 * np.pi * freq * (1 + 0.01 * i) * t) + 0.02 * rng.standard_normal(t.size)
            sf.write(DATA / "genres_original" / genre / f"{genre}.0000{i}.wav", tone.astype(np.float32), sr)
            stripes = np.zeros((288, 432, 3), dtype=np.uint8)
            stripes[:, :, 0] = (np.sin(np.arange(432) * freq / 2000 + i) * 127 + 128).astype(np.uint8)
            Image.fromarray(stripes).save(DATA / "images_original" / genre / f"{genre}0000{i}.png")


make_data()
