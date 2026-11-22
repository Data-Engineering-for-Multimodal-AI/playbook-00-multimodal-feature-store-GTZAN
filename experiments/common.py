"""Shared helpers for the Try this experiments: synthetic clips from quickstart.py, the repo's
encoders, and a leave-one-out search (cosine similarity, the metric the Milvus collections use).
No Docker, no GTZAN."""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mmfeaturestore"))
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
import quickstart  # noqa: E402

OUT = ROOT / "mmfeaturestore" / "output" / "experiments"


def synthetic(per_genre: int = 8, seconds: int = None, folder: str = "clips"):
    """Write per_genre synthetic clips per genre (and their spectrogram images); return the songs."""
    target = OUT / folder
    if seconds:
        quickstart.SECONDS = seconds
    quickstart.make_synthetic_data(target, per_genre)
    return quickstart.find_songs(target, per_genre)


def search_scores(vectors: np.ndarray, genres):
    """Leave-one-out: for each clip, is its most similar other clip the same genre? Also the mean
    similarity to clips of the same genre and of other genres."""
    s = vectors @ vectors.T
    n = len(genres)
    np.fill_diagonal(s, -np.inf)
    top1 = np.mean([genres[int(np.argmax(s[i]))] == genres[i] for i in range(n)])
    same = np.mean([s[i, j] for i in range(n) for j in range(n) if i != j and genres[i] == genres[j]])
    other = np.mean([s[i, j] for i in range(n) for j in range(n) if genres[i] != genres[j]])
    return float(top1), float(same), float(other)


def timed(fn, items):
    started = time.perf_counter()
    out = np.stack([fn(x) for x in items])
    return out, (time.perf_counter() - started) / len(items)
