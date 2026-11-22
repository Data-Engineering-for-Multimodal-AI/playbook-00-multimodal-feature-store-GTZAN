"""Try this 1: what does standardizing the audio features do? (make try-scaler)

The audio encoder turns a clip into 28 numbers: 13 MFCC means, the spectral centroid and rolloff (in
Hz, thousands), 12 chroma means and the zero-crossing rate. Before normalizing the vector, it
standardizes each number with the mean and standard deviation in models/audio_scaler.json (measured
over 999 GTZAN clips). This encodes 24 synthetic clips (3 made-up genres) with and without that step
and searches each clip against the others. No Docker, no GTZAN.

Set PER_GENRE, or change which features are scaled in src/encoders/audio.py, and run it again.
"""

import sys

import numpy as np

from common import search_scores, synthetic, timed
from src.encoders.audio import AudioEncoder

PER_GENRE = 8


def main() -> int:
    songs = synthetic(PER_GENRE)
    genres = [s["genre"] for s in songs]
    encoder = AudioEncoder()
    print(f"{len(songs)} synthetic clips, {len(set(genres))} genres; each clip searched against the other {len(songs) - 1}\n")
    print(f"{'audio features':<16}  {'nearest is same genre':>21}  {'same genre':>10}  {'other genres':>12}  {'gap':>6}  {'pairs above 0.98':>16}")
    rows = [("standardized", encoder.mean, encoder.std), ("raw", None, None)]
    for name, mean, std in rows:
        encoder.mean, encoder.std = mean, std
        vectors, _ = timed(lambda s: encoder.encode(str(s["audio"])), songs)
        top1, same, other = search_scores(vectors, genres)
        sims = vectors @ vectors.T
        above = int((sims[np.triu_indices(len(songs), 1)] > 0.98).sum())
        pairs = len(songs) * (len(songs) - 1) // 2
        print(f"{name:<16}  {top1:>20.0%}  {same:>10.3f}  {other:>12.3f}  {same - other:>6.3f}  {above:>9} of {pairs}")
    print("\nsame genre / other genres: average cosine similarity between two clips. gap: the difference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
