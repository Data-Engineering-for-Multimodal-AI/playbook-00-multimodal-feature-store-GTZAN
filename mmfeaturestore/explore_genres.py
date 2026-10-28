#!/usr/bin/env python3
"""Songs per genre in the store, and for one song of each genre, the genres of its 5 nearest songs
by audio (python run.py must be running)."""
import sys
from collections import Counter

import requests

API = "http://127.0.0.1:8000"


def main() -> int:
    try:
        counts = requests.get(f"{API}/genres", timeout=5).json()
    except requests.RequestException:
        sys.exit(f"The API isn't running at {API}: python run.py")
    print("Songs per genre: " + ", ".join(f"{g} {n}" for g, n in counts.items()))
    print("\nNearest 5 songs by audio, for the first clip of each genre:")
    for genre in counts:
        song = f"{genre}.00000"
        r = requests.post(f"{API}/songs/{song}/similar", json={"modality": "audio", "top_k": 5})
        if r.ok:
            genres = Counter(h["genre"] for h in r.json()["similar_songs"])
            print(f"  {song:<16} -> " + ", ".join(f"{g} {n}" for g, n in genres.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
