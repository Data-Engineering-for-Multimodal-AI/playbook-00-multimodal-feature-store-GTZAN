#!/usr/bin/env python3
"""The songs most similar to one song, by audio and by spectrogram (python run.py must be running).

    python test_similarity.py                    # rock.00000
    python test_similarity.py --song classical.00003 --top 5
"""
import argparse
import sys

import requests

API = "http://127.0.0.1:8000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", default="rock.00000")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--api", default=API)
    args = parser.parse_args()
    try:
        requests.get(f"{args.api}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        sys.exit(f"The API isn't running at {args.api}: python run.py")
    for modality in ("audio", "image"):
        r = requests.post(f"{args.api}/songs/{args.song}/similar", json={"modality": modality, "top_k": args.top})
        if not r.ok:
            print(f"{modality}: {r.json().get('detail')}")
            return 1
        hits = r.json()["similar_songs"]
        same = sum(h["genre"] == args.song.split(".")[0] for h in hits)
        print(f"Most similar to {args.song} by {'spectrogram' if modality == 'image' else 'audio'} "
              f"({same} of {len(hits)} in the same genre):")
        for h in hits:
            print(f"  {h['song_id']:<16} {h['genre']:<10} similarity {h['similarity']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
