#!/usr/bin/env python3
"""Add one song through the API and extract its features (python run.py must be running).

    python test_new_song.py                       # rock.00000
    python test_new_song.py --song jazz.00012
"""
import argparse
import sys
import time

import requests

API = "http://127.0.0.1:8000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", default="rock.00000", help="a GTZAN clip, e.g. rock.00000")
    parser.add_argument("--api", default=API)
    args = parser.parse_args()
    genre, number = args.song.split(".")
    try:
        requests.get(f"{args.api}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        sys.exit(f"The API isn't running at {args.api}: python run.py")

    song = requests.post(f"{args.api}/songs", json={"id": args.song, "title": args.song, "genre": genre}).json()
    print(f"Created song {song['id']} ({song['genre']})")
    for modality, path in (("audio", f"sample_data/genres_original/{genre}/{args.song}.wav"),
                           ("image", f"sample_data/images_original/{genre}/{genre}{number}.png")):
        r = requests.put(f"{args.api}/songs/{args.song}/data/{modality}", json={"path": path})
        print(f"  {modality} path: {path}" if r.ok else f"  {modality}: {r.json()['detail']}")
    job = requests.post(f"{args.api}/songs/{args.song}/process", json={"modalities": ["audio", "image"]}).json()
    print(f"Processing job {job['job_id']}")
    for _ in range(120):
        status = requests.get(f"{args.api}/jobs/{job['job_id']}").json()
        if all(s in ("processed", "failed") for s in status["status"].values()):
            break
        time.sleep(0.5)
    print(f"  status: {status['status']}" + (f" ({status['error_message']})" if status.get("error_message") else ""))
    for modality in ("audio", "image"):
        r = requests.get(f"{args.api}/songs/{args.song}/features/{modality}")
        if r.ok:
            v = r.json()["embedding"]
            print(f"  {modality} features: {len(v)} numbers, first three {[round(x, 3) for x in v[:3]]}")
    return 0 if all(s == "processed" for s in status["status"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
