#!/usr/bin/env python3
"""
Find the stored songs that sound most like an audio file, or look most like an image.

    python scripts/similar.py path/to/clip.wav
    python scripts/similar.py path/to/image.png --top-k 5

Encodes the file with the same encoder the worker uses and searches the Milvus collection for that
modality (song_audio or song_image). Needs the services running and songs stored (make run).
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mmfeaturestore"))
warnings.filterwarnings("ignore")

from src.db.milvus import MilvusClient  # noqa: E402
from src.models.pipeline import Modality  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="a .wav file (audio search) or an image (image search)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    logging.disable(logging.INFO)
    if not args.file.is_file():
        print(f"{args.file}: no such file")
        return 1
    if args.file.suffix.lower() == ".wav":
        from src.encoders.audio import AudioEncoder
        modality, vector = Modality.AUDIO, AudioEncoder().encode(str(args.file))
    else:
        from src.encoders.image import ImageEncoder
        modality, vector = Modality.IMAGE, ImageEncoder().encode(str(args.file))
    milvus = MilvusClient()
    milvus.connect()
    try:
        hits = milvus.search_similar(modality, list(vector), args.top_k)
    finally:
        milvus.disconnect()
    print(f"Most similar songs by {modality.value} to {args.file} (cosine similarity):")
    for song_id, score in hits:
        print(f"  {score:.4f}  {song_id}")
    if not hits:
        print("  none: store some songs first (make run)")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
