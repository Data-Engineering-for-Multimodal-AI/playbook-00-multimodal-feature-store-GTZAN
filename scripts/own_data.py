#!/usr/bin/env python3
"""
Use your own music: check a folder in the GTZAN layout, then store its features and search it.

    python scripts/own_data.py --dir my_music --check-only   # check every file, store nothing
    python scripts/own_data.py --dir my_music                # check, then quickstart.py --data my_music

Layout (the same as GTZAN's):

    my_music/
      genres_original/<genre>/<name>.<number>.wav     audio, any sample rate (read at 22,050 Hz)
      images_original/<genre>/<genre><number>.png     optional: an image per clip (GTZAN: its spectrogram)

The second command needs PostgreSQL and Milvus running (docker compose up -d --wait).
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
MIN_SECONDS = 1.0
MIN_SIDE = 32
ID_LENGTH = 36    # Milvus and PostgreSQL keep the first 36 characters of the file name as the song ID


def _lfs(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(100).startswith(b"version https://git-lfs")


def _check_audio(path: Path, rel: str, problems: List[str]) -> bool:
    if path.stat().st_size == 0:
        problems.append(f"{rel}: empty file")
        return False
    if _lfs(path):
        problems.append(f"{rel}: a Git LFS pointer, not the audio (run git lfs pull)")
        return False
    try:
        info = sf.info(str(path))
        y, _ = sf.read(str(path), frames=int(info.samplerate * 30), dtype="float32")
    except Exception as e:  # noqa: BLE001 - any reader error means the encoder can't read it either
        problems.append(f"{rel}: can't read as WAV audio ({type(e).__name__})")
        return False
    if info.duration < MIN_SECONDS:
        problems.append(f"{rel}: {info.duration:.2f} s long, needs at least {MIN_SECONDS:.0f} s")
        return False
    if not np.any(np.abs(y) > 1e-6):
        problems.append(f"{rel}: silent (every sample is 0)")
        return False
    return True


def _check_image(path: Path, rel: str, problems: List[str]) -> None:
    if _lfs(path):
        problems.append(f"{rel}: a Git LFS pointer, not the image (run git lfs pull)")
        return
    try:
        with Image.open(path) as im:
            im.load()
            size = im.size
    except (UnidentifiedImageError, OSError) as e:
        problems.append(f"{rel}: can't open as an image ({type(e).__name__})")
        return
    if min(size) < MIN_SIDE:
        problems.append(f"{rel}: too small ({size[0]}x{size[1]}), needs at least {MIN_SIDE}x{MIN_SIDE}")


def scan(folder: Path) -> Tuple[Dict[str, int], List[str], List[str]]:
    """Check the folder. Returns (clips and images per genre, problems, notes)."""
    folder = Path(folder)
    problems: List[str] = []
    notes: List[str] = []
    counts = {"clips": 0, "images": 0, "genres": 0}
    audio_root = folder / "genres_original"
    if not folder.is_dir():
        return counts, [f"{folder}: not a folder"], notes
    if not audio_root.is_dir():
        return counts, [f"{folder}: no genres_original/ folder"], notes
    genres = sorted(p for p in audio_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not genres:
        problems.append("genres_original/: no genre folders")
    ids: Dict[str, str] = {}
    for genre in genres:
        files = sorted(p for p in genre.iterdir() if p.is_file() and not p.name.startswith("."))
        clips = [p for p in files if p.suffix == ".wav"]
        for other in files:
            if other.suffix != ".wav":
                problems.append(f"genres_original/{genre.name}/{other.name}: not a .wav file "
                                "(only .wav is read; convert it first)")
        if not clips and not [f for f in files if f.suffix != ".wav"]:
            problems.append(f"genres_original/{genre.name}/: no .wav files")
        counts["genres"] += bool(clips)
        for clip in clips:
            rel = f"genres_original/{genre.name}/{clip.name}"
            if not _check_audio(clip, rel, problems):
                continue
            counts["clips"] += 1
            song_id = clip.stem[:ID_LENGTH]
            if song_id in ids:
                problems.append(f"{rel}: same song ID {song_id!r} as {ids[song_id]} "
                                f"(IDs are the first {ID_LENGTH} characters of the name)")
            else:
                ids[song_id] = rel
            if len(clip.stem) > ID_LENGTH:
                notes.append(f"{rel}: name longer than {ID_LENGTH} characters; the song ID is {song_id!r}")
            number = clip.stem.split(".")[-1]
            image = folder / "images_original" / genre.name / f"{genre.name}{number}.png"
            if image.exists():
                counts["images"] += 1
                _check_image(image, f"images_original/{genre.name}/{image.name}", problems)
            else:
                notes.append(f"{rel}: no image (looked for images_original/{genre.name}/{image.name}); audio only")
    return counts, problems, notes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path, help="a folder with genres_original/ (and images_original/)")
    parser.add_argument("--check-only", action="store_true", help="check the files and store nothing")
    args = parser.parse_args(argv)
    counts, problems, notes = scan(args.dir)
    if problems:
        print(f"{len(problems)} problem{'s' if len(problems) > 1 else ''} in {args.dir}; nothing was stored:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"{args.dir}: OK, {counts['clips']} clips in {counts['genres']} genres, {counts['images']} with an image")
    for n in notes[:10]:
        print(f"  note: {n}")
    if len(notes) > 10:
        print(f"  ... and {len(notes) - 10} more notes")
    if args.check_only:
        return 0
    sys.path.insert(0, str(ROOT / "mmfeaturestore"))
    import quickstart
    return asyncio.run(quickstart.main(["--data", str(args.dir), "--per-genre", str(counts["clips"])]))


if __name__ == "__main__":
    sys.exit(main())
