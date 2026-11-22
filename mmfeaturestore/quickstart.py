#!/usr/bin/env python
"""
End-to-end run of the feature store on a small data set (used by `make run`).

    python quickstart.py                                   # SYNTHETIC clips, generated on the fly
    python quickstart.py --data sample_data --per-genre 5  # GTZAN, after `make data`

Steps: write song metadata to PostgreSQL, extract audio features (28 numbers) and image features
(512 numbers, ResNet-18) with the same worker the API uses, store them in Milvus, then run one
similarity search per genre and print the nearest songs.

The default data is SYNTHETIC: tones and noise written by this script into output/synthetic/, with a
mel spectrogram image for each clip (GTZAN's images are mel spectrograms too). The GTZAN audio comes
from commercial recordings and is not part of this repository. To use it, run `make data` (needs a
Kaggle API token), then pass --data sample_data.

Needs PostgreSQL and Milvus running: docker compose up -d --wait
"""
import argparse
import asyncio
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.db.milvus import MilvusClient  # noqa: E402
from src.db.postgres import PostgresClient  # noqa: E402
from src.models.pipeline import FeatureStatus, Modality, Song  # noqa: E402
from src.worker.tasks import process_song_features  # noqa: E402

SYNTHETIC = ROOT / "output" / "synthetic"
SR = 22050
SECONDS = 5
# Three made-up "genres", each a different kind of signal. Not music.
SYNTHETIC_GENRES = ("synthetic_bass", "synthetic_treble", "synthetic_noise")


def make_synthetic_data(out: Path, per_genre: int) -> None:
    """Write per_genre WAV clips and mel spectrogram PNGs for each synthetic genre."""
    import librosa
    import soundfile as sf
    from PIL import Image

    rng = np.random.default_rng(0)
    t = np.arange(SR * SECONDS) / SR
    for genre in SYNTHETIC_GENRES:
        (out / "genres_original" / genre).mkdir(parents=True, exist_ok=True)
        (out / "images_original" / genre).mkdir(parents=True, exist_ok=True)
        for i in range(per_genre):
            if genre == "synthetic_bass":      # low tone with harmonics, slow pulse
                f0 = 110.0 * (1 + 0.03 * i)
                y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
                y *= 0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)
            elif genre == "synthetic_treble":  # high tones, fast tremolo
                f0 = 2000.0 * (1 + 0.03 * i)
                y = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * f0 * t)
                y *= 0.6 + 0.4 * np.sin(2 * np.pi * 8 * t)
            else:                               # noise bursts
                y = rng.standard_normal(t.size) * (np.sin(2 * np.pi * (1 + i) * t) > 0)
            y = 0.5 * y / np.max(np.abs(y)) + 0.01 * rng.standard_normal(t.size)
            sf.write(out / "genres_original" / genre / f"{genre}.{i:05d}.wav", y.astype(np.float32), SR)

            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128), ref=np.max)
            pixels = ((mel - mel.min()) / (mel.max() - mel.min() + 1e-9) * 255).astype(np.uint8)[::-1]
            Image.fromarray(pixels).convert("RGB").resize((432, 288)).save(
                out / "images_original" / genre / f"{genre}{i:05d}.png")


def find_songs(data: Path, per_genre: int):
    songs = []
    for genre_dir in sorted(p for p in (data / "genres_original").iterdir() if p.is_dir()):
        for clip in sorted(genre_dir.glob("*.wav"))[:per_genre]:
            number = clip.stem.split(".")[-1]
            image = data / "images_original" / genre_dir.name / f"{genre_dir.name}{number}.png"
            songs.append({"id": clip.stem[:36], "genre": genre_dir.name, "audio": clip,
                          "image": image if image.exists() else None})
    return songs


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="a folder with genres_original/ and images_original/ "
                        "(default: generate synthetic clips)")
    parser.add_argument("--per-genre", type=int, default=4, help="clips per genre (default 4)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, force=True)
    # src/db/milvus.py uses pymilvus's ORM API, which pymilvus 3.0 marks as deprecated
    warnings.filterwarnings("ignore", message=".*ORM-style PyMilvus API.*")

    if args.data is None:
        make_synthetic_data(SYNTHETIC, args.per_genre)
        data = SYNTHETIC
        print(f"Data: SYNTHETIC clips (not GTZAN) in {data.relative_to(ROOT)}/")
    else:
        data = args.data.resolve()
        if not (data / "genres_original").is_dir():
            sys.exit(f"No genres_original/ in {data}. For GTZAN, run `make data` (needs a Kaggle API token).")
        print(f"Data: {data}")

    songs = find_songs(data, args.per_genre)
    print(f"{len(songs)} clips in {len(set(s['genre'] for s in songs))} genres")

    postgres, milvus = PostgresClient(), MilvusClient()
    await postgres.connect()
    await postgres.create_tables()

    # Rerunning replaces each song's vectors instead of adding copies
    logging.getLogger("src.db.milvus").setLevel(logging.ERROR)   # "collection does not exist" on a first run
    milvus.connect()
    for modality in (Modality.AUDIO, Modality.IMAGE):
        for s in songs:
            milvus.delete_embedding(s["id"], modality)
    milvus.disconnect()   # the worker opens and closes its own Milvus connection
    logging.getLogger("src.db.milvus").setLevel(logging.NOTSET)

    failed = []
    for s in songs:
        await postgres.create_song(Song(id=s["id"], title=s["id"], genre=s["genre"]))
        await postgres.update_song_data_path(s["id"], Modality.AUDIO, str(s["audio"]))
        modalities = [Modality.AUDIO]
        if s["image"]:
            await postgres.update_song_data_path(s["id"], Modality.IMAGE, str(s["image"]))
            modalities.append(Modality.IMAGE)
        job = await process_song_features(s["id"], modalities)
        failed += [f"{s['id']}/{m}" for m, v in job.status.items() if v != FeatureStatus.PROCESSED.value]
    print(f"Stored features for {len(songs)} songs in PostgreSQL and Milvus"
          + (f"; failed: {', '.join(failed)}" if failed else ""))

    milvus.connect()
    genre_of = {s["id"]: s["genre"] for s in songs}
    queries = {}
    for s in songs:
        queries.setdefault(s["genre"], s["id"])
    found = 0
    for modality in (Modality.AUDIO, Modality.IMAGE):
        same = total = 0
        print(f"\nMost similar songs by {modality.value} (cosine similarity):")
        for genre, song_id in queries.items():
            vector = milvus.get_embedding(song_id, modality)
            if vector is None:
                continue
            hits = [h for h in milvus.search_similar(modality, list(vector), args.top_k + 1) if h[0] != song_id]
            hits = hits[:args.top_k]
            found += len(hits)
            print(f"  {song_id}")
            for hit_id, score in hits:
                print(f"    {score:.4f}  {hit_id}")
                same += genre_of.get(hit_id) == genre
                total += 1
        if total:
            print(f"  Same genre as the query: {same} of {total}")

    milvus.disconnect()
    await postgres.disconnect()
    if not found or failed:
        print("No search results" if not found else "Some features failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
