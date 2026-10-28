#!/usr/bin/env python3
"""
Load GTZAN into the feature store: one song per clip (genre as metadata), its audio and spectrogram
features in Milvus. The same encoders as the API, without HTTP, so 1,000 songs take minutes.

    python ingest_gtzan.py                   # all of sample_data/ (download_data.py first)
    python ingest_gtzan.py --per-genre 10    # a quick subset
    python ingest_gtzan.py --fit-scaler      # recompute models/audio_scaler.json from these clips first

Raw features are cached in output/features.npz for evaluate.py.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.db.milvus import MilvusClient  # noqa: E402
from src.db.postgres import PostgresClient  # noqa: E402
from src.encoders.audio import AudioEncoder, fit_scaler  # noqa: E402
from src.encoders.image import ImageEncoder  # noqa: E402
from src.models.pipeline import Modality, Song  # noqa: E402


def find_songs(per_genre=None):
    songs = []
    for genre_dir in sorted((config.DATA_ROOT / "genres_original").iterdir()):
        clips = sorted(genre_dir.glob("*.wav"))[:per_genre]
        for clip in clips:
            number = clip.stem.split(".")[1]
            image = config.DATA_ROOT / "images_original" / genre_dir.name / f"{genre_dir.name}{number}.png"
            songs.append({"id": clip.stem, "genre": genre_dir.name, "audio": clip,
                          "image": image if image.exists() else None})
    return songs


def rel(path: Path) -> str:
    return str(path.relative_to(config.ROOT))


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-genre", type=int, help="only the first N clips of each genre")
    parser.add_argument("--fit-scaler", action="store_true", help="recompute models/audio_scaler.json")
    parser.add_argument("--reset", action="store_true", help="drop the Milvus collections first")
    args = parser.parse_args(argv)
    if not (config.DATA_ROOT / "genres_original").exists():
        sys.exit(f"No GTZAN data in {config.DATA_ROOT}: python download_data.py")

    songs = find_songs(args.per_genre)
    print(f"{len(songs)} clips in {len(set(s['genre'] for s in songs))} genres "
          f"({sum(s['image'] is not None for s in songs)} with a spectrogram)")

    postgres, milvus = PostgresClient(), MilvusClient()
    await postgres.connect()
    milvus.connect()
    if args.reset:
        for modality in Modality:
            milvus.drop(modality)

    started = time.perf_counter()
    for s in songs:
        paths = {Modality.AUDIO.value: rel(s["audio"])}
        if s["image"]:
            paths[Modality.IMAGE.value] = rel(s["image"])
        await postgres.upsert_song(Song(id=s["id"], title=s["id"], genre=s["genre"], data_paths=paths))
    print(f"  PostgreSQL: {len(songs)} songs ({time.perf_counter() - started:.1f} s)")

    started = time.perf_counter()
    raw, audio_ids, failed = [], [], []
    for i, s in enumerate(songs, 1):
        try:
            raw.append(AudioEncoder.raw_features(str(s["audio"])))
            audio_ids.append(s["id"])
        except Exception as e:  # GTZAN's jazz.00054.wav can't be decoded
            failed.append(f"{s['id']} ({type(e).__name__})")
        if i % 100 == 0:
            print(f"    audio {i}/{len(songs)}", flush=True)
    raw = np.array(raw)
    audio_seconds = time.perf_counter() - started
    if args.fit_scaler:
        fit_scaler(raw, f"{len(raw)} GTZAN clips (download_data.py), ingest_gtzan.py --fit-scaler")
        print(f"  Fitted {AudioEncoder.__module__.split('.')[-1]} scaler -> models/audio_scaler.json")
    audio = AudioEncoder()
    vectors = np.array([audio.normalize(r) for r in raw])
    for start in range(0, len(audio_ids), 200):
        milvus.upsert(Modality.AUDIO, audio_ids[start:start + 200], vectors[start:start + 200].tolist())
    print(f"  Audio: {len(audio_ids)} clips, 28 features each ({audio_seconds:.0f} s, "
          f"{audio_seconds / max(1, len(audio_ids)):.2f} s per clip)" + (f"; unreadable: {', '.join(failed)}" if failed else ""))

    started = time.perf_counter()
    image = ImageEncoder()
    with_image = [s for s in songs if s["image"]]
    image_vectors = np.array([image.encode(str(s["image"])) for s in with_image])
    for start in range(0, len(with_image), 200):
        milvus.upsert(Modality.IMAGE, [s["id"] for s in with_image[start:start + 200]],
                      image_vectors[start:start + 200].tolist())
    image_seconds = time.perf_counter() - started
    print(f"  Spectrograms: {len(with_image)} images, 512 features each ({image_seconds:.0f} s, "
          f"{image_seconds / max(1, len(with_image)):.2f} s per image)")
    milvus.flush()
    for modality in (Modality.AUDIO, Modality.IMAGE):
        print(f"  Milvus {milvus.name(modality)}: {milvus.count(modality)} vectors")

    out = config.ROOT / "output"
    out.mkdir(exist_ok=True)
    np.savez(out / "features.npz", audio_ids=np.array(audio_ids), audio_raw=raw,
             image_ids=np.array([s["id"] for s in with_image]), image=image_vectors)
    await postgres.disconnect()
    milvus.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
