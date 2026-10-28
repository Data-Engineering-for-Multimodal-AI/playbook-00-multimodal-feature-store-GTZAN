#!/usr/bin/env python3
"""
How useful are the stored features? For every song, ask Milvus for its 10 most similar songs (not
itself) and count how many share its genre (precision@10), and whether the majority genre of those 10
is right (10-NN genre accuracy). Chance is about 0.10: 10 genres of about 100 songs.

    python evaluate.py                  # audio, spectrogram, and both combined
    python evaluate.py --first-edition  # also the first edition's encoders, for comparison (about 1 minute)

The combined score averages a song's cosine similarity under both features (computed from the vectors
stored in Milvus). Run ingest_gtzan.py first.
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.db.milvus import MilvusClient  # noqa: E402
from src.models.pipeline import Modality  # noqa: E402


def genre(song_id: str) -> str:
    return song_id.split(".")[0]


def score(neighbours: dict) -> dict:
    """neighbours: song -> its 10 nearest songs, best first."""
    precision, correct, per_genre = [], [], {}
    for song, near in neighbours.items():
        same = [genre(n) == genre(song) for n in near]
        precision.append(np.mean(same))
        votes = Counter(genre(n) for n in near).most_common()
        correct.append(votes[0][0] == genre(song))
        per_genre.setdefault(genre(song), []).append(np.mean(same))
    return {"songs": len(neighbours), "precision_at_10": round(float(np.mean(precision)), 3),
            "knn_genre_accuracy": round(float(np.mean(correct)), 3),
            "per_genre_precision": {g: round(float(np.mean(v)), 2) for g, v in sorted(per_genre.items())}}


def in_memory(ids, vectors, k=10) -> dict:
    x = np.asarray(vectors, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)[:, :k]
    return {ids[i]: [ids[j] for j in row] for i, row in enumerate(order)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-edition", action="store_true", help="also score the first edition's encoders")
    args = parser.parse_args(argv)
    milvus = MilvusClient()
    milvus.connect()
    results, latencies = {}, []
    stored = {}
    for modality in (Modality.AUDIO, Modality.IMAGE):
        stored[modality] = milvus.all_embeddings(modality)
        if not stored[modality]:
            sys.exit("No features in Milvus yet: python ingest_gtzan.py")
        neighbours = {}
        for song, vector in stored[modality].items():
            started = time.perf_counter()
            hits = milvus.search_similar(modality, vector, 10, exclude=song)
            latencies.append((time.perf_counter() - started) * 1000)
            neighbours[song] = [h[0] for h in hits]
        results[f"{'spectrogram' if modality == Modality.IMAGE else 'audio'} (Milvus search)"] = score(neighbours)
    milvus.disconnect()

    both = sorted(set(stored[Modality.AUDIO]) & set(stored[Modality.IMAGE]))
    fused = [np.concatenate([stored[Modality.AUDIO][s], stored[Modality.IMAGE][s]]) / np.sqrt(2) for s in both]
    results["audio + spectrogram (combined)"] = score(in_memory(both, fused))

    if args.first_edition:
        cache = np.load(config.ROOT / "output" / "features.npz")
        ids = list(cache["audio_ids"])
        results["first edition: audio without standardizing"] = score(in_memory(ids, cache["audio_raw"]))
        from src.encoders.image import ImageEncoder
        import torch
        torch.manual_seed(0)
        untrained = ImageEncoder(pretrained=False)
        image_ids = list(cache["image_ids"])
        paths = {p.stem: p for p in (config.DATA_ROOT / "images_original").glob("*/*.png")}
        vectors = [untrained.encode(str(paths[i.replace(".", "")])) for i in image_ids]
        results["first edition: spectrogram, untrained ResNet-18"] = score(in_memory(image_ids, vectors))

    report = {"results": results, "chance_precision_at_10": round(99 / 998, 3),
              "milvus_search_ms_median": round(statistics.median(latencies), 2),
              "milvus_search_ms_p95": round(float(np.percentile(latencies, 95)), 2), "searches": len(latencies)}
    (config.ROOT / "output").mkdir(exist_ok=True)
    (config.ROOT / "output" / "evaluation.json").write_text(json.dumps(report, indent=2))

    print(f"{'Features':<48}{'Songs':>6}{'P@10':>7}{'10-NN acc':>11}")
    for name, r in results.items():
        print(f"{name:<48}{r['songs']:>6}{r['precision_at_10']:>7.3f}{r['knn_genre_accuracy']:>11.3f}")
    print(f"{'chance':<48}{'':>6}{report['chance_precision_at_10']:>7.3f}{0.1:>11.3f}")
    print(f"\nMilvus search: median {report['milvus_search_ms_median']} ms, p95 {report['milvus_search_ms_p95']} ms "
          f"({report['searches']} searches, top 10, HNSW)")
    best = results["audio + spectrogram (combined)"]["per_genre_precision"]
    print("Precision@10 per genre (combined): " + ", ".join(f"{g} {p:.2f}" for g, p in
                                                         sorted(best.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
