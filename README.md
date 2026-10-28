# Multimodal Feature Store for Music (GTZAN)

A companion project for *Data Engineering for Multimodal AI* (O'Reilly, 2026). It isn't printed in the book; it is the audio counterpart of the [Chapter 2 playbook](https://github.com/Data-Engineering-for-Multimodal-AI/playbook-02-multimodal-feature-store-streambuy)'s feature store.
Book: https://oreil.ly/data-engineering-multimodal

> The version before the 2026 update is tagged `first-edition`; see [Differences from the first edition](#differences-from-the-first-edition).

## What you'll build

A feature store for music: song metadata in PostgreSQL, one vector per song and modality in Milvus, and a REST API to add songs, extract their features in the background, and find similar songs.

- **Audio**: 28 numbers per 30-second clip (MFCCs, chroma, spectral centroid and rolloff, zero-crossing rate), standardized, so no single feature dominates the similarity.
- **Spectrogram**: the clip's mel spectrogram, as an image, through ResNet-18 pretrained on ImageNet (512 numbers).
- **Lyrics** (optional): sentence embeddings, for your own songs; GTZAN has no lyrics.
- **Data**: [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), 1,000 clips in 10 genres with their spectrograms, downloaded from Kaggle.
- **A measured evaluation**: for every song, are its 10 nearest songs in Milvus from the same genre?

What it shows, measured below:

- **Each modality on its own finds the genre about 4.4 times as often as chance**, and combining them does better.
- **Preprocessing decides quality.** The first edition's features did much worse: audio not standardized (precision 0.327) and spectrograms through a network with random weights (0.352). Fixing both: 0.441 and 0.444.
- **Genres differ.** Classical and metal are easy (0.83 and 0.76); rock, disco and country are mixed up with each other (about 0.25).

## Quick start

You need Python 3.12 or newer, Docker, and a Kaggle account for the data (an API token in `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`, see `.env.example`).

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-00-multimodal-feature-store-GTZAN.git
cd playbook-00-multimodal-feature-store-GTZAN

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

cd mmfeaturestore
docker compose up -d --wait            # PostgreSQL (127.0.0.1:5433) and Milvus 3.0.3 (127.0.0.1:19530)
python download_data.py                # GTZAN from Kaggle, 1.2 GB, into sample_data/
python ingest_gtzan.py                 # all songs and their features (about 3 minutes)
python evaluate.py --first-edition     # how well the features find the genre
```

With [uv](https://docs.astral.sh/uv/): `cd mmfeaturestore && uv sync`, then prefix commands with `uv run`. From the repository root, the Makefile has `make services`, `make data`, `make ingest`, `make evaluate`, `make api` and `make smoke`. `docker compose down` stops the services (`down -v` also deletes their data).

### The API

```bash
python run.py                          # http://127.0.0.1:8000/docs
python test_new_song.py                # add rock.00000 through the API and extract its features
python test_similarity.py --song classical.00003
python explore_genres.py
./test_api_curl.sh                     # the same with curl
```

| Endpoint | What it does |
|---|---|
| `POST /songs` | create or update a song (`id`, `title`, `artist`, `genre`, ...) |
| `PUT /songs/{id}/data/{modality}` | the file for `audio`, `image` or `lyrics`; it must be inside `sample_data/` (or `FEATURESTORE_DATA`) |
| `POST /songs/{id}/process` | extract the features in the background; returns a job |
| `GET /jobs/{job_id}` | the job's status per modality (kept in PostgreSQL) |
| `GET /songs/{id}/features/{modality}` | the stored vector |
| `POST /songs/{id}/similar` | `{"modality": "audio", "top_k": 10}`: the most similar songs, not counting itself |
| `GET /genres` | songs per genre |

## Tested on

| Date | OS | Python | What |
|---|---|---|---|
| 2026-10-27 | macOS 26.5.1 (Apple M3, 8 GB) | 3.13.16 | `make smoke` (4 tests); GTZAN downloaded, ingested and evaluated; the API and its scripts (output below) |
| 2026-10-27 | macOS 26.5.1 (Apple M3) | 3.12.15 | Fresh clone: `pip install -r requirements.txt` (77 s), the services, the tests |
| 2026-10-27 | Ubuntu (GitHub Actions) | 3.12, 3.13 | The tests with PostgreSQL and Milvus in Docker; `pip install -r requirements.txt` and the API. CI uses synthetic clips: GTZAN can't be redistributed |

Versions are pinned in `mmfeaturestore/uv.lock` and `mmfeaturestore/requirements.txt` (pymilvus 3.0, Milvus 3.0.3, PostgreSQL 17, librosa 0.11, torch 2.14 (CPU build on Linux), FastAPI).

## Differences from the first edition

| First edition | Now | Why |
|---|---|---|
| Image features from ResNet-18 with random weights (`weights=None`) | ImageNet-pretrained weights | Random weights encode noise; measured: precision 0.352 → 0.444 |
| Audio features L2-normalized as they were | Standardized first (`models/audio_scaler.json`), then normalized | Spectral centroid and rolloff (thousands of Hz) outweighed the other 26 features; 0.327 → 0.442 |
| The README called the images album artwork | They are GTZAN's mel spectrograms (Kaggle version) | Accuracy |
| The README said sample audio and images were in the repository; its download link (marsyas.info) no longer answers | `download_data.py` downloads GTZAN from Kaggle and checks every file | Nothing was committed; the audio can't be redistributed |
| 102 MB of database files from one local run committed (`mmfeaturestore/volumes/`) | Docker volumes | Generated state, not source; the pipeline rebuilds it |
| `docker-compose.yml`: Milvus 2.3.3, MinIO's own image, no PostgreSQL; ports on every interface | PostgreSQL 17, Milvus 3.0.3 (Milvus's own compose setup), on 127.0.0.1 | The API needs PostgreSQL; MinIO stopped publishing its image |
| pymilvus ORM API (`connections`, `Collection`) | pymilvus `MilvusClient` | The ORM API is deprecated |
| Each job opened its own database connections; `datetime.now()` fixed at import | Shared connections; timestamps per object | |
| Searches returned the song itself as its own best match | Left out | |
| `librosa`'s zero-crossing rate | The same computation in NumPy | librosa's numba code crashed the process on macOS with PyTorch loaded, as in the worker |
| Scripts with absolute paths from one machine; `batch_process.py`, `run_test.py` and six manual test scripts | `ingest_gtzan.py`, `evaluate.py`, pytest; the README's scripts (`test_new_song.py`, `test_similarity.py`, `explore_genres.py`, `test_api_curl.sh`) work | |
| A `data` submodule link with no `.gitmodules`, an empty `new_data/`, a second `docker-compose.yml` | Removed | |
| API on 0.0.0.0; any file path accepted | 127.0.0.1; paths inside the data folder only | |
| Python 3.8+, unpinned; clone URL `yourusername/mmfeaturestore` | Python 3.12+, pinned; this repository | |

Full list: [CHANGELOG.md](CHANGELOG.md).

## Data and licenses

| Data | Source | License |
|---|---|---|
| GTZAN clips (`sample_data/genres_original/`, downloaded, not in the repository): 1,000 × 30 s, 10 genres | G. Tzanetakis and P. Cook, "Musical genre classification of audio signals", IEEE Transactions on Speech and Audio Processing, 2002. Kaggle mirror: [andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Kaggle lists "Other". The clips come from commercial recordings; GTZAN is used for research and isn't redistributed here. `jazz.00054.wav` is unreadable |
| Spectrograms (`sample_data/images_original/`, downloaded): 999 mel spectrograms | The same Kaggle dataset | As above |
| `models/audio_scaler.json`: the 28 features' means and standard deviations over the 999 readable clips | Computed by `ingest_gtzan.py --fit-scaler` | Statistics only; no audio |
| ResNet-18 ImageNet weights (downloaded on first use) | torchvision | BSD-3-Clause (torchvision); trained on ImageNet |

Known issues with GTZAN (B. Sturm, 2013): repeated clips, mislabellings and several clips from the same artist in a genre. They make the genre scores here somewhat optimistic.

## Expected output

On 2026-10-27, macOS 26.5.1 (Apple M3, 8 GB), Python 3.13.16, with the services in Docker. This is the output of this version.

`python ingest_gtzan.py --reset`:

```
1000 clips in 10 genres (999 with a spectrogram)
  PostgreSQL: 1000 songs (1.3 s)
  Audio: 999 clips, 28 features each (128 s, 0.13 s per clip); unreadable: jazz.00054 (LibsndfileError)
  Spectrograms: 999 images, 512 features each (31 s, 0.03 s per image)
  Milvus song_audio: 999 vectors
  Milvus song_image: 999 vectors
```

`python evaluate.py --first-edition`. P@10: the share of a song's 10 nearest songs in its genre; 10-NN acc: the majority genre of those 10 is right.

```
Features                                         Songs   P@10  10-NN acc
audio (Milvus search)                              999  0.441      0.562
spectrogram (Milvus search)                        999  0.444      0.609
audio + spectrogram (combined)                     999  0.483      0.603
first edition: audio without standardizing         999  0.327      0.425
first edition: spectrogram, untrained ResNet-18    999  0.352      0.506
chance                                                  0.099      0.100

Milvus search: median 3.19 ms, p95 7.22 ms (1998 searches, top 10, HNSW)
Precision@10 per genre (combined): classical 0.83, metal 0.76, pop 0.67, jazz 0.53, blues 0.50, hiphop 0.44, reggae 0.32, country 0.28, disco 0.26, rock 0.25
```

`python test_new_song.py` (with `python run.py` running):

```
Created song rock.00000 (rock)
  audio path: sample_data/genres_original/rock/rock.00000.wav
  image path: sample_data/images_original/rock/rock00000.png
Processing job cab5efc6-21c7-4e38-9afd-452f0f7c08e8
  status: {'audio': 'processed', 'image': 'processed'}
  audio features: 28 numbers, first three [0.079, 0.088, -0.186]
  image features: 512 numbers, first three [0.079, 0.001, 0.051]
```

`python test_similarity.py --song classical.00003 --top 5`:

```
Most similar to classical.00003 by audio (4 of 5 in the same genre):
  classical.00029  classical  similarity 0.855
  classical.00013  classical  similarity 0.846
  classical.00010  classical  similarity 0.834
  classical.00073  classical  similarity 0.820
  jazz.00007       jazz       similarity 0.810
Most similar to classical.00003 by spectrogram (5 of 5 in the same genre):
  classical.00026  classical  similarity 0.947
  classical.00075  classical  similarity 0.946
  classical.00009  classical  similarity 0.945
  classical.00023  classical  similarity 0.943
  classical.00013  classical  similarity 0.941
```

`python explore_genres.py`:

```
Songs per genre: blues 100, classical 100, country 100, disco 100, hiphop 100, jazz 100, metal 100, pop 100, reggae 100, rock 100

Nearest 5 songs by audio, for the first clip of each genre:
  blues.00000      -> country 2, rock 1, reggae 1, blues 1
  classical.00000  -> classical 4, jazz 1
  country.00000    -> pop 3, country 1, disco 1
  disco.00000      -> hiphop 3, reggae 2
  hiphop.00000     -> metal 3, disco 1, rock 1
  jazz.00000       -> classical 4, jazz 1
  metal.00000      -> hiphop 2, pop 1, reggae 1, metal 1
  pop.00000        -> pop 4, disco 1
  reggae.00000     -> disco 2, blues 1, jazz 1, country 1
  rock.00000       -> country 2, rock 2, metal 1
```

## License

MIT; see [LICENSE](LICENSE). GTZAN is not part of the repository; see [Data and licenses](#data-and-licenses).
