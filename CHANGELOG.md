# Changelog

## Unreleased (v1.1.0)

Not printed in the book. Nothing the book shows has changed.

### Added
- `make similar FILE=...` (`scripts/similar.py`): the stored songs most like a WAV file or an image.
- `make check-data DIR=...` and `make load DIR=...` (`scripts/own_data.py`): check a folder of your
  own clips in the GTZAN layout, then store and search them. Tests in `tests/test_own_data.py`;
  `make smoke` runs `tests/test_smoke.py` as before.
- Three measured experiments on synthetic clips, no Docker: `make try-scaler`, `make try-image-crop`,
  `make try-noise` (`experiments/`).
- README: Run it, time tiers, Try this, Use your own data, What you end up with, Before production,
  What it costs; the project structure lists the new folders.
- CI: the Run it steps with checks on their output, the own-data checks, the three experiments, and
  the time to first output in the devcontainer job.
- `make setup`, `make run`, `make serve`, `make smoke`, `make data` and `make clean`. `make run` starts
  PostgreSQL and Milvus, stores audio and image features for 12 synthetic clips, and prints the most
  similar songs per genre. It needs no Kaggle account. `make data` downloads GTZAN from Kaggle
  (needs a Kaggle API token); `make run-gtzan` then runs the same steps on it.
- `pyproject.toml` and `uv.lock` (Python 3.12 or newer). `pip install -r requirements.txt` still works.
- `mmfeaturestore/quickstart.py` (the `make run` script) and `tests/test_smoke.py`.
- `.devcontainer/` so the repo opens in GitHub Codespaces, with Docker-in-Docker for the services.
- CI (`.github/workflows/smoke.yml`): Python 3.12 and 3.13, a weekly scheduled run, `make smoke`,
  `make run`, the `pip install -r requirements.txt` path with the API, and a job that builds the
  devcontainer and runs `make run` inside it.

### Fixed
- Audio features are standardized with `mmfeaturestore/models/audio_scaler.json` (statistics over 999
  GTZAN clips) before normalizing. Before, every pair of clips scored about 1.0.
- `make run` can run again with the services still up: deleting old vectors loads the Milvus
  collection first (Milvus 3 needs it loaded).
- PostgreSQL was not in `docker-compose.yml`, and the API hard-coded its connection settings. It now
  runs in compose on host port 5433, and `POSTGRES_*` environment variables set the connection.
- Image features came from a ResNet-18 with random weights. It now loads the ImageNet weights.
- A job created by `POST /songs/{id}/process` stayed `pending`: the worker created a second job and
  updated that one. The worker now updates the API's job.
- `data` was a submodule link with no `.gitmodules`. The link is removed.
- The GTZAN link (marsyas.info) no longer answers. The READMEs link to the Kaggle page.

### Changed
- README: "Overview" is now "How it works"; "Setup & Installation" is now "Run it the way the book
  does", with the same content.
- README opening: a title that says what you'll do, three sentences on why, the example and what you'll do, then the data engineering problem and what you'll work with.
- `mmfeaturestore/docker-compose.yml` is the one compose file. The root `docker-compose.yml` includes
  it, so `docker-compose up -d` starts the same containers from either folder. Milvus 3.0.3 (was
  2.3.3), etcd 3.5.25 and MinIO with health checks, data in named Docker volumes.
- `pymilvus` is kept below 3.1, which removes the ORM API that `src/db/milvus.py` uses.

### Removed
- `mmfeaturestore/volumes/` (102 MB): etcd, Milvus and MinIO files from one local run, which loaded
  stale data when compose ran from `mmfeaturestore/`. A first run recreates this state. That was
  checked before the removal: with the folder deleted, `make run` created fresh volumes, stored 12
  vectors per modality and returned search results (details in the pull request). The folder is now
  in `.gitignore`.

### Credits
These fixes were copied or adapted from the author's `update-2026` branch (closed PR #3):
- `mmfeaturestore/docker-compose.yml`: PostgreSQL 17 on 5433, Milvus 3.0.3, etcd, MinIO, health
  checks, named volumes and the volume-ownership step.
- `mmfeaturestore/src/encoders/image.py`: ImageNet weights for ResNet-18.
- `mmfeaturestore/src/config.py`: the PostgreSQL defaults (user, password, host, port, database).
- `mmfeaturestore/src/worker/tasks.py`: the worker updates the job it is given.
- `mmfeaturestore/download_data.py`: GTZAN from Kaggle with a checksum.
- `mmfeaturestore/pyproject.toml`: the dependency list and the CPU-only PyTorch index on Linux.
- `.github/workflows/smoke.yml`: the CI layout, the PyTorch install and the API health check.
- Removing `mmfeaturestore/volumes/` and the `data` link.
