# Changelog

## 2026.10 (2026-10-27)

The version before this update is tagged `first-edition`. Personal details were removed from the
repository's history before that tag was made.

### Fixed
- Image features came from a ResNet-18 with random weights; now ImageNet-pretrained.
- Audio features are standardized before comparing them; two features measured in Hz dominated.
- The API needed PostgreSQL, which docker-compose didn't start; jobs opened new connections per song;
  timestamps were fixed at import; similar-song searches returned the song itself.
- The process could crash on macOS (librosa's numba code with PyTorch loaded); zero-crossing rate is
  computed in NumPy.
- The example scripts used absolute paths from one machine.

### Added
- `download_data.py` (GTZAN from Kaggle, verified), `ingest_gtzan.py`, `evaluate.py` (genre retrieval,
  with the first edition's encoders for comparison).
- Tests on synthetic clips; Makefile; CI on Python 3.12 and 3.13 with PostgreSQL and Milvus.

### Changed
- Milvus 3.0.3 through pymilvus's `MilvusClient` API; PostgreSQL 17; services on 127.0.0.1.
- Pinned dependencies (`mmfeaturestore/uv.lock`, `requirements.txt`), Python 3.12+.
- README: GTZAN's images are spectrograms; measured results; data and licenses.

### Removed
- `mmfeaturestore/volumes/` (database files from one local run), an empty `data` submodule link,
  `new_data/`, the duplicate root `docker-compose.yml`, `batch_process.py`, `run_test.py` and six
  manual test scripts.
