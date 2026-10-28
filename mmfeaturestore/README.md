# mmfeaturestore

The feature store's code. Run the commands in the [repository README](../README.md) from this folder.

| Path | What |
|---|---|
| `src/api/main.py` | The REST API (FastAPI); `python run.py` starts it |
| `src/encoders/` | Audio (librosa), spectrogram (ResNet-18) and lyrics (sentence-transformers) encoders |
| `src/db/` | PostgreSQL (metadata, jobs) and Milvus (vectors) clients |
| `src/worker/tasks.py` | Background feature extraction for one song |
| `download_data.py`, `ingest_gtzan.py`, `evaluate.py` | GTZAN: download, load, measure |
| `test_new_song.py`, `test_similarity.py`, `explore_genres.py`, `test_api_curl.sh` | API examples |
| `models/audio_scaler.json` | Means and standard deviations for standardizing the audio features |
| `tests/` | pytest, on synthetic clips |
