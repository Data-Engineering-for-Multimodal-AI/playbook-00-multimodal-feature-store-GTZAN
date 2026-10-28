import time

import librosa
import numpy as np
import pytest

from conftest import DATA, GENRES
from src import config
from src.encoders.audio import AudioEncoder, zero_crossing_rate
from src.worker.tasks import resolve_data_path


def test_zero_crossing_rate_matches_librosa():
    y, _ = librosa.load(str(next((DATA / "genres_original" / "mid").glob("*.wav"))), sr=22050)
    assert abs(zero_crossing_rate(y) - float(librosa.feature.zero_crossing_rate(y).mean())) < 1e-3


def test_audio_features_are_standardized_with_the_scaler():
    clip = str(next((DATA / "genres_original" / "low").glob("*.wav")))
    raw = AudioEncoder.raw_features(clip)
    assert raw.shape == (28,)
    vector = AudioEncoder().encode(clip)
    assert abs(np.linalg.norm(vector) - 1) < 1e-6
    # Without the scaler the vector is dominated by the centroid and rolloff (thousands of Hz)
    plain = raw / np.linalg.norm(raw)
    assert abs(plain[13]) + abs(plain[27]) > 0.99 and abs(vector[13]) + abs(vector[27]) < 0.99


def test_data_paths_stay_in_the_data_folder():
    assert resolve_data_path(str(DATA / "genres_original" / "low" / "low.00000.wav")).is_file()
    with pytest.raises(ValueError):
        resolve_data_path("/etc/hosts")
    with pytest.raises(FileNotFoundError):
        resolve_data_path(str(DATA / "missing.wav"))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api import main
    try:
        with TestClient(main.app) as c:
            yield c, main
    except Exception as e:  # PostgreSQL or Milvus not running
        pytest.skip(f"services not available: {e}")


def test_api_flow(client):
    c, main = client
    ids = []
    for genre in GENRES:
        for i in range(3):
            song_id = f"pytest-{genre}.0000{i}"
            ids.append(song_id)
            assert c.post("/songs", json={"id": song_id, "title": song_id, "genre": genre}).status_code == 200
            assert c.put(f"/songs/{song_id}/data/audio",
                         json={"path": str(DATA / "genres_original" / genre / f"{genre}.0000{i}.wav")}).status_code == 200
            assert c.put(f"/songs/{song_id}/data/image",
                         json={"path": str(DATA / "images_original" / genre / f"{genre}0000{i}.png")}).status_code == 200
    assert c.put(f"/songs/{ids[0]}/data/audio", json={"path": "/etc/hosts"}).status_code == 400
    assert c.put(f"/songs/{ids[0]}/data/video", json={"path": "x"}).status_code == 400
    assert c.get("/songs/pytest-nope").status_code == 404

    jobs = [c.post(f"/songs/{s}/process", json={"modalities": ["audio", "image"]}).json()["job_id"] for s in ids]
    for job in jobs:
        for _ in range(240):
            status = c.get(f"/jobs/{job}").json()["status"]
            if all(v in ("processed", "failed") for v in status.values()):
                break
            time.sleep(0.25)
        assert status == {"audio": "processed", "image": "processed"}

    features = c.get(f"/songs/{ids[0]}/features/audio").json()
    assert features["dimensions"] == 28
    for modality in ("audio", "image"):
        similar = c.post(f"/songs/{ids[0]}/similar", json={"modality": modality, "top_k": 2}).json()["similar_songs"]
        assert ids[0] not in [s["song_id"] for s in similar]
        # The other two clips of the same synthetic genre are the nearest
        assert all(s["genre"] == "low" for s in similar), (modality, similar)

    async def cleanup():
        async with main.postgres.pool.acquire() as conn:
            await conn.execute("DELETE FROM songs WHERE id LIKE 'pytest-%'")
    c.portal.call(cleanup)   # on the TestClient's event loop, where the pool lives
    for modality in main.Modality:
        main.milvus.drop(modality)
