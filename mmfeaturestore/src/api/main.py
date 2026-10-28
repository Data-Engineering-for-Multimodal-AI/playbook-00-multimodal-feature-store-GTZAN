"""
The feature store's REST API.

  POST /songs                              create (or update) a song
  GET  /songs/{id}                         a song and its data paths
  PUT  /songs/{id}/data/{modality}         set the file for a modality (audio, image, lyrics)
  POST /songs/{id}/process                 extract features in the background -> a job
  GET  /jobs/{job_id}                      the job's status per modality
  GET  /songs/{id}/features/{modality}     the stored vector
  POST /songs/{id}/similar                 the most similar songs, by one modality
  GET  /genres                             songs per genre
"""
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..db.milvus import MilvusClient
from ..db.postgres import PostgresClient
from ..encoders.factory import EncoderFactory
from ..models.pipeline import FeatureStatus, Modality, ProcessingJob, Song
from ..worker.tasks import process_song_features, resolve_data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
postgres = PostgresClient()
milvus = MilvusClient()
encoders = EncoderFactory()
background: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await postgres.connect()
    milvus.connect()
    yield
    for task in list(background):
        await task
    await postgres.disconnect()
    milvus.disconnect()


app = FastAPI(title="Multimodal Feature Store", version="2026.10",
              description="Audio and spectrogram features for music, in PostgreSQL and Milvus", lifespan=lifespan)


class SongCreate(BaseModel):
    id: Optional[str] = None
    title: str
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None


class DataPathUpdate(BaseModel):
    path: str


class ProcessingRequest(BaseModel):
    modalities: List[str] = [Modality.AUDIO.value, Modality.IMAGE.value]


class SimilarSongsRequest(BaseModel):
    modality: str
    top_k: int = 10


class SimilarSong(BaseModel):
    song_id: str
    title: str
    genre: Optional[str]
    similarity: float


class SimilarSongsResponse(BaseModel):
    song_id: str
    modality: str
    similar_songs: List[SimilarSong]


def _modality(value: str) -> Modality:
    try:
        return Modality(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid modality {value!r}: one of {[m.value for m in Modality]}")


async def _song(song_id: str) -> Song:
    song = await postgres.get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    return song


@app.get("/")
async def root():
    return {"message": "Multimodal Feature Store API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "postgres": postgres.pool is not None, "milvus": milvus.connected}


@app.post("/songs", response_model=Song)
async def create_song(song: SongCreate):
    created = Song(id=song.id or str(uuid.uuid4()), **song.model_dump(exclude={"id"}))
    await postgres.upsert_song(created)
    return await postgres.get_song(created.id)


@app.get("/songs/{song_id}", response_model=Song)
async def get_song(song_id: str):
    return await _song(song_id)


@app.get("/genres")
async def genres():
    return await postgres.genre_counts()


@app.put("/songs/{song_id}/data/{modality}", response_model=Song)
async def update_song_data_path(song_id: str, modality: str, data: DataPathUpdate):
    await _song(song_id)
    try:
        resolve_data_path(data.path)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    await postgres.update_song_data_path(song_id, _modality(modality), data.path)
    return await postgres.get_song(song_id)


@app.post("/songs/{song_id}/process", response_model=ProcessingJob)
async def process_song(song_id: str, request: ProcessingRequest = ProcessingRequest()):
    await _song(song_id)
    modalities = [_modality(m) for m in request.modalities]
    job = ProcessingJob(job_id=str(uuid.uuid4()), song_id=song_id, modalities=[m.value for m in modalities],
                        status={m.value: FeatureStatus.PENDING.value for m in modalities})
    await postgres.create_processing_job(job)
    task = asyncio.create_task(process_song_features(postgres, milvus, encoders, song_id, modalities, job.job_id))
    background.add(task)
    task.add_done_callback(background.discard)
    return job


@app.get("/jobs/{job_id}", response_model=ProcessingJob)
async def get_job(job_id: str):
    job = await postgres.get_processing_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/songs/{song_id}/features/{modality}")
async def get_song_features(song_id: str, modality: str):
    await _song(song_id)
    vector = milvus.get_embedding(song_id, _modality(modality))
    if vector is None:
        raise HTTPException(status_code=404, detail=f"No {modality} features for this song yet")
    return {"song_id": song_id, "modality": modality, "dimensions": len(vector), "embedding": vector}


@app.post("/songs/{song_id}/similar", response_model=SimilarSongsResponse)
async def find_similar_songs(song_id: str, request: SimilarSongsRequest):
    await _song(song_id)
    modality = _modality(request.modality)
    vector = milvus.get_embedding(song_id, modality)
    if vector is None:
        raise HTTPException(status_code=404, detail=f"No {modality.value} features for this song yet")
    hits = milvus.search_similar(modality, vector, max(1, min(request.top_k, 100)), exclude=song_id)
    songs = {s.id: s for s in await postgres.get_songs([h[0] for h in hits])}
    return SimilarSongsResponse(song_id=song_id, modality=modality.value, similar_songs=[
        SimilarSong(song_id=h[0], title=songs[h[0]].title if h[0] in songs else h[0],
                    genre=songs[h[0]].genre if h[0] in songs else None, similarity=round(h[1], 4)) for h in hits])
