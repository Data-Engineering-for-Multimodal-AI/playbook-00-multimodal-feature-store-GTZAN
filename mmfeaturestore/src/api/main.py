import os
import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..models.pipeline import Modality, Song, ProcessingJob, ProcessingResult
from ..db.postgres import PostgresClient
from ..db.milvus import MilvusClient
from ..worker.tasks import process_song_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal Feature Store",
    description="API for storing and retrieving multimodal song features",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connections
postgres_client = PostgresClient(
    user="featurestore_user",
    password="",  # No password for local development
    host="localhost",
    port="5432",
    database="featurestore"
)
milvus_client = MilvusClient()

# Models for API requests/responses
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
    modalities: List[str] = [m.value for m in Modality]

class SimilarSongsRequest(BaseModel):
    modality: str
    top_k: int = 10

class SimilarSongsResponse(BaseModel):
    song_id: str
    similar_songs: List[str]
    modality: str

@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    await postgres_client.connect()
    await postgres_client.create_tables()
    
    try:
        milvus_client.connect()
        logger.info("Connected to Milvus")
    except Exception as e:
        logger.warning(f"Failed to connect to Milvus: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    await postgres_client.disconnect()
    if milvus_client.connected:
        milvus_client.disconnect()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Multimodal Feature Store API"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "postgres": postgres_client.connected,
        "milvus": milvus_client.connected
    }

@app.post("/songs", response_model=Song)
async def create_song(song: SongCreate):
    """Create a new song"""
    song_id = song.id or str(uuid.uuid4())
    
    song_model = Song(
        id=song_id,
        title=song.title,
        artist=song.artist,
        album=song.album,
        genre=song.genre,
        year=song.year
    )
    
    await postgres_client.create_song(song_model)
    return song_model

@app.get("/songs/{song_id}", response_model=Song)
async def get_song(song_id: str):
    """Get a song by ID"""
    song = await postgres_client.get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    return song

@app.put("/songs/{song_id}/data/{modality}", response_model=Song)
async def update_song_data_path(song_id: str, modality: str, data: DataPathUpdate):
    """Update a song's data path for a specific modality"""
    # Check if song exists
    song = await postgres_client.get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    
    # Check if modality is valid
    try:
        modality_enum = Modality(modality)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid modality: {modality}")
    
    # Update data path
    await postgres_client.update_song_data_path(song_id, modality_enum, data.path)
    
    # Return updated song
    return await postgres_client.get_song(song_id)

@app.post("/songs/{song_id}/process", response_model=ProcessingJob)
async def process_song(song_id: str, request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Process song features for specified modalities"""
    # Check if song exists
    song = await postgres_client.get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    
    # Check if modalities are valid
    modalities = []
    for m in request.modalities:
        try:
            modality_enum = Modality(m)
            modalities.append(modality_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid modality: {m}")
    
    # Create processing job
    job_id = str(uuid.uuid4())
    job = ProcessingJob(
        job_id=job_id,
        song_id=song_id,
        modalities=request.modalities,
        status={m: "pending" for m in request.modalities}
    )
    
    # Create the job in the database first
    await postgres_client.create_processing_job(job)
    
    # Run processing in background
    background_tasks.add_task(process_song_features, song_id, modalities)
    
    return job

@app.get("/jobs/{job_id}", response_model=ProcessingJob)
async def get_job(job_id: str):
    """Get a processing job by ID"""
    from ..worker.tasks import get_job_status
    
    job = await get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/songs/{song_id}/features/{modality}")
async def get_features(song_id: str, modality: str):
    """Get features for a song's modality"""
    from ..worker.tasks import get_song_feature
    
    # Check if modality is valid
    try:
        modality_enum = Modality(modality)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid modality: {modality}")
    
    # Get features
    features = await get_song_feature(song_id, modality_enum)
    if not features:
        raise HTTPException(status_code=404, detail=f"Features not found for song {song_id} / {modality}")
    
    return {"song_id": song_id, "modality": modality, "features": features}

@app.post("/songs/{song_id}/similar", response_model=SimilarSongsResponse)
async def find_similar(song_id: str, request: SimilarSongsRequest):
    """Find similar songs based on a specific modality"""
    from ..worker.tasks import find_similar_songs
    
    # Check if modality is valid
    try:
        modality_enum = Modality(request.modality)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid modality: {request.modality}")
    
    # Find similar songs
    similar_songs = await find_similar_songs(song_id, modality_enum, request.top_k)
    
    return SimilarSongsResponse(
        song_id=song_id,
        similar_songs=similar_songs,
        modality=request.modality
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 