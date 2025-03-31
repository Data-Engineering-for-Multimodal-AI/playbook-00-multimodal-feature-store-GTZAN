from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from enum import Enum
from datetime import datetime

class SongCreate(BaseModel):
    """Model for creating a new song"""
    title: str
    artist: str
    album: Optional[str] = None
    lyrics: Optional[str] = None
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Example Song",
                "artist": "Example Artist",
                "album": "Example Album",
                "lyrics": "These are the lyrics to the song...",
                "audio_path": "/path/to/audio.mp3",
                "image_path": "/path/to/cover.jpg"
            }
        }


class SongResponse(BaseModel):
    """Model for song response"""
    id: str
    title: str
    artist: str
    album: Optional[str] = None
    processing_job_id: Optional[str] = None
    status: str = "complete"  # "processing" or "complete"
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Example Song",
                "artist": "Example Artist",
                "album": "Example Album",
                "processing_job_id": "123e4567-e89b-12d3-a456-426614174001",
                "status": "processing"
            }
        }


class SongFeatures(BaseModel):
    """Model for song features (embeddings)"""
    song_id: str
    lyrics: Optional[List[float]] = None
    audio: Optional[List[float]] = None
    image: Optional[List[float]] = None


class SearchRequest(BaseModel):
    """Model for search request"""
    modality: str  # "lyrics", "audio", or "image"
    vector: List[float]
    top_k: int = Field(10, ge=1, le=100)
    
    class Config:
        schema_extra = {
            "example": {
                "modality": "lyrics",
                "vector": [0.1, 0.2, 0.3, 0.4],  # Shortened for example
                "top_k": 5
            }
        }


class SearchResponse(BaseModel):
    """Model for search response"""
    id: str
    title: str
    artist: str
    album: Optional[str] = None
    similarity: float
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Example Song",
                "artist": "Example Artist",
                "album": "Example Album",
                "similarity": 0.95
            }
        } 