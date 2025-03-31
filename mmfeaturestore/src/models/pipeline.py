from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


class Modality(str, Enum):
    """Modalities for song features"""
    LYRICS = "lyrics"
    AUDIO = "audio"
    IMAGE = "image"


class FeatureStatus(str, Enum):
    """Status of feature processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Song(BaseModel):
    """Song metadata"""
    id: str
    title: str
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    data_paths: Dict[str, str] = {}
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class ProcessingJob(BaseModel):
    """Job for processing song features"""
    job_id: str
    song_id: str
    modalities: List[str]
    status: Dict[str, str] = {}
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    error_message: Optional[str] = None


class ProcessingResult(BaseModel):
    """Result of processing song features"""
    job_id: str
    song_id: str
    modality: str
    status: str
    feature_id: Optional[str] = None
    error_message: Optional[str] = None 