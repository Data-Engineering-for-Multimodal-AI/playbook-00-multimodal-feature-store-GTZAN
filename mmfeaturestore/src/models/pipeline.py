from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Modality(str, Enum):
    """Modalities for song features"""
    LYRICS = "lyrics"
    AUDIO = "audio"
    IMAGE = "image"   # the GTZAN images are mel spectrograms of the audio


class FeatureStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Song(BaseModel):
    id: str
    title: str
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    data_paths: Dict[str, str] = {}
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ProcessingJob(BaseModel):
    job_id: str
    song_id: str
    modalities: List[str]
    status: Dict[str, str] = {}
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ProcessingResult(BaseModel):
    job_id: str
    song_id: str
    modality: str
    status: str
    error_message: Optional[str] = None
