from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class ModalityFeature(BaseModel):
    modality: Modality
    data: Union[str, bytes]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class FeatureRequest(BaseModel):
    id: str
    features: List[ModalityFeature]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class FeatureResponse(BaseModel):
    id: str
    stored: bool
    modalities: List[Modality]
    message: Optional[str] = None

class SearchRequest(BaseModel):
    modality: Modality
    data: Union[str, bytes]
    limit: int = 10
    metadata_filter: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int
    message: Optional[str] = None 