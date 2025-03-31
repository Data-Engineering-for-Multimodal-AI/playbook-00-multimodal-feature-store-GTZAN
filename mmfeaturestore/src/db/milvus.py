import os
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    connections,
    utility,
)

from ..models.pipeline import Modality

class MilvusClient:
    """Client for Milvus vector database operations"""

    def __init__(self, host: Optional[str] = None, port: Optional[str] = None):
        """Initialize Milvus client with host and port"""
        self.host = host or os.environ.get("MILVUS_HOST", "localhost")
        self.port = port or os.environ.get("MILVUS_PORT", "19530")
        self.logger = logging.getLogger(__name__)
        self.collections = {}
        self.connected = False

    def connect(self) -> None:
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            self.connected = True
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Milvus"""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            self.logger.info("Disconnected from Milvus")

    def create_collection(self, modality: Modality, dim: int) -> None:
        """Create a collection for a specific modality if it doesn't exist"""
        collection_name = f"song_{modality.value}"
        
        if utility.has_collection(collection_name):
            self.logger.info(f"Collection {collection_name} already exists")
            self.collections[modality] = Collection(collection_name)
            return
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
            FieldSchema(name="song_id", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description=f"Song {modality.value} embeddings")
        
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        self.logger.info(f"Created collection {collection_name} with dimension {dim}")
        self.collections[modality] = collection

    def insert_embedding(self, song_id: str, modality: Modality, embedding: List[float]) -> str:
        """Insert embedding for a song"""
        collection_name = f"song_{modality.value}"
        
        if modality not in self.collections:
            self.create_collection(modality, len(embedding))
        
        collection = self.collections[modality]
        
        # Generate a unique ID for this embedding
        import uuid
        embedding_id = str(uuid.uuid4())
        
        # Insert the embedding
        data = [
            [embedding_id],
            [song_id],
            [embedding]
        ]
        collection.insert(data)
        
        self.logger.info(f"Inserted embedding for song {song_id} in modality {modality.value}")
        return embedding_id

    def get_embedding(self, song_id: str, modality: Modality) -> Optional[List[float]]:
        """Get embedding for a song"""
        collection_name = f"song_{modality.value}"
        
        if not utility.has_collection(collection_name):
            self.logger.warning(f"Collection {collection_name} does not exist")
            return None
        
        if modality not in self.collections:
            self.collections[modality] = Collection(collection_name)
        
        collection = self.collections[modality]
        collection.load()
        
        # Query the embedding
        results = collection.query(
            expr=f'song_id == "{song_id}"',
            output_fields=["embedding"]
        )
        
        collection.release()
        
        if not results:
            return None
        
        return results[0]["embedding"]

    def search_similar(self, modality: Modality, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar songs based on embedding"""
        collection_name = f"song_{modality.value}"
        
        if not utility.has_collection(collection_name):
            self.logger.warning(f"Collection {collection_name} does not exist")
            return []
        
        if modality not in self.collections:
            self.collections[modality] = Collection(collection_name)
        
        collection = self.collections[modality]
        collection.load()
        
        # Search for similar embeddings
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["song_id"]
        )
        
        collection.release()
        
        # Extract song_ids and distances
        similar_songs = []
        for hits in results:
            for hit in hits:
                similar_songs.append((hit.entity.get("song_id"), hit.distance))
        
        return similar_songs

    def delete_embedding(self, song_id: str, modality: Modality) -> None:
        """Delete embedding for a song"""
        collection_name = f"song_{modality.value}"
        
        if not utility.has_collection(collection_name):
            self.logger.warning(f"Collection {collection_name} does not exist")
            return
        
        if modality not in self.collections:
            self.collections[modality] = Collection(collection_name)
        
        collection = self.collections[modality]
        
        # Delete embeddings for this song
        collection.delete(f'song_id == "{song_id}"')
        
        self.logger.info(f"Deleted embeddings for song {song_id} in modality {modality.value}")
        
    def has_embedding(self, song_id: str, modality: Modality) -> bool:
        """Check if a song has an embedding for a specific modality"""
        embedding = self.get_embedding(song_id, modality)
        return embedding is not None 