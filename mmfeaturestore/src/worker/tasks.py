import asyncio
import time
import logging
import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from ..models.pipeline import Modality, FeatureStatus, ProcessingJob, ProcessingResult
from ..db.postgres import PostgresClient
from ..db.milvus import MilvusClient
from ..encoders.factory import EncoderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def process_song_features(song_id: str, modalities: List[Modality] = None) -> ProcessingJob:
    """
    Process song features for specified modalities
    
    Args:
        song_id: ID of the song to process
        modalities: List of modalities to process, or None for all
    
    Returns:
        ProcessingJob object with status
    """
    logger.info(f"Starting processing of song {song_id} for modalities {modalities}")
    
    # Initialize clients
    logger.info(f"Initializing clients for song {song_id}")
    postgres_client = PostgresClient(
        user="featurestore_user",
        password="",  # No password for local development
        database="featurestore"
    )
    milvus_client = MilvusClient()
    
    # Connect to databases
    await postgres_client.connect()
    try:
        milvus_client.connect()
    except Exception as e:
        logger.warning(f"Failed to connect to Milvus, vector operations will be disabled: {e}")
    
    try:
        # Get song data
        song = await postgres_client.get_song(song_id)
        if not song:
            raise ValueError(f"Song {song_id} not found")
        
        # Determine which modalities to process
        if modalities is None:
            modalities = []
            for modality in Modality:
                if modality.value in song.data_paths:
                    modalities.append(modality)
        
        # Create a processing job
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            song_id=song_id,
            modalities=[m.value for m in modalities],
            status={m.value: FeatureStatus.PENDING.value for m in modalities}
        )
        
        await postgres_client.create_processing_job(job)
        
        # Process each modality
        encoder_factory = EncoderFactory()
        
        for modality in modalities:
            # Skip if no data path
            if modality.value not in song.data_paths:
                await postgres_client.update_job_status(
                    job_id, modality, FeatureStatus.FAILED.value
                )
                continue
            
            # Update status to processing
            await postgres_client.update_job_status(
                job_id, modality, FeatureStatus.PROCESSING.value
            )
            
            try:
                # Get data path
                data_path = song.data_paths[modality.value]
                
                # Get encoder for this modality
                encoder = encoder_factory.create_encoder(modality)
                
                # Process data
                if modality == Modality.LYRICS:
                    # For lyrics, we need to read the file
                    with open(data_path, 'r') as f:
                        text = f.read()
                    features = encoder.encode(text)
                else:
                    # For audio and images, we pass the file path
                    features = encoder.encode(data_path)
                
                # Store features in vector database
                if milvus_client.connected:
                    embedding_id = milvus_client.insert_embedding(song_id, modality, features.tolist())
                    logger.info(f"Stored {modality.value} features for song {song_id}")
                else:
                    logger.warning(f"Milvus not connected, skipping feature storage for {modality.value}")
                
                # Update status to processed
                await postgres_client.update_job_status(
                    job_id, modality, FeatureStatus.PROCESSED.value
                )
                
            except Exception as e:
                logger.error(f"Error processing {modality.value} for song {song_id}: {e}")
                await postgres_client.update_job_status(
                    job_id, modality, FeatureStatus.FAILED.value
                )
        
        # Get the updated job
        job = await postgres_client.get_processing_job(job_id)
        return job
    
    finally:
        # Clean up
        if milvus_client.connected:
            milvus_client.disconnect()
        await postgres_client.disconnect()

async def get_job_status(job_id: str) -> Optional[ProcessingJob]:
    """Get the status of a processing job"""
    postgres_client = PostgresClient(
        user="featurestore_user",
        password="featurestore_password",
        database="featurestore"
    )
    await postgres_client.connect()
    
    try:
        job = await postgres_client.get_processing_job(job_id)
        return job
    finally:
        await postgres_client.disconnect()

async def get_song_feature(song_id: str, modality: Modality) -> Optional[List[float]]:
    """Get features for a song's modality"""
    logger.info(f"Getting features for song {song_id} / {modality.value}")
    milvus_client = MilvusClient()
    
    try:
        milvus_client.connect()
        features = milvus_client.get_embedding(song_id, modality)
        if features is None:
            logger.warning(f"No features found for song {song_id} / {modality.value}")
            return None
        
        # Ensure features is a list, not a numpy array
        if hasattr(features, 'tolist'):
            features = features.tolist()
            
        logger.info(f"Retrieved features for song {song_id} / {modality.value} with length {len(features)}")
        return features
    except Exception as e:
        logger.error(f"Error getting features for song {song_id} / {modality.value}: {e}")
        return None
    finally:
        if milvus_client.connected:
            milvus_client.disconnect()

async def find_similar_songs(song_id: str, modality: Modality, top_k: int = 10) -> List[str]:
    """Find similar songs based on a specific modality"""
    milvus_client = MilvusClient()
    
    try:
        milvus_client.connect()
        
        # Get the embedding for the source song
        embedding = milvus_client.get_embedding(song_id, modality)
        if not embedding:
            logger.warning(f"No embedding found for song {song_id} / {modality.value}")
            return []
        
        # Search for similar songs
        similar_songs = milvus_client.search_similar(modality, embedding, top_k)
        
        # Extract song IDs
        song_ids = [s[0] for s in similar_songs if s[0] != song_id]
        
        return song_ids
    except Exception as e:
        logger.error(f"Error finding similar songs for {song_id} / {modality.value}: {e}")
        return []
    finally:
        if milvus_client.connected:
            milvus_client.disconnect() 