#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
import uuid
import requests
import json
from pathlib import Path

# Setup Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.pipeline import Modality, Song, ProcessingJob
from src.db.postgres import PostgresClient
from src.db.milvus import MilvusClient
from src.worker.tasks import process_song_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test-direct")

async def create_process_check():
    """Test the API and worker in one script"""
    # Start with a direct API call to create a song
    base_url = "http://localhost:8000"
    
    try:
        # Create a song via API
        song_data = {
            "title": "Direct Test Song",
            "artist": "Direct Test Artist",
            "genre": "rock"
        }
        response = requests.post(f"{base_url}/songs", json=song_data)
        if response.status_code != 200:
            logger.error(f"Failed to create song: {response.text}")
            return False
        
        song = response.json()
        song_id = song["id"]
        logger.info(f"Created song with ID: {song_id}")
        
        # Update data paths via API
        sample_dir = Path(os.path.dirname(os.path.abspath(__file__)), "sample_data")
        rock_audio = sample_dir / "genres_original/rock/rock.00000.wav"
        rock_image = sample_dir / "images_original/rock/rock00000.png"
        
        audio_data = {"path": str(rock_audio.absolute())}
        response = requests.put(f"{base_url}/songs/{song_id}/data/audio", json=audio_data)
        if response.status_code != 200:
            logger.error(f"Failed to update audio path: {response.text}")
            return False
        
        image_data = {"path": str(rock_image.absolute())}
        response = requests.put(f"{base_url}/songs/{song_id}/data/image", json=image_data)
        if response.status_code != 200:
            logger.error(f"Failed to update image path: {response.text}")
            return False
        
        logger.info("Updated data paths")
        
        # Now process directly instead of via API
        logger.info("Processing features directly...")
        job = await process_song_features(song_id, [Modality.AUDIO, Modality.IMAGE])
        
        logger.info(f"Direct processing job completed with status: {job.status}")
        
        # Now check via API if the job status is reflected
        logger.info("Checking job status via API...")
        response = requests.get(f"{base_url}/jobs/{job.job_id}")
        if response.status_code != 200:
            logger.error(f"Failed to get job status: {response.text}")
            return False
        
        api_job = response.json()
        logger.info(f"API job status: {api_job['status']}")
        
        # Check features via API
        logger.info("Checking features via API...")
        for modality in ["audio", "image"]:
            response = requests.get(f"{base_url}/songs/{song_id}/features/{modality}")
            if response.status_code != 200:
                logger.error(f"Failed to get {modality} features: {response.text}")
                continue
            
            features = response.json()
            logger.info(f"Got {modality} features with length: {len(features['features'])}")
        
        # Search for similar songs
        logger.info("Searching for similar songs...")
        search_data = {"modality": "audio", "top_k": 5}
        response = requests.post(f"{base_url}/songs/{song_id}/similar", json=search_data)
        if response.status_code == 200:
            similar = response.json()
            logger.info(f"Similar songs: {similar['similar_songs']}")
        else:
            logger.error(f"Failed to search for similar songs: {response.text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(create_process_check())
    sys.exit(0 if result else 1) 