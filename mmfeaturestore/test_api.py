#!/usr/bin/env python3
import os
import sys
import requests
import json
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test-api")

def test_api():
    """Test the API using sample data"""
    base_url = "http://localhost:8000"
    sample_dir = Path(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    
    # Check if API is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            logger.error(f"API server is not healthy: {response.text}")
            logger.error("Please start the API server with 'python run.py' in another terminal")
            return False
        logger.info(f"API server is healthy: {response.json()}")
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to API server")
        logger.error("Please start the API server with 'python run.py' in another terminal")
        return False
    
    # Create a song
    logger.info("Creating a song...")
    song_data = {
        "title": "Test Rock Song API",
        "artist": "API Test Artist",
        "genre": "rock"
    }
    response = requests.post(f"{base_url}/songs", json=song_data)
    if response.status_code != 200:
        logger.error(f"Failed to create song: {response.text}")
        return False
    
    song = response.json()
    song_id = song["id"]
    logger.info(f"Created song with ID: {song_id}")
    
    # Update data paths
    logger.info("Updating data paths...")
    rock_audio = sample_dir / "genres_original/rock/rock.00000.wav"
    rock_image = sample_dir / "images_original/rock/rock00000.png"
    
    # Update audio path
    audio_data = {"path": str(rock_audio.absolute())}
    response = requests.put(f"{base_url}/songs/{song_id}/data/audio", json=audio_data)
    if response.status_code != 200:
        logger.error(f"Failed to update audio path: {response.text}")
        return False
    logger.info("Updated audio path")
    
    # Update image path
    image_data = {"path": str(rock_image.absolute())}
    response = requests.put(f"{base_url}/songs/{song_id}/data/image", json=image_data)
    if response.status_code != 200:
        logger.error(f"Failed to update image path: {response.text}")
        return False
    logger.info("Updated image path")
    
    # Process features
    logger.info("Processing features...")
    process_data = {"modalities": ["audio", "image"]}
    response = requests.post(f"{base_url}/songs/{song_id}/process", json=process_data)
    if response.status_code != 200:
        logger.error(f"Failed to process features: {response.text}")
        return False
    
    job = response.json()
    job_id = job["job_id"]
    logger.info(f"Created processing job with ID: {job_id}")
    
    # Wait for processing to complete
    logger.info("Waiting for processing to complete...")
    max_retries = 20
    retries = 0
    job_complete = False
    
    while retries < max_retries and not job_complete:
        time.sleep(3)
        response = requests.get(f"{base_url}/jobs/{job_id}")
        if response.status_code != 200:
            logger.error(f"Failed to get job status: {response.text}")
            retries += 1
            continue
        
        job = response.json()
        status = job["status"]
        logger.info(f"Job status: {status}")
        
        # Check if all modalities are processed
        if all(s == "processed" for s in status.values()):
            job_complete = True
        else:
            retries += 1
    
    if not job_complete:
        logger.error("Processing job did not complete in time")
        return False
    
    logger.info("Processing completed successfully")
    
    # Get song details
    logger.info("Getting song details...")
    response = requests.get(f"{base_url}/songs/{song_id}")
    if response.status_code != 200:
        logger.error(f"Failed to get song details: {response.text}")
        return False
    
    song = response.json()
    logger.info(f"Song details: {json.dumps(song, indent=2)}")
    
    # Get features
    logger.info("Getting features...")
    for modality in ["audio", "image"]:
        response = requests.get(f"{base_url}/songs/{song_id}/features/{modality}")
        if response.status_code != 200:
            logger.error(f"Failed to get {modality} features: {response.text}")
            continue
        
        features = response.json()
        logger.info(f"Got {modality} features with shape: {len(features['features'])}")
    
    # Search for similar songs by audio
    logger.info("Searching for similar songs by audio...")
    search_data = {"modality": "audio", "top_k": 5}
    response = requests.post(f"{base_url}/songs/{song_id}/similar", json=search_data)
    if response.status_code != 200:
        logger.error(f"Failed to search for similar songs: {response.text}")
    else:
        similar = response.json()
        logger.info(f"Similar songs by audio: {similar['similar_songs']}")
    
    # Search for similar songs by image
    logger.info("Searching for similar songs by image...")
    search_data = {"modality": "image", "top_k": 5}
    response = requests.post(f"{base_url}/songs/{song_id}/similar", json=search_data)
    if response.status_code != 200:
        logger.error(f"Failed to search for similar songs: {response.text}")
    else:
        similar = response.json()
        logger.info(f"Similar songs by image: {similar['similar_songs']}")
    
    logger.info("API tests completed successfully!")
    return True

if __name__ == "__main__":
    test_api() 