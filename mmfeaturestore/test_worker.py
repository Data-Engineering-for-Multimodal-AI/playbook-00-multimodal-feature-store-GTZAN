#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
import uuid
from pathlib import Path

# Setup Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.pipeline import Modality, Song, ProcessingJob
from src.db.postgres import PostgresClient
from src.worker.tasks import process_song_features, get_job_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test-worker")

async def test_worker():
    """Test the worker tasks directly"""
    logger.info("Starting worker test...")
    
    # Initialize postgres client
    postgres_client = PostgresClient(
        user="featurestore_user",
        password="",  # No password for local development
        host="localhost",
        port="5432",
        database="featurestore"
    )
    
    # Connect to postgres
    await postgres_client.connect()
    
    try:
        # Create test tables if needed
        await postgres_client.create_tables()
        
        # Get sample data paths
        sample_dir = Path("./sample_data")
        rock_audio = sample_dir / "genres_original/rock/rock.00000.wav"
        rock_image = sample_dir / "images_original/rock/rock00000.png"
        
        # Create a test song
        song_id = str(uuid.uuid4())
        song = Song(
            id=song_id,
            title="Worker Test Song",
            artist="Test Artist",
            genre="rock"
        )
        
        # Store the song
        await postgres_client.create_song(song)
        logger.info(f"Created test song with ID: {song_id}")
        
        # Update data paths
        await postgres_client.update_song_data_path(song_id, Modality.AUDIO, str(rock_audio.absolute()))
        await postgres_client.update_song_data_path(song_id, Modality.IMAGE, str(rock_image.absolute()))
        logger.info("Updated data paths")
        
        # Process the song directly
        logger.info("Processing song features directly...")
        job = await process_song_features(song_id, [Modality.AUDIO, Modality.IMAGE])
        
        logger.info(f"Processing completed with status: {job.status}")
        
        # Verify the job was processed by retrieving it
        retrieved_job = await get_job_status(job.job_id)
        logger.info(f"Retrieved job status: {retrieved_job.status}")
        
        # Success if all modalities were processed
        success = all(status == "processed" for status in retrieved_job.status.values())
        
        if success:
            logger.info("Worker test completed successfully!")
        else:
            logger.error("Worker test failed: Not all modalities were processed")
        
        return success
        
    finally:
        # Clean up
        await postgres_client.disconnect()

if __name__ == "__main__":
    result = asyncio.run(test_worker())
    sys.exit(0 if result else 1) 