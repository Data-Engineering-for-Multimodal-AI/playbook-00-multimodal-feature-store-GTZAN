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
from src.db.milvus import MilvusClient
from src.worker.tasks import process_song_features, find_similar_songs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test-workflow")

async def main():
    """Test the full workflow using sample data"""
    logger.info("Starting full workflow test with sample data...")
    
    # Initialize clients
    postgres_client = PostgresClient(
        user="featurestore_user",
        password="",
        host="localhost",
        port="5432",
        database="featurestore"
    )
    milvus_client = MilvusClient()
    
    # Connect to databases
    logger.info("Connecting to databases...")
    await postgres_client.connect()
    milvus_client.connect()
    
    try:
        # Get sample data paths
        sample_dir = Path("./sample_data")
        rock_audio = sample_dir / "genres_original/rock/rock.00000.wav"
        rock_image = sample_dir / "images_original/rock/rock00000.png"
        
        # Create songs for each genre
        logger.info("Creating songs...")
        rock_song_id = str(uuid.uuid4())
        rock_song = Song(
            id=rock_song_id,
            title="Rock Sample 00000",
            artist="Unknown Artist",
            genre="rock"
        )
        
        # Store songs in database
        await postgres_client.create_song(rock_song)
        
        # Update data paths
        logger.info("Updating data paths...")
        await postgres_client.update_song_data_path(rock_song_id, Modality.AUDIO, str(rock_audio.absolute()))
        await postgres_client.update_song_data_path(rock_song_id, Modality.IMAGE, str(rock_image.absolute()))
        
        # Process features
        logger.info("Processing features...")
        job = await process_song_features(rock_song_id, [Modality.AUDIO, Modality.IMAGE])
        logger.info(f"Processing job complete with status: {job.status}")
        
        # Verify the song was processed
        processed_song = await postgres_client.get_song(rock_song_id)
        logger.info(f"Processed song: {processed_song.title} with data paths: {processed_song.data_paths}")
        
        # Add one more song for similarity testing
        logger.info("Adding another song for similarity testing...")
        rock_song2_id = str(uuid.uuid4())
        rock_song2 = Song(
            id=rock_song2_id,
            title="Rock Sample 00001",
            artist="Unknown Artist",
            genre="rock"
        )
        
        await postgres_client.create_song(rock_song2)
        
        # Using another rock sample
        rock_audio2 = sample_dir / "genres_original/rock/rock.00001.wav"
        rock_image2 = sample_dir / "images_original/rock/rock00001.png"
        
        # Check if files exist
        if not rock_audio2.exists() or not rock_image2.exists():
            logger.warning("Second sample files don't exist, using the same samples")
            rock_audio2 = rock_audio
            rock_image2 = rock_image
        
        await postgres_client.update_song_data_path(rock_song2_id, Modality.AUDIO, str(rock_audio2.absolute()))
        await postgres_client.update_song_data_path(rock_song2_id, Modality.IMAGE, str(rock_image2.absolute()))
        
        job2 = await process_song_features(rock_song2_id, [Modality.AUDIO, Modality.IMAGE])
        logger.info(f"Second processing job complete with status: {job2.status}")
        
        # Test similarity search
        logger.info("Testing similarity search...")
        similar_songs_audio = await find_similar_songs(rock_song_id, Modality.AUDIO, 5)
        logger.info(f"Similar songs by audio: {similar_songs_audio}")
        
        similar_songs_image = await find_similar_songs(rock_song_id, Modality.IMAGE, 5)
        logger.info(f"Similar songs by image: {similar_songs_image}")
        
        logger.info("Full workflow test completed successfully!")
        
    finally:
        # Clean up
        logger.info("Cleaning up...")
        if milvus_client.connected:
            milvus_client.disconnect()
        await postgres_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 