#!/usr/bin/env python
import asyncio
import os
import logging
import uuid
from src.db.postgres import PostgresClient
from src.db.milvus import MilvusClient
from src.models.pipeline import Modality, Song
from src.worker.tasks import process_song_features, find_similar_songs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def main():
    # Define sample data paths
    audio_path = "mmfeaturestore/sample_data/genres_original/rock/rock.00000.wav"
    image_path = "mmfeaturestore/sample_data/images_original/rock/rock00000.png"
    
    # Initialize database clients
    postgres_client = PostgresClient(
        user="featurestore_user",
        password="",  # No password for local development
        database="featurestore"
    )
    
    await postgres_client.connect()
    
    try:
        # Create a unique song ID
        song_id = str(uuid.uuid4())
        
        # Create song record
        song = Song(
            id=song_id,
            title="Rock Sample 00000",
            artist="Sample Artist",
            album="Rock Collection",
            genre="rock",
            year=2024
        )
        
        # Insert into database
        await postgres_client.create_song(song)
        logger.info(f"Created song: {song_id} - {song.title}")
        
        # Set data paths
        await postgres_client.update_song_data_path(song_id, Modality.AUDIO, audio_path)
        logger.info(f"Added audio path: {audio_path}")
        
        await postgres_client.update_song_data_path(song_id, Modality.IMAGE, image_path)
        logger.info(f"Added image path: {image_path}")
        
        # Process features for both modalities
        modalities = [Modality.AUDIO, Modality.IMAGE]
        logger.info(f"Processing features for modalities: {modalities}")
        job = await process_song_features(song_id, modalities)
        logger.info(f"Processed features: {job.status}")
        
        # Test similarity search - audio
        logger.info("Testing audio similarity search...")
        similar_songs_audio = await find_similar_songs(song_id, Modality.AUDIO, top_k=3)
        
        if similar_songs_audio:
            logger.info(f"Found {len(similar_songs_audio)} similar songs by audio:")
            for i, similar_song_id in enumerate(similar_songs_audio, 1):
                similar_song = await postgres_client.get_song(similar_song_id)
                if similar_song:
                    logger.info(f"  {i}. {similar_song.title}")
        else:
            logger.info("No similar songs found for audio")
        
        # Test similarity search - image
        logger.info("Testing image similarity search...")
        similar_songs_image = await find_similar_songs(song_id, Modality.IMAGE, top_k=3)
        
        if similar_songs_image:
            logger.info(f"Found {len(similar_songs_image)} similar songs by image:")
            for i, similar_song_id in enumerate(similar_songs_image, 1):
                similar_song = await postgres_client.get_song(similar_song_id)
                if similar_song:
                    logger.info(f"  {i}. {similar_song.title}")
        else:
            logger.info("No similar songs found for image")
            
    finally:
        await postgres_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 