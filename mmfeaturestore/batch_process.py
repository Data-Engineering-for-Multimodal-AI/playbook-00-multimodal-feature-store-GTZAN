#!/usr/bin/env python
import asyncio
import os
import glob
import logging
import uuid
from src.db.postgres import PostgresClient
from src.db.milvus import MilvusClient
from src.models.pipeline import Modality, Song
from src.worker.tasks import process_song_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def main():
    # Define sample data paths
    audio_samples = glob.glob("mmfeaturestore/sample_data/genres_original/*/*.wav")[:10]
    image_samples = glob.glob("mmfeaturestore/sample_data/images_original/*/*.png")[:10]
    
    # Initialize database clients
    postgres_client = PostgresClient(
        user="featurestore_user",
        password="",  # No password for local development
        database="featurestore"
    )
    
    await postgres_client.connect()
    
    try:
        # Process audio files
        logger.info(f"Processing {len(audio_samples)} audio files")
        for audio_path in audio_samples:
            genre = os.path.basename(os.path.dirname(audio_path))
            filename = os.path.basename(audio_path)
            song_id = str(uuid.uuid4())
            
            # Create song record
            song_title = f"{genre} - {filename}"
            song = Song(
                id=song_id,
                title=song_title,
                artist="Sample Artist",
                album=f"{genre} Collection",
                genre=genre,
                year=2024
            )
            
            # Insert into database
            await postgres_client.create_song(song)
            logger.info(f"Created song: {song_id} - {song_title}")
            
            # Set data path for audio
            await postgres_client.update_song_data_path(song_id, Modality.AUDIO, audio_path)
            
            # Find matching image if possible
            matching_image = None
            image_filename = filename.replace(".wav", ".png")
            for image_path in image_samples:
                if image_filename in image_path:
                    matching_image = image_path
                    break
            
            # If no exact match, just use the first image of the same genre
            if matching_image is None:
                for image_path in image_samples:
                    if genre in image_path:
                        matching_image = image_path
                        break
            
            if matching_image:
                await postgres_client.update_song_data_path(song_id, Modality.IMAGE, matching_image)
                logger.info(f"Added image path: {matching_image}")
                
                # Process features for both modalities
                modalities = [Modality.AUDIO, Modality.IMAGE]
                job = await process_song_features(song_id, modalities)
                logger.info(f"Processed features for {song_id}: {job.status}")
            else:
                # Process audio only
                modalities = [Modality.AUDIO]
                job = await process_song_features(song_id, modalities)
                logger.info(f"Processed audio features for {song_id}: {job.status}")
        
        logger.info("Batch processing completed!")
        
    finally:
        await postgres_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 