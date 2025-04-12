#!/usr/bin/env python
import asyncio
import logging
from src.db.postgres import PostgresClient
from src.worker.tasks import find_similar_songs
from src.models.pipeline import Modality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def main():
    # Initialize database client
    postgres_client = PostgresClient(
        user="featurestore_user",
        password="",  # No password for local development
        database="featurestore"
    )
    
    await postgres_client.connect()
    
    try:
        # Get all songs IDs using a direct SQL query
        async with postgres_client.pool.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM songs LIMIT 5")
            
        song_ids = [row['id'] for row in rows]
        
        if not song_ids:
            logger.error("No songs found in the database")
            return
        
        logger.info(f"Found {len(song_ids)} songs")
        
        # Test similarity search for each song
        for song_id in song_ids:
            song = await postgres_client.get_song(song_id)
            if not song:
                logger.warning(f"Could not find song with ID: {song_id}")
                continue
                
            logger.info(f"\nTesting similarity search for song: {song.title} (ID: {song.id})")
            
            # Test audio similarity
            if 'audio' in song.data_paths:
                logger.info("Testing audio similarity search...")
                similar_songs_audio = await find_similar_songs(song.id, Modality.AUDIO, top_k=3)
                
                if similar_songs_audio:
                    logger.info(f"Found {len(similar_songs_audio)} similar songs by audio:")
                    for i, similar_song_id in enumerate(similar_songs_audio, 1):
                        similar_song = await postgres_client.get_song(similar_song_id)
                        if similar_song:
                            logger.info(f"  {i}. {similar_song.title}")
                else:
                    logger.info("No similar songs found for audio")
            
            # Test image similarity
            if 'image' in song.data_paths:
                logger.info("Testing image similarity search...")
                similar_songs_image = await find_similar_songs(song.id, Modality.IMAGE, top_k=3)
                
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