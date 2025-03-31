import os
import logging
import json
from typing import Dict, List, Optional, Any
import asyncpg
from datetime import datetime

from ..models.pipeline import Modality, Song, ProcessingJob

class PostgresClient:
    """Client for PostgreSQL database operations"""

    def __init__(self, user: Optional[str] = None, password: Optional[str] = None, 
                 host: Optional[str] = None, port: Optional[str] = None, 
                 database: Optional[str] = None):
        """Initialize PostgreSQL client with connection details"""
        self.user = user or os.environ.get("POSTGRES_USER", "postgres")
        self.password = password or os.environ.get("POSTGRES_PASSWORD", "postgres")
        self.host = host or os.environ.get("POSTGRES_HOST", "localhost")
        self.port = port or os.environ.get("POSTGRES_PORT", "5432")
        self.database = database or os.environ.get("POSTGRES_DB", "mmfeaturestore")
        self.logger = logging.getLogger(__name__)
        self.pool = None

    async def connect(self) -> None:
        """Connect to PostgreSQL"""
        try:
            self.pool = await asyncpg.create_pool(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database
            )
            self.logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Disconnected from PostgreSQL")

    async def create_tables(self) -> None:
        """Create tables if they don't exist"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            # Create songs table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS songs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artist TEXT,
                    album TEXT,
                    genre TEXT,
                    year INTEGER,
                    data_paths JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            ''')
            
            # Create jobs table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY,
                    song_id TEXT NOT NULL REFERENCES songs(id),
                    modalities JSONB NOT NULL,
                    status JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    error_message TEXT
                )
            ''')
            
            self.logger.info("Created tables if they didn't exist")

    async def create_song(self, song: Song) -> str:
        """Create a song in the database"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            song_id = song.id
            data_paths_json = json.dumps(song.data_paths)
            
            await conn.execute('''
                INSERT INTO songs (id, title, artist, album, genre, year, data_paths, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    title = $2,
                    artist = $3,
                    album = $4,
                    genre = $5,
                    year = $6,
                    data_paths = $7,
                    updated_at = $9
            ''', song_id, song.title, song.artist, song.album, song.genre, song.year,
                data_paths_json, song.created_at, song.updated_at)
            
            self.logger.info(f"Created song: {song_id}")
            return song_id

    async def get_song(self, song_id: str) -> Optional[Song]:
        """Get a song from the database"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT id, title, artist, album, genre, year, data_paths, created_at, updated_at
                FROM songs WHERE id = $1
            ''', song_id)
            
            if row:
                data_paths = json.loads(row['data_paths']) if row['data_paths'] else {}
                
                return Song(
                    id=row['id'],
                    title=row['title'],
                    artist=row['artist'],
                    album=row['album'],
                    genre=row['genre'],
                    year=row['year'],
                    data_paths=data_paths,
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            
            return None

    async def update_song_data_path(self, song_id: str, modality: Modality, path: str) -> None:
        """Update a song's data path for a specific modality"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            # First get the existing data paths
            row = await conn.fetchrow('''
                SELECT data_paths FROM songs WHERE id = $1
            ''', song_id)
            
            if not row:
                raise ValueError(f"Song {song_id} not found")
            
            data_paths = json.loads(row['data_paths']) if row['data_paths'] else {}
            data_paths[modality.value] = path
            
            # Update the data paths
            await conn.execute('''
                UPDATE songs SET data_paths = $1, updated_at = $2 WHERE id = $3
            ''', json.dumps(data_paths), datetime.now(), song_id)
            
            self.logger.info(f"Updated data path for song {song_id} / {modality.value}: {path}")

    async def create_processing_job(self, job: ProcessingJob) -> str:
        """Create a processing job in the database"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            modalities_json = json.dumps(job.modalities)
            status_json = json.dumps(job.status)
            
            await conn.execute('''
                INSERT INTO processing_jobs 
                (job_id, song_id, modalities, status, created_at, updated_at, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', job.job_id, job.song_id, modalities_json, status_json, 
                job.created_at, job.updated_at, job.error_message)
            
            self.logger.info(f"Created job: {job.job_id}")
            return job.job_id

    async def update_job_status(self, job_id: str, modality: Modality, status: str) -> None:
        """Update a job's status for a specific modality"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            # First get the existing status
            row = await conn.fetchrow('''
                SELECT status FROM processing_jobs WHERE job_id = $1
            ''', job_id)
            
            if not row:
                raise ValueError(f"Job {job_id} not found")
            
            job_status = json.loads(row['status']) if row['status'] else {}
            job_status[modality.value] = status
            
            # Update the status
            await conn.execute('''
                UPDATE processing_jobs SET status = $1, updated_at = $2 WHERE job_id = $3
            ''', json.dumps(job_status), datetime.now(), job_id)
            
            self.logger.info(f"Updated job {job_id} status: {modality.value} -> {status}")

    async def get_processing_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a processing job from the database"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT job_id, song_id, modalities, status, created_at, updated_at, error_message
                FROM processing_jobs WHERE job_id = $1
            ''', job_id)
            
            if row:
                modalities = json.loads(row['modalities']) if row['modalities'] else []
                status = json.loads(row['status']) if row['status'] else {}
                
                return ProcessingJob(
                    job_id=row['job_id'],
                    song_id=row['song_id'],
                    modalities=modalities,
                    status=status,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    error_message=row['error_message']
                )
            
            return None 