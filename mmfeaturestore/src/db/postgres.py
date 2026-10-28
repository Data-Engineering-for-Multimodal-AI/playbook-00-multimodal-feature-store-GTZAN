"""PostgreSQL: song metadata and data paths, and processing jobs (their status survives restarts)."""
import json
import logging
from typing import List, Optional

import asyncpg

from .. import config
from ..models.pipeline import Modality, ProcessingJob, Song

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    genre TEXT,
    year INTEGER,
    data_paths JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id TEXT PRIMARY KEY,
    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    modalities JSONB NOT NULL,
    status JSONB NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class PostgresClient:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or config.POSTGRES_DSN
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=8)
        await self.create_tables()

    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def create_tables(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA)

    async def upsert_song(self, song: Song) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO songs (id, title, artist, album, genre, year, data_paths)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)
                   ON CONFLICT (id) DO UPDATE SET title = $2, artist = $3, album = $4, genre = $5, year = $6,
                       data_paths = songs.data_paths || $7, updated_at = NOW()""",
                song.id, song.title, song.artist, song.album, song.genre, song.year, json.dumps(song.data_paths))

    async def get_song(self, song_id: str) -> Optional[Song]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM songs WHERE id = $1", song_id)
        return _song(row) if row else None

    async def get_songs(self, song_ids: List[str]) -> List[Song]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM songs WHERE id = ANY($1::text[])", song_ids)
        by_id = {r["id"]: _song(r) for r in rows}
        return [by_id[i] for i in song_ids if i in by_id]

    async def list_songs(self, genre: Optional[str] = None, limit: int = 1000) -> List[Song]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM songs WHERE ($1::text IS NULL OR genre = $1) ORDER BY id LIMIT $2",
                                    genre, limit)
        return [_song(r) for r in rows]

    async def genre_counts(self) -> dict:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT genre, COUNT(*) AS n FROM songs GROUP BY genre ORDER BY genre")
        return {r["genre"]: r["n"] for r in rows}

    async def update_song_data_path(self, song_id: str, modality: Modality, path: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE songs SET data_paths = data_paths || $2, updated_at = NOW() WHERE id = $1",
                               song_id, json.dumps({modality.value: path}))

    async def create_processing_job(self, job: ProcessingJob) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("INSERT INTO processing_jobs (job_id, song_id, modalities, status) VALUES ($1, $2, $3, $4)",
                               job.job_id, job.song_id, json.dumps(job.modalities), json.dumps(job.status))

    async def update_job_status(self, job_id: str, modality: Modality, status: str, error: Optional[str] = None) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE processing_jobs SET status = status || $2, updated_at = NOW(),
                       error_message = COALESCE($3, error_message) WHERE job_id = $1""",
                job_id, json.dumps({modality.value: status}), error)

    async def get_processing_job(self, job_id: str) -> Optional[ProcessingJob]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM processing_jobs WHERE job_id = $1", job_id)
        if not row:
            return None
        return ProcessingJob(job_id=row["job_id"], song_id=row["song_id"], modalities=json.loads(row["modalities"]),
                             status=json.loads(row["status"]), error_message=row["error_message"],
                             created_at=row["created_at"], updated_at=row["updated_at"])


def _song(row) -> Song:
    return Song(id=row["id"], title=row["title"], artist=row["artist"], album=row["album"], genre=row["genre"],
                year=row["year"], data_paths=json.loads(row["data_paths"]), created_at=row["created_at"],
                updated_at=row["updated_at"])
