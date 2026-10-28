"""Feature extraction for one song: encode each requested modality and store the vector in Milvus,
recording progress in the job's status in PostgreSQL."""
import asyncio
import logging
from pathlib import Path
from typing import List

from .. import config
from ..db.milvus import MilvusClient
from ..db.postgres import PostgresClient
from ..encoders.factory import EncoderFactory
from ..models.pipeline import FeatureStatus, Modality

logger = logging.getLogger(__name__)
_encode_lock = asyncio.Lock()   # one encoding at a time: the models are shared and use all CPU cores


def resolve_data_path(path: str) -> Path:
    """Data paths must point inside the data folder (FEATURESTORE_DATA, default sample_data/)."""
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else config.ROOT / candidate).resolve()
    if not resolved.is_relative_to(config.DATA_ROOT):
        raise ValueError(f"{path} is outside the data folder ({config.DATA_ROOT})")
    if not resolved.is_file():
        raise FileNotFoundError(f"{path} doesn't exist")
    return resolved


async def process_song_features(postgres: PostgresClient, milvus: MilvusClient, encoders: EncoderFactory,
                                song_id: str, modalities: List[Modality], job_id: str) -> None:
    song = await postgres.get_song(song_id)
    for modality in modalities:
        path = song.data_paths.get(modality.value) if song else None
        if not path:
            await postgres.update_job_status(job_id, modality, FeatureStatus.FAILED.value, f"no {modality.value} path")
            continue
        await postgres.update_job_status(job_id, modality, FeatureStatus.PROCESSING.value)
        try:
            source = resolve_data_path(path)
            encoder = encoders.create_encoder(modality)
            async with _encode_lock:
                if modality == Modality.LYRICS:
                    vector = await asyncio.to_thread(encoder.encode, source.read_text())
                else:
                    vector = await asyncio.to_thread(encoder.encode, str(source))
            milvus.upsert(modality, [song_id], [vector.tolist()])
            await postgres.update_job_status(job_id, modality, FeatureStatus.PROCESSED.value)
        except Exception as e:  # recorded on the job; the API reports it
            logger.exception(f"{modality.value} features failed for {song_id}")
            await postgres.update_job_status(job_id, modality, FeatureStatus.FAILED.value, f"{modality.value}: {e}")
