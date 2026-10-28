"""Milvus: one collection of vectors per modality (song_audio, song_image, song_lyrics), cosine
similarity, HNSW index. Uses pymilvus's MilvusClient API. Collections persist across restarts; the
song ID is the primary key, so processing a song again replaces its vector."""
import logging
from typing import Dict, List, Optional, Tuple

from pymilvus import DataType
from pymilvus import MilvusClient as _Milvus

from .. import config
from ..models.pipeline import Modality


class MilvusClient:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.uri = f"http://{host or config.MILVUS_HOST}:{port or config.MILVUS_PORT}"
        self.logger = logging.getLogger(__name__)
        self.client: Optional[_Milvus] = None
        self.ready: Dict[Modality, bool] = {}

    @property
    def connected(self) -> bool:
        return self.client is not None

    def connect(self) -> None:
        self.client = _Milvus(uri=self.uri)

    def disconnect(self) -> None:
        if self.client:
            self.client.close()
            self.client = None

    @staticmethod
    def name(modality: Modality) -> str:
        return f"{config.COLLECTION_PREFIX}_{modality.value}"

    def ensure(self, modality: Modality, dim: Optional[int] = None) -> bool:
        """Make sure the modality's collection exists (created when dim is given) and is loaded."""
        if self.ready.get(modality):
            return True
        name = self.name(modality)
        if not self.client.has_collection(name):
            if dim is None:
                return False
            schema = self.client.create_schema(auto_id=False, description=f"Song {modality.value} embeddings")
            schema.add_field("song_id", DataType.VARCHAR, max_length=64, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
            index = self.client.prepare_index_params()
            index.add_index("embedding", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
            self.client.create_collection(name, schema=schema, index_params=index)
            self.logger.info(f"Created collection {name} (dim {dim})")
        self.client.load_collection(name)
        self.ready[modality] = True
        return True

    def count(self, modality: Modality) -> int:
        if not self.ensure(modality):
            return 0
        return self.client.query(self.name(modality), filter="", output_fields=["count(*)"])[0]["count(*)"]

    def drop(self, modality: Modality) -> None:
        self.ready.pop(modality, None)
        if self.client.has_collection(self.name(modality)):
            self.client.drop_collection(self.name(modality))

    def upsert(self, modality: Modality, song_ids: List[str], embeddings: List[List[float]]) -> None:
        self.ensure(modality, len(embeddings[0]))
        self.client.upsert(self.name(modality), [{"song_id": s, "embedding": e} for s, e in zip(song_ids, embeddings)])

    def flush(self) -> None:
        for modality in self.ready:
            self.client.flush(self.name(modality))

    def get_embedding(self, song_id: str, modality: Modality) -> Optional[List[float]]:
        if not self.ensure(modality):
            return None
        rows = self.client.get(self.name(modality), ids=[song_id], output_fields=["embedding"])
        return [float(x) for x in rows[0]["embedding"]] if rows else None

    def all_embeddings(self, modality: Modality) -> Dict[str, List[float]]:
        if not self.ensure(modality):
            return {}
        out = {}
        iterator = self.client.query_iterator(self.name(modality), batch_size=500, filter="",
                                              output_fields=["song_id", "embedding"])
        while batch := iterator.next():
            out.update({row["song_id"]: [float(x) for x in row["embedding"]] for row in batch})
        iterator.close()
        return out

    def search_similar(self, modality: Modality, query: List[float], top_k: int = 10,
                       exclude: Optional[str] = None) -> List[Tuple[str, float]]:
        """The most similar songs (cosine similarity, highest first), without `exclude` (the query song)."""
        if not self.ensure(modality):
            return []
        hits = self.client.search(self.name(modality), data=[list(map(float, query))], limit=top_k,
                                  filter=f'song_id != "{_quote(exclude)}"' if exclude else "",
                                  search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k * 2)}},
                                  output_fields=["song_id"])
        return [(hit["entity"]["song_id"], float(hit["distance"])) for hit in hits[0]]


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
