"""Settings (environment variables override the defaults, which match docker-compose.yml)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("FEATURESTORE_DATA", ROOT / "sample_data")).resolve()
MODELS = ROOT / "models"

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql://{user}:{password}@{host}:{port}/{db}".format(
        user=os.environ.get("POSTGRES_USER", "featurestore_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "featurestore"),
        host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        port=os.environ.get("POSTGRES_PORT", "5433"),
        db=os.environ.get("POSTGRES_DB", "featurestore")))
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
COLLECTION_PREFIX = os.environ.get("MILVUS_COLLECTION_PREFIX", "song")   # collections: <prefix>_audio, <prefix>_image, ...
