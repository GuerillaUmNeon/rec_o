import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

ML_OUTPUTS_DIR = PROJECT_ROOT / "ml" / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


def _env(primary: str, fallback: str | None = None, default: str | None = None) -> str | None:
    value = os.getenv(primary)
    if value:
        return value
    if fallback:
        return os.getenv(fallback, default)
    return default


# Artist artifact filename (local train + upload lookup)
ARTIST_MODEL_LOCAL_FILENAME = _env(
    "ARTIST_MODEL_LOCAL_FILENAME",
    "MODEL_LOCAL_FILENAME",
    "knn_model_test_joris_slim.pkl",
)
ARTIST_CANONICAL_MODEL_PATH = MODEL_DIR / ARTIST_MODEL_LOCAL_FILENAME

# GCS destination when uploading the artist artifact
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
ARTIST_MODEL_BLOB_NAME = _env(
    "ARTIST_MODEL_BLOB_NAME",
    "MODEL_BLOB_NAME",
    "models/knn_model_test_joris_slim.pkl",
)

# Optional cap for faster local artist training (see train_local --limit)
ARTIST_ML_MAX_ARTISTS = _env("ARTIST_ML_MAX_ARTISTS", "ML_MAX_ARTISTS")
ARTIST_TRAINING_FEATURES_CACHE = ML_OUTPUTS_DIR / "artist_training_features.pkl"

# Batch size for extended genre SQL (scoped artist training)
ARTIST_ML_GENRE_CHUNK_SIZE = int(
    _env("ARTIST_ML_GENRE_CHUNK_SIZE", "ML_GENRE_CHUNK_SIZE", "2000") or "2000"
)
