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


RELEASE_GROUP_MODEL_LOCAL_FILENAME = _env(
    "RELEASE_GROUP_MODEL_LOCAL_FILENAME",
    None,
    "release_group_knn_model.pkl",
)
RELEASE_GROUP_CANONICAL_MODEL_PATH = MODEL_DIR / RELEASE_GROUP_MODEL_LOCAL_FILENAME

MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
RELEASE_GROUP_MODEL_BLOB_NAME = _env(
    "RELEASE_GROUP_MODEL_BLOB_NAME",
    None,
    "models/release_group_knn_model.pkl",
)

RELEASE_GROUP_ML_MAX_ROWS = _env("RELEASE_GROUP_ML_MAX_ROWS")
RELEASE_GROUP_TRAINING_FEATURES_CACHE = ML_OUTPUTS_DIR / "release_group_training_features.pkl"
RELEASE_GROUP_ML_TRACK_META_CHUNK_SIZE = int(
    _env("RELEASE_GROUP_ML_TRACK_META_CHUNK_SIZE", None, "5000") or "5000"
)

DEFAULT_N_NEIGHBORS = int(_env("RELEASE_GROUP_ML_N_NEIGHBORS", None, "10") or "10")

TRACKS_FOR_ALBUM = 8
LENGTH_MS_FOR_ALBUM = 30 * 60 * 1000
LENGTH_MS_FOR_SINGLE = 7 * 60 * 1000
