import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIST_MODEL_BLOB = "models/knn_baseline_model.pkl"

MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
ARTIST_MODEL_BLOB_NAME = os.getenv(
    "ARTIST_MODEL_BLOB_NAME",
    os.getenv("MODEL_BLOB_NAME", DEFAULT_ARTIST_MODEL_BLOB),
)


def artist_local_path_raw() -> str:
    return (
        os.getenv("ARTIST_MODEL_LOCAL_PATH", "").strip()
        or os.getenv("MODEL_LOCAL_PATH", "").strip()
    )
