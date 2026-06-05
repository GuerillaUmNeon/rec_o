import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIST_MODEL_BLOB = "models/knn_model_test_joris_slim.pkl"

MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
ARTIST_MODEL_BLOB_NAME = os.getenv(
    "ARTIST_MODEL_BLOB_NAME",
    os.getenv("MODEL_BLOB_NAME", DEFAULT_ARTIST_MODEL_BLOB),
)


def artist_local_path_raw() -> str:
    explicit = (
        os.getenv("ARTIST_MODEL_LOCAL_PATH", "").strip()
        or os.getenv("MODEL_LOCAL_PATH", "").strip()
    )
    if explicit:
        return explicit

    filename = os.getenv("ARTIST_MODEL_LOCAL_FILENAME", "").strip()
    if filename:
        candidate = APP_ROOT / "models" / filename
        if candidate.is_file():
            return str(candidate)

    return ""
