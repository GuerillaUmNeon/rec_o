import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RELEASE_GROUP_MODEL_BLOB = "models/release_group_knn_model.pkl"

MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
RELEASE_GROUP_MODEL_BLOB_NAME = os.getenv(
    "RELEASE_GROUP_MODEL_BLOB_NAME",
    DEFAULT_RELEASE_GROUP_MODEL_BLOB,
)


def release_group_local_path_raw() -> str:
    explicit = os.getenv("RELEASE_GROUP_MODEL_LOCAL_PATH", "").strip()
    if explicit:
        return explicit

    filename = os.getenv("RELEASE_GROUP_MODEL_LOCAL_FILENAME", "").strip()
    if filename:
        candidate = APP_ROOT / "models" / filename
        if candidate.is_file():
            return str(candidate)

    return ""
