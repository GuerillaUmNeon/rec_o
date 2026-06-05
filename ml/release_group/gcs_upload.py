"""Upload a local release group KNN artifact to Google Cloud Storage."""

from pathlib import Path

from google.api_core.exceptions import Forbidden, GoogleAPIError
from google.auth.exceptions import GoogleAuthError, TransportError
from google.cloud import storage

from ml.release_group.config import (
    ML_OUTPUTS_DIR,
    MODEL_BUCKET_NAME,
    RELEASE_GROUP_CANONICAL_MODEL_PATH,
    RELEASE_GROUP_MODEL_BLOB_NAME,
    RELEASE_GROUP_MODEL_LOCAL_FILENAME,
)


def resolve_release_group_knn_model_path(path: str | None) -> Path:
    if path:
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Release group model file not found: {model_path}")
        return model_path

    for candidate in (
        RELEASE_GROUP_CANONICAL_MODEL_PATH,
        ML_OUTPUTS_DIR / RELEASE_GROUP_MODEL_LOCAL_FILENAME,
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No release group KNN model found. Train first with: "
        "python -m ml.release_group.scripts.train_local"
    )


def upload_release_group_knn_model_to_gcs(model_path: Path) -> None:
    if not MODEL_BUCKET_NAME:
        raise RuntimeError("MODEL_BUCKET_NAME is not set in .env")

    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(RELEASE_GROUP_MODEL_BLOB_NAME)

    try:
        blob.upload_from_filename(model_path)
    except (Forbidden, GoogleAPIError, GoogleAuthError, TransportError) as exc:
        raise RuntimeError(f"GCS upload failed: {exc}") from exc

    print(
        f"Uploaded {model_path} → gs://{MODEL_BUCKET_NAME}/{RELEASE_GROUP_MODEL_BLOB_NAME}"
    )
