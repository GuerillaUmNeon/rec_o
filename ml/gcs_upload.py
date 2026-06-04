"""Upload a local model artifact to Google Cloud Storage."""

from pathlib import Path

from google.api_core.exceptions import Forbidden, GoogleAPIError
from google.auth.exceptions import GoogleAuthError, TransportError
from google.cloud import storage

from ml.config import (
    LATEST_MODEL_PATH,
    MODEL_BLOB_NAME,
    MODEL_BUCKET_NAME,
    MODEL_LOCAL_FILENAME,
    ML_OUTPUTS_DIR,
)


def resolve_model_path(path: str | None) -> Path:
    if path:
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    for candidate in (
        LATEST_MODEL_PATH,
        ML_OUTPUTS_DIR / MODEL_LOCAL_FILENAME,
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No model file found. Train first with: python -m ml.scripts.run_local"
    )


def upload_model_to_gcs(model_path: Path) -> None:
    if not MODEL_BUCKET_NAME:
        raise RuntimeError("MODEL_BUCKET_NAME is not set in .env")

    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_NAME)

    try:
        blob.upload_from_filename(model_path)
    except (Forbidden, GoogleAPIError, GoogleAuthError, TransportError) as exc:
        raise RuntimeError(f"GCS upload failed: {exc}") from exc

    print(f"Uploaded {model_path} → gs://{MODEL_BUCKET_NAME}/{MODEL_BLOB_NAME}")
