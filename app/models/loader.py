"""Shared artifact resolution: local path or GCS download."""

import tempfile
from pathlib import Path

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError, TransportError
from google.cloud import storage


def resolve_local_path(
    app_root: Path,
    raw_path: str,
    *,
    env_name: str,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = app_root / path
    if not path.is_file():
        raise RuntimeError(f"{env_name} not found: {path}")
    return path


def download_gcs_blob(
    *,
    bucket_name: str,
    blob_name: str,
) -> Path | None:
    cache_path = Path(tempfile.gettempdir()) / Path(blob_name).name
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    try:
        if not blob.exists():
            return None
        blob.download_to_filename(cache_path)
    except (GoogleAPIError, GoogleAuthError, TransportError):
        return None

    return cache_path


def resolve_artifact_path(
    *,
    app_root: Path,
    local_path: str | None,
    local_path_env: str,
    bucket_name: str | None,
    blob_name: str,
) -> tuple[Path, str] | None:
    if local_path:
        return resolve_local_path(app_root, local_path, env_name=local_path_env), "local"

    if bucket_name:
        gcs_path = download_gcs_blob(bucket_name=bucket_name, blob_name=blob_name)
        if gcs_path is not None:
            return gcs_path, "gcs"

    return None


def build_load_info(
    *,
    loaded: bool,
    source: str | None = None,
    path: Path | None = None,
    bucket_name: str | None = None,
    blob_name: str | None = None,
) -> dict:
    if not loaded:
        return {
            "loaded": False,
            "source": None,
            "path": None,
            "filename": None,
            "gcs_uri": None,
        }

    gcs_uri = None
    if source == "gcs" and bucket_name and blob_name:
        gcs_uri = f"gs://{bucket_name}/{blob_name}"

    return {
        "loaded": True,
        "source": source,
        "path": str(path),
        "filename": path.name if path else None,
        "gcs_uri": gcs_uri,
    }
