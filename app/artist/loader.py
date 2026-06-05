"""Load the artist KNN artifact at startup (local file or GCS)."""

import joblib

from app.artist.config import (
    APP_ROOT,
    ARTIST_MODEL_BLOB_NAME,
    MODEL_BUCKET_NAME,
    artist_local_path_raw,
)
from app.models.loader import build_load_info, resolve_artifact_path

_artist_model = None
_artist_model_load_info: dict = build_load_info(loaded=False)


def get_artist_model_info() -> dict:
    return dict(_artist_model_load_info)


def load_artist_model():
    """Load artist artifact from ARTIST_MODEL_LOCAL_PATH or GCS."""
    global _artist_model, _artist_model_load_info

    resolved = resolve_artifact_path(
        app_root=APP_ROOT,
        local_path=artist_local_path_raw() or None,
        local_path_env="ARTIST_MODEL_LOCAL_PATH",
        bucket_name=MODEL_BUCKET_NAME,
        blob_name=ARTIST_MODEL_BLOB_NAME,
    )

    if resolved is None:
        _artist_model = None
        _artist_model_load_info = build_load_info(loaded=False)
        return None

    model_path, source = resolved
    _artist_model = joblib.load(model_path)
    _artist_model_load_info = build_load_info(
        loaded=True,
        source=source,
        path=model_path,
        bucket_name=MODEL_BUCKET_NAME,
        blob_name=ARTIST_MODEL_BLOB_NAME,
    )
    return _artist_model


def get_artist_model():
    return _artist_model
