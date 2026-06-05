"""Deprecated — use app.artist and app.models instead."""

from app.artist import (
    enrich_artists_from_db,
    get_artist_model_info,
    load_artist_model,
    recommend_artist_ids,
)
from app.models import get_models_info, load_models

# Backward-compatible aliases
load_model = load_artist_model
get_model_info = get_artist_model_info
predict_playlist = recommend_artist_ids
predict_artist = enrich_artists_from_db

__all__ = [
    "load_model",
    "get_model_info",
    "load_models",
    "get_models_info",
    "predict_playlist",
    "predict_artist",
    "recommend_artist_ids",
    "enrich_artists_from_db",
]
