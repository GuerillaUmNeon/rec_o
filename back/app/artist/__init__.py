"""Artist KNN inference: load artifact, recommend IDs, enrich from DB."""

from app.artist.enrichment import enrich_artists_from_db
from app.artist.loader import get_artist_model_info, load_artist_model
from app.artist.recommender import recommend_artist_ids

__all__ = [
    "load_artist_model",
    "get_artist_model_info",
    "recommend_artist_ids",
    "enrich_artists_from_db",
]
