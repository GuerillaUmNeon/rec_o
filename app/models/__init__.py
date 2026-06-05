"""Model registry: load and expose status for all recommender artifacts."""

_NOT_LOADED: dict = {
    "loaded": False,
    "source": None,
    "path": None,
    "filename": None,
    "gcs_uri": None,
}


def load_models() -> None:
    """Load all available recommender artifacts at startup."""
    from app.artist.loader import load_artist_model

    load_artist_model()
    # release_group: load_release_group_model()  # future
    # genre: load_genre_model()                  # future


def get_models_info() -> dict:
    """Return load metadata per model type (artist, release_group, genre)."""
    from app.artist.loader import get_artist_model_info

    return {
        "artist": get_artist_model_info(),
        "release_group": dict(_NOT_LOADED),
        "genre": dict(_NOT_LOADED),
    }
