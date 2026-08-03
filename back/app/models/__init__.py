"""Model registry: load and expose status for all recommender artifacts."""

_NOT_LOADED: dict = {
    "loaded": False,
    "source": None,
    "path": None,
    "filename": None,
}


def load_models() -> None:
    """Load all available recommender artifacts at startup."""
    from app.artist.loader import load_artist_model
    from app.release_group.loader import load_release_group_model

    load_artist_model()
    load_release_group_model()
    # genre: load_genre_model()  # future


def get_models_info() -> dict:
    """Return load metadata per model type (artist, release_group, genre)."""
    from app.artist.loader import get_artist_model_info
    from app.release_group.loader import get_release_group_model_info

    return {
        "artist": get_artist_model_info(),
        "release_group": get_release_group_model_info(),
        "genre": dict(_NOT_LOADED),
    }
