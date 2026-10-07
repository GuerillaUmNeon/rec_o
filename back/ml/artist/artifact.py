"""Save trained artist KNN artifacts locally (no GCS)."""

from pathlib import Path

import joblib

from back.ml.artist.config import (
    ARTIST_CANONICAL_MODEL_PATH,
    MODEL_DIR,
)


def save_artist_knn_artifact(artifact: dict, filename: str | None = None) -> Path:
    """Dump one artist KNN artifact to the canonical models/ path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIST_CANONICAL_MODEL_PATH if filename is None else MODEL_DIR / Path(filename).name

    joblib.dump(artifact, model_path)

    print(f"Saved locally: {model_path}")

    return model_path


# Backward-compatible alias
save_artifact = save_artist_knn_artifact
