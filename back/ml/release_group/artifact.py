"""Save trained release group KNN artifacts locally (no GCS)."""

from pathlib import Path

import joblib

from back.ml.release_group.config import (
    MODEL_DIR,
    RELEASE_GROUP_CANONICAL_MODEL_PATH,
)


def save_release_group_knn_artifact(artifact: dict, filename: str | None = None) -> Path:
    """Dump one release group KNN artifact to the canonical models/ path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RELEASE_GROUP_CANONICAL_MODEL_PATH if filename is None else MODEL_DIR / Path(filename).name

    joblib.dump(artifact, model_path)

    print(f"Saved locally: {model_path}")

    return model_path
