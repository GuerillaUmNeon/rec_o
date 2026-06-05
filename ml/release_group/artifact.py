"""Save trained release group KNN artifacts locally (no GCS)."""

from datetime import datetime, timezone
from pathlib import Path

import joblib

from ml.release_group.config import (
    ML_OUTPUTS_DIR,
    MODEL_DIR,
    RELEASE_GROUP_CANONICAL_MODEL_PATH,
    RELEASE_GROUP_MODEL_LOCAL_FILENAME,
)


def save_release_group_knn_artifact(artifact: dict, filename: str | None = None) -> Path:
    """Dump release group KNN artifact to models/ and ml/outputs/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ML_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    default_stem = Path(RELEASE_GROUP_MODEL_LOCAL_FILENAME).stem
    default_suffix = Path(RELEASE_GROUP_MODEL_LOCAL_FILENAME).suffix or ".pkl"

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{default_stem}_{timestamp}{default_suffix}"

    model_path = MODEL_DIR / Path(filename).name
    if model_path.suffix not in {".pkl", ".joblib"}:
        model_path = model_path.with_suffix(default_suffix)

    output_copy = ML_OUTPUTS_DIR / RELEASE_GROUP_MODEL_LOCAL_FILENAME

    joblib.dump(artifact, model_path)
    joblib.dump(artifact, RELEASE_GROUP_CANONICAL_MODEL_PATH)
    joblib.dump(artifact, output_copy)

    print(f"Saved locally: {model_path}")
    print(f"Canonical copy: {RELEASE_GROUP_CANONICAL_MODEL_PATH}")
    print(f"Copy in outputs: {output_copy}")

    return model_path
