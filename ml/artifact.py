"""
Save trained artifacts locally (no GCS).

Copied from app/predictor.py (save only; upload is ml/scripts/upload_to_gcs.py).
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib

from ml.config import (
    CANONICAL_MODEL_PATH,
    MODEL_DIR,
    MODEL_LOCAL_FILENAME,
    ML_OUTPUTS_DIR,
)


def save_artifact(artifact: dict, filename: str | None = None) -> Path:
    """Dump artifact to models/ (canonical + timestamped) and ml/outputs/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ML_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    default_stem = Path(MODEL_LOCAL_FILENAME).stem
    default_suffix = Path(MODEL_LOCAL_FILENAME).suffix or ".pkl"

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{default_stem}_{timestamp}{default_suffix}"

    model_path = MODEL_DIR / Path(filename).name
    if model_path.suffix not in {".pkl", ".joblib"}:
        model_path = model_path.with_suffix(default_suffix)

    output_copy = ML_OUTPUTS_DIR / MODEL_LOCAL_FILENAME

    joblib.dump(artifact, model_path)
    joblib.dump(artifact, CANONICAL_MODEL_PATH)
    joblib.dump(artifact, output_copy)

    print(f"Saved locally: {model_path}")
    print(f"Canonical copy: {CANONICAL_MODEL_PATH}")
    print(f"Copy in outputs: {output_copy}")

    return model_path
