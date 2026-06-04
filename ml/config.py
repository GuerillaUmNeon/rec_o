import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

ML_OUTPUTS_DIR = PROJECT_ROOT / "ml" / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Prod default: knn_baseline_model.pkl — override in .env for local/test work
MODEL_LOCAL_FILENAME = os.getenv("MODEL_LOCAL_FILENAME", "knn_baseline_model.pkl")
LATEST_MODEL_PATH = PROJECT_ROOT / MODEL_LOCAL_FILENAME

MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
MODEL_BLOB_NAME = os.getenv(
    "MODEL_BLOB_NAME",
    "models/knn_baseline_model.pkl",
)

# Optional cap for faster local training (see ml/scripts/run_local.py --limit)
ML_MAX_ARTISTS = os.getenv("ML_MAX_ARTISTS")
TRAINING_FEATURES_CACHE = ML_OUTPUTS_DIR / "training_features.pkl"

# Batch size for extended genre SQL (scoped training)
ML_GENRE_CHUNK_SIZE = int(os.getenv("ML_GENRE_CHUNK_SIZE", "2000"))
