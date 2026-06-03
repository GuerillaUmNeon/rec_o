import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError, TransportError
from google.cloud import storage
import joblib

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_ROOT / "models"
DEFAULT_MODEL_FILENAME = "knn_baseline_model.pkl"
LATEST_MODEL_PATH = APP_ROOT / DEFAULT_MODEL_FILENAME
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
MODEL_BLOB_NAME = os.getenv("MODEL_BLOB_NAME", f"models/{DEFAULT_MODEL_FILENAME}")
CACHED_MODEL_PATH = Path(tempfile.gettempdir()) / DEFAULT_MODEL_FILENAME

model = None


def _iter_saved_models() -> list[Path]:
    candidates = []
    if LATEST_MODEL_PATH.is_file():
        candidates.append(LATEST_MODEL_PATH)
    if MODEL_DIR.is_dir():
        candidates.extend(MODEL_DIR.glob("*.pkl"))
        candidates.extend(MODEL_DIR.glob("*.joblib"))
    return candidates


def get_latest_model_path() -> Path | None:
    saved_models = _iter_saved_models()
    if not saved_models:
        return None
    return max(saved_models, key=lambda path: path.stat().st_mtime)


def _download_model_from_gcs() -> Path | None:
    if not MODEL_BUCKET_NAME:
        return None

    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_NAME)

    try:
        if not blob.exists():
            return None

        blob.download_to_filename(CACHED_MODEL_PATH)
    except (GoogleAPIError, GoogleAuthError, TransportError):
        return None

    return CACHED_MODEL_PATH


def _upload_model_to_gcs(model_path: Path) -> None:
    if not MODEL_BUCKET_NAME:
        return

    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_NAME)
    blob.upload_from_filename(model_path)


def load_latest_model():
    global model

    model_path = get_latest_model_path() or _download_model_from_gcs()
    if model_path is None:
        model = None
        return None

    model = joblib.load(model_path)
    return model


def save_model(trained_model, filename: str | None = None) -> Path:
    global model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"knn_baseline_model_{timestamp}.pkl"

    model_path = MODEL_DIR / Path(filename).name
    if model_path.suffix not in {".pkl", ".joblib"}:
        model_path = model_path.with_suffix(".pkl")

    joblib.dump(trained_model, model_path)
    joblib.dump(trained_model, LATEST_MODEL_PATH)
    _upload_model_to_gcs(LATEST_MODEL_PATH)
    model = trained_model

    return model_path


def _recommend_artist_ids_from_artifact(
    artifact: dict,
    artist_ids: list[int],
    top_n: int,
) -> list[int]:
    recommender = artifact["model"]
    vectorizer = artifact["vectorizer"]
    data = artifact.get("data")
    if data is None:
        data = artifact.get("df_clean")

    if data is None:
        raise RuntimeError(
            "Model artifact must contain df_clean as 'data'. "
            "Save {'vectorizer': vectorizer, 'model': knn_model, 'data': df_clean}."
        )

    df_clean = data.copy()
    df_clean["genres"] = df_clean["genres"].fillna("")

    seed_ids = {int(artist_id) for artist_id in artist_ids}
    matches = df_clean[df_clean["artist_id"].isin(seed_ids)]
    found_ids = {int(artist_id) for artist_id in matches["artist_id"].tolist()}
    missing_ids = sorted(seed_ids - found_ids)
    if missing_ids:
        raise RuntimeError(f"Unknown artist IDs: {missing_ids}")

    query_vectors = vectorizer.transform(matches["genres"].astype(str))
    if len(matches) == 1:
        query_vector = query_vectors[0]
    else:
        query_vector = query_vectors.mean(axis=0).A

    n_neighbors = min(len(df_clean), top_n + len(seed_ids) + 50)
    distances, indices = recommender.kneighbors(query_vector, n_neighbors=n_neighbors)

    recommendations = []
    for idx in indices[0]:
        row = df_clean.iloc[idx]
        artist_id = int(row["artist_id"])

        if artist_id in seed_ids or artist_id in recommendations:
            continue

        recommendations.append(artist_id)
        if len(recommendations) >= top_n:
            break

    return recommendations


def predict_playlist(artist_ids: list[int], top_n: int = 5) -> list[int]:
    if model is None:
        load_latest_model()
    if model is None:
        raise RuntimeError("No model saved yet. Train a model and call save_model(model).")

    if isinstance(model, dict):
        recommendations = _recommend_artist_ids_from_artifact(model, artist_ids, top_n)
        if not recommendations:
            raise RuntimeError("No recommendation found for these artist IDs.")
        return recommendations

    pred_result = model.predict([artist_ids])[0]
    if hasattr(pred_result, "tolist"):
        pred_result = pred_result.tolist()
    if not isinstance(pred_result, list):
        pred_result = list(pred_result)
    return [int(artist_id) for artist_id in pred_result[:top_n]]
