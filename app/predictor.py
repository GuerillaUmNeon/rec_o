from datetime import datetime, timezone
import math
import os
from pathlib import Path
import tempfile

from dotenv import load_dotenv
from google.cloud import storage
import joblib
import pandas as pd

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

    if not blob.exists():
        return None

    blob.download_to_filename(CACHED_MODEL_PATH)
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

    model_path = _download_model_from_gcs() or get_latest_model_path()
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


def _prediction_to_playlist(pred_result) -> tuple[str, str]:
    if isinstance(pred_result, pd.Series):
        pred_result = pred_result.to_list()
    elif hasattr(pred_result, "tolist"):
        pred_result = pred_result.tolist()

    if isinstance(pred_result, dict):
        return str(pred_result["artist_name"]), str(pred_result["artist_genre"])

    if not isinstance(pred_result, (list, tuple)) or len(pred_result) < 2:
        raise RuntimeError("Model prediction must contain an artist name and a genre.")

    return str(pred_result[0]), str(pred_result[1])


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _recommend_from_artifact(
    artifact: dict,
    artist_name: str,
    genre_filter: str | None,
    top_n: int = 1,
) -> list[dict]:
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
    df_clean["artist_name"] = df_clean["artist_name"].fillna("")
    df_clean["genres"] = df_clean["genres"].fillna("")

    matches = df_clean[
        df_clean["artist_name"].str.lower() == artist_name.lower()
    ]
    if matches.empty:
        return []

    seed_index = matches.index[0]
    seed_position = df_clean.index.get_loc(seed_index)
    query_vector = vectorizer.transform([df_clean.iloc[seed_position]["genres"]])
    n_neighbors = min(len(df_clean), top_n + 50)
    distances, indices = recommender.kneighbors(query_vector, n_neighbors=n_neighbors)

    recommendations = []
    seed_artist_id = df_clean.iloc[seed_position]["artist_id"]

    for distance, idx in zip(distances[0], indices[0]):
        row = df_clean.iloc[idx]

        if row["artist_id"] == seed_artist_id:
            continue

        if genre_filter:
            genre = genre_filter.lower()
            genres = str(row["genres"]).lower()
            if genre not in genres:
                continue

        result = {
            "artist_id": int(row["artist_id"]),
            "artist_name": row["artist_name"],
            "genres": row["genres"],
            "similarity_score": round(float(1 - distance), 4),
        }

        for column in ("artist_gid", "artist_type", "area_name"):
            if column in df_clean.columns:
                result[column] = _clean_value(row[column])

        recommendations.append(result)

        if len(recommendations) >= top_n:
            break

    return recommendations


def predict_playlist(artist_name: str, artist_genre: str) -> tuple[str, str]:
    if model is None:
        load_latest_model()
    if model is None:
        raise RuntimeError("No model saved yet. Train a model and call save_model(model).")

    if isinstance(model, dict):
        recommendations = _recommend_from_artifact(model, artist_name, artist_genre)
        if not recommendations:
            raise RuntimeError("No recommendation found for this artist and genre.")
        recommendation = recommendations[0]
        return str(recommendation["artist_name"]), str(recommendation["genres"])

    data_dict = {
        "artist_name": [artist_name],
        "artist_genre": [artist_genre],
    }
    data_df = pd.DataFrame(data_dict)
    pred_result = model.predict(data_df)[0]
    return _prediction_to_playlist(pred_result)
