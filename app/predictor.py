"""Inference only: load artifact, predict playlist, enrich artist rows. Training is in ml/."""

import os
from pathlib import Path
import tempfile

import pandas as pd
from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError, TransportError
from google.cloud import storage
import joblib

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIST_MODEL_BLOB = "models/knn_baseline_model.pkl"
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
ARTIST_MODEL_BLOB_NAME = os.getenv(
    "ARTIST_MODEL_BLOB_NAME",
    os.getenv("MODEL_BLOB_NAME", DEFAULT_ARTIST_MODEL_BLOB),
)

model = None
model_load_info: dict = {
    "loaded": False,
    "source": None,
    "path": None,
    "filename": None,
    "gcs_uri": None,
}


def _local_model_path() -> Path | None:
    """ARTIST_MODEL_LOCAL_PATH from .env (absolute or relative to project root)."""
    raw = (
        os.getenv("ARTIST_MODEL_LOCAL_PATH", "").strip()
        or os.getenv("MODEL_LOCAL_PATH", "").strip()
    )
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = APP_ROOT / path
    if not path.is_file():
        raise RuntimeError(f"ARTIST_MODEL_LOCAL_PATH not found: {path}")
    return path


def _download_model_from_gcs() -> Path | None:
    if not MODEL_BUCKET_NAME:
        return None

    cache_path = Path(tempfile.gettempdir()) / Path(ARTIST_MODEL_BLOB_NAME).name
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET_NAME)
    blob = bucket.blob(ARTIST_MODEL_BLOB_NAME)

    try:
        if not blob.exists():
            return None
        blob.download_to_filename(cache_path)
    except (GoogleAPIError, GoogleAuthError, TransportError):
        return None

    return cache_path


def _resolve_model_path() -> tuple[Path, str] | None:
    local = _local_model_path()
    if local is not None:
        return local, "local"
    gcs_path = _download_model_from_gcs()
    if gcs_path is not None:
        return gcs_path, "gcs"
    return None


def _set_model_load_info(*, loaded: bool, source: str | None = None, path: Path | None = None) -> None:
    global model_load_info
    if not loaded:
        model_load_info = {
            "loaded": False,
            "source": None,
            "path": None,
            "filename": None,
            "gcs_uri": None,
        }
        return

    info = {
        "loaded": True,
        "source": source,
        "path": str(path),
        "filename": path.name if path else None,
        "gcs_uri": None,
    }
    if source == "gcs" and MODEL_BUCKET_NAME:
        info["gcs_uri"] = f"gs://{MODEL_BUCKET_NAME}/{ARTIST_MODEL_BLOB_NAME}"
    model_load_info = info


def get_model_info() -> dict:
    """Return which artifact is loaded and whether it came from local disk or GCS."""
    return dict(model_load_info)


def load_model():
    """Load artist artifact from ARTIST_MODEL_LOCAL_PATH or GCS (MODEL_BUCKET_NAME + ARTIST_MODEL_BLOB_NAME)."""
    global model

    resolved = _resolve_model_path()
    if resolved is None:
        model = None
        _set_model_load_info(loaded=False)
        return None

    model_path, source = resolved
    model = joblib.load(model_path)
    _set_model_load_info(loaded=True, source=source, path=model_path)
    return model


def predict_artist(artist_ids, conn):
    """
    Return one row per artist with genres aggregated as a list.

    Parameters
    ----------
    artist_ids : list[int]
        List of MusicBrainz artist.id values.
    conn :
        DB connection compatible with pandas.read_sql_query.

    Returns
    -------
    pd.DataFrame
        Columns: id, gid, name, genre
        genre is a list of unique non-null genres for each artist.
    """
    if not artist_ids:
        return pd.DataFrame(columns=["id", "gid", "name", "genre", "url"])

    placeholders = ",".join(["%s"] * len(artist_ids))

    query = f"""
        SELECT
            artist.id AS id,
            artist.gid AS gid,
            artist.name AS name,
            genre.name AS genre,
            url.url AS urls,
            link_type.id AS link_type_id
        FROM artist
        LEFT JOIN artist_tag
            ON artist_tag.artist = artist.id
        LEFT JOIN tag
            ON artist_tag.tag = tag.id
        LEFT JOIN genre
            ON tag.name = genre.name
        LEFT JOIN l_artist_url
            ON l_artist_url.entity0 = artist.id
        LEFT JOIN url
            ON url.id = l_artist_url.entity1
        LEFT JOIN link
            ON link.id = l_artist_url.link
        LEFT JOIN link_type
            ON link_type.id = link.link_type
        WHERE artist.id IN ({placeholders})
          AND artist_tag.count > 0
    """

    result = pd.read_sql_query(query, conn, params=artist_ids)

    if result.empty:
        return pd.DataFrame(columns=["id", "gid", "name", "genre", "urls"])

    def agg_genres(series):
        return [g.capitalize() for g in series.dropna().unique().tolist()]

    def agg_urls(group):
        pairs = (
            group[["urls", "link_type_id"]]
            .dropna(subset=["urls", "link_type_id"])
            .drop_duplicates()
            .to_dict("records")
        )
        return [
            {"url": item["urls"], "type": int(item["link_type_id"])}
            for item in pairs
        ]

    grouped = (
        result.groupby(["id", "gid", "name"], as_index=False)
              .apply(lambda g: pd.Series({
                  "genre": agg_genres(g["genre"]),
                  "urls": agg_urls(g)
              }))
              .reset_index(drop=True)
    )

    return grouped

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
        load_model()
    if model is None:
        raise RuntimeError(
            "No model available. Train with: python -m ml.artist.scripts.train_local "
            "then upload: python -m ml.artist.scripts.upload_artist"
        )

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
