import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import pandas as pd
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


def _sql_placeholders(values: list[int]) -> str:
    return ",".join(["%s"] * len(values))


def _ordered_unique(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if pd.isna(value):
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _release_group_genres_query(where_clause: str = "") -> str:
    return f"""
        SELECT id, genre
        FROM (
            SELECT
                l_artist_release_group.entity0 AS id,
                genre.name AS genre
            FROM l_artist_release_group
            JOIN release_group_tag
                ON release_group_tag.release_group = l_artist_release_group.entity1
               AND release_group_tag.count > 0
            JOIN tag
                ON tag.id = release_group_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)

            UNION

            SELECT
                l_artist_release_group.entity0 AS id,
                genre.name AS genre
            FROM l_artist_release_group
            JOIN release_group
                ON release_group.id = l_artist_release_group.entity1
            JOIN release
                ON release.release_group = release_group.id
            JOIN release_tag
                ON release_tag.release = release.id
               AND release_tag.count > 0
            JOIN tag
                ON tag.id = release_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)
        ) release_group_genres
        {where_clause}
    """


def _fetch_release_group_genres(artist_ids: list[int], conn) -> pd.DataFrame:
    """Return genre names inferred from an artist's release groups."""
    if not artist_ids:
        return pd.DataFrame(columns=["id", "genre"])

    placeholders = _sql_placeholders(artist_ids)
    query = _release_group_genres_query(
        f"WHERE release_group_genres.id IN ({placeholders})"
    )

    return pd.read_sql_query(query, conn, params=artist_ids)


def fetch_artist_training_data(conn) -> pd.DataFrame:
    """
    Build artist features for KNN training without dropping artists that lack tags.

    The first query gets artist metadata and artist-level tags. The second query
    fills genre information from each artist's release groups, mirroring the
    album feature source.
    """
    artist_query = """
        SELECT
            artist.id AS artist_id,
            artist.gid AS artist_gid,
            artist.name AS artist_name,
            artist_type.name AS artist_type,
            area.name AS area_name,
            tag.name AS tag,
            artist_tag.count AS tag_count,
            genre.name AS genre
        FROM artist
        LEFT JOIN artist_type
            ON artist_type.id = artist.type
        LEFT JOIN area
            ON area.id = artist.area
        LEFT JOIN artist_tag
            ON artist_tag.artist = artist.id
           AND artist_tag.count > 0
        LEFT JOIN tag
            ON tag.id = artist_tag.tag
        LEFT JOIN genre
            ON LOWER(genre.name) = LOWER(tag.name)
        WHERE artist.name IS NOT NULL
          AND LOWER(artist.name) != 'various artists'
    """

    artist_rows = pd.read_sql_query(artist_query, conn)
    if artist_rows.empty:
        return pd.DataFrame(
            columns=[
                "artist_id",
                "artist_gid",
                "artist_name",
                "artist_type",
                "area_name",
                "tags",
                "genres",
                "tag_count_sum",
            ]
        )

    release_group_genres_query = _release_group_genres_query()
    release_group_genres = pd.read_sql_query(release_group_genres_query, conn)

    grouped = (
        artist_rows.groupby(
            ["artist_id", "artist_gid", "artist_name", "artist_type", "area_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            tags=("tag", lambda values: " ".join(_ordered_unique(values))),
            tag_genres=("genre", lambda values: " ".join(_ordered_unique(values))),
            tag_count_sum=("tag_count", lambda values: values.dropna().clip(lower=0).sum()),
        )
    )

    if release_group_genres.empty:
        grouped["release_group_genres"] = ""
    else:
        release_group_genres = (
            release_group_genres.groupby("id", as_index=False)["genre"]
            .agg(lambda values: " ".join(_ordered_unique(values)))
            .rename(columns={"id": "artist_id", "genre": "release_group_genres"})
        )
        grouped = grouped.merge(release_group_genres, on="artist_id", how="left")
        grouped["release_group_genres"] = grouped["release_group_genres"].fillna("")

    grouped["genres"] = (
        grouped["tag_genres"].fillna("")
        + " "
        + grouped["release_group_genres"].fillna("")
    ).str.split().str.join(" ")
    grouped = grouped.drop(columns=["tag_genres", "release_group_genres"])
    grouped["tag_count_sum"] = grouped["tag_count_sum"].fillna(0)

    return grouped


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

    placeholders = _sql_placeholders(artist_ids)

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
           AND artist_tag.count > 0
        LEFT JOIN tag
            ON artist_tag.tag = tag.id
        LEFT JOIN genre
            ON LOWER(tag.name) = LOWER(genre.name)
        LEFT JOIN l_artist_url
            ON l_artist_url.entity0 = artist.id
        LEFT JOIN url
            ON url.id = l_artist_url.entity1
        LEFT JOIN link
            ON link.id = l_artist_url.link
        LEFT JOIN link_type
            ON link_type.id = link.link_type
        WHERE artist.id IN ({placeholders})
    """

    result = pd.read_sql_query(query, conn, params=artist_ids)

    if result.empty:
        return pd.DataFrame(columns=["id", "gid", "name", "genre", "urls"])

    release_group_genres = _fetch_release_group_genres(artist_ids, conn)
    if release_group_genres.empty:
        fallback_genres_by_artist = {}
    else:
        fallback_genres_by_artist = (
            release_group_genres.groupby("id")["genre"]
            .apply(_ordered_unique)
            .to_dict()
        )

    def agg_genres(artist_id, series):
        genres = _ordered_unique(series)
        genres.extend(
            genre
            for genre in fallback_genres_by_artist.get(artist_id, [])
            if genre.lower() not in {existing.lower() for existing in genres}
        )
        return [genre.capitalize() for genre in genres]

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
                  "genre": agg_genres(int(g["id"].iloc[0]), g["genre"]),
                  "urls": agg_urls(g)
              }))
              .reset_index(drop=True)
    )

    return grouped


def _fetch_seed_genres_from_release_groups(
    missing_ids: list[int],
    conn,
) -> dict[int, str]:
    if conn is None or not missing_ids:
        return {}

    release_group_genres = _fetch_release_group_genres(missing_ids, conn)
    if release_group_genres.empty:
        return {}

    return (
        release_group_genres.groupby("id")["genre"]
        .agg(lambda values: " ".join(_ordered_unique(values)))
        .to_dict()
    )


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
    conn=None,
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
    fallback_genres_by_artist = _fetch_seed_genres_from_release_groups(missing_ids, conn)
    still_missing_ids = sorted(
        artist_id
        for artist_id in missing_ids
        if not fallback_genres_by_artist.get(artist_id, "").strip()
    )
    if still_missing_ids:
        raise RuntimeError(f"Unknown artist IDs: {still_missing_ids}")

    seed_genres = matches["genres"].astype(str).tolist()
    seed_genres.extend(
        fallback_genres_by_artist[artist_id]
        for artist_id in missing_ids
        if fallback_genres_by_artist.get(artist_id, "").strip()
    )

    query_vectors = vectorizer.transform(seed_genres)
    if len(seed_genres) == 1:
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


def predict_playlist(artist_ids: list[int], top_n: int = 5, conn=None) -> list[int]:
    if model is None:
        load_latest_model()
    if model is None:
        raise RuntimeError("No model saved yet. Train a model and call save_model(model).")

    if isinstance(model, dict):
        recommendations = _recommend_artist_ids_from_artifact(
            model,
            artist_ids,
            top_n,
            conn=conn,
        )
        if not recommendations:
            raise RuntimeError("No recommendation found for these artist IDs.")
        return recommendations

    pred_result = model.predict([artist_ids])[0]
    if hasattr(pred_result, "tolist"):
        pred_result = pred_result.tolist()
    if not isinstance(pred_result, list):
        pred_result = list(pred_result)
    return [int(artist_id) for artist_id in pred_result[:top_n]]
