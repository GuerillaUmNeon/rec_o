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


def _sql_values_placeholders(values: list[int]) -> str:
    return ",".join(["(%s)"] * len(values))


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


def _artist_genres_query(selected_artists: list[int] | None = None) -> str:
    has_artist_scope = selected_artists is not None
    selected_artists_cte = ""
    artist_credit_scope = ""
    artist_tag_scope = ""
    l_artist_genre_scope = ""
    l_artist_release_group_scope = ""
    l_artist_release_scope = ""
    l_artist_recording_scope = ""
    l_artist_work_scope = ""

    if has_artist_scope:
        values = _sql_values_placeholders(selected_artists)
        selected_artists_cte = f"""
        selected_artists(id) AS (
            VALUES {values}
        ),
        """
        artist_credit_scope = (
            "JOIN selected_artists ON selected_artists.id = artist_credit_name.artist"
        )
        artist_tag_scope = (
            "JOIN selected_artists ON selected_artists.id = artist_tag.artist"
        )
        l_artist_genre_scope = (
            "JOIN selected_artists ON selected_artists.id = l_artist_genre.entity0"
        )
        l_artist_release_group_scope = (
            "JOIN selected_artists ON selected_artists.id = l_artist_release_group.entity0"
        )
        l_artist_release_scope = (
            "JOIN selected_artists ON selected_artists.id = l_artist_release.entity0"
        )
        l_artist_recording_scope = (
            "JOIN selected_artists ON selected_artists.id = l_artist_recording.entity0"
        )
        l_artist_work_scope = (
            "JOIN selected_artists ON selected_artists.id = l_artist_work.entity0"
        )

    return f"""
        WITH
        {selected_artists_cte}
        credited_release_groups AS (
            SELECT
                artist_credit_name.artist AS id,
                release_group.id AS release_group_id
            FROM artist_credit_name
            {artist_credit_scope}
            JOIN release_group
                ON release_group.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                artist_credit_name.artist AS id,
                release.release_group AS release_group_id
            FROM artist_credit_name
            {artist_credit_scope}
            JOIN release
                ON release.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                l_artist_release_group.entity0 AS id,
                l_artist_release_group.entity1 AS release_group_id
            FROM l_artist_release_group
            {l_artist_release_group_scope}

            UNION

            SELECT
                l_artist_release.entity0 AS id,
                release.release_group AS release_group_id
            FROM l_artist_release
            {l_artist_release_scope}
            JOIN release
                ON release.id = l_artist_release.entity1
        ),
        credited_releases AS (
            SELECT
                artist_credit_name.artist AS id,
                release.id AS release_id
            FROM artist_credit_name
            {artist_credit_scope}
            JOIN release
                ON release.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                credited_release_groups.id AS id,
                release.id AS release_id
            FROM credited_release_groups
            JOIN release
                ON release.release_group = credited_release_groups.release_group_id

            UNION

            SELECT
                l_artist_release.entity0 AS id,
                l_artist_release.entity1 AS release_id
            FROM l_artist_release
            {l_artist_release_scope}
        ),
        credited_recordings AS (
            SELECT
                artist_credit_name.artist AS id,
                recording.id AS recording_id
            FROM artist_credit_name
            {artist_credit_scope}
            JOIN recording
                ON recording.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                l_artist_recording.entity0 AS id,
                l_artist_recording.entity1 AS recording_id
            FROM l_artist_recording
            {l_artist_recording_scope}
        ),
        credited_works AS (
            SELECT
                l_artist_work.entity0 AS id,
                l_artist_work.entity1 AS work_id
            FROM l_artist_work
            {l_artist_work_scope}
        ),
        primary_artist_genres AS (
            SELECT
                artist_tag.artist AS id,
                genre.name AS genre
            FROM artist_tag
            {artist_tag_scope}
            JOIN tag
                ON tag.id = artist_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)
            WHERE artist_tag.count > 0

            UNION

            SELECT
                l_artist_genre.entity0 AS id,
                genre.name AS genre
            FROM l_artist_genre
            {l_artist_genre_scope}
            JOIN genre
                ON genre.id = l_artist_genre.entity1

            UNION

            SELECT
                credited_release_groups.id AS id,
                genre.name AS genre
            FROM credited_release_groups
            JOIN release_group_tag
                ON release_group_tag.release_group = credited_release_groups.release_group_id
               AND release_group_tag.count > 0
            JOIN tag
                ON tag.id = release_group_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)

            UNION

            SELECT
                credited_releases.id AS id,
                genre.name AS genre
            FROM credited_releases
            JOIN release_tag
                ON release_tag.release = credited_releases.release_id
               AND release_tag.count > 0
            JOIN tag
                ON tag.id = release_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)
        ),
        fallback_artist_genres AS (
            SELECT
                credited_recordings.id AS id,
                genre.name AS genre
            FROM credited_recordings
            JOIN recording_tag
                ON recording_tag.recording = credited_recordings.recording_id
               AND recording_tag.count > 0
            JOIN tag
                ON tag.id = recording_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)

            UNION

            SELECT
                credited_works.id AS id,
                genre.name AS genre
            FROM credited_works
            JOIN work_tag
                ON work_tag.work = credited_works.work_id
               AND work_tag.count > 0
            JOIN tag
                ON tag.id = work_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)
        )
        SELECT id, genre
        FROM primary_artist_genres

        UNION

        SELECT fallback_artist_genres.id, fallback_artist_genres.genre
        FROM fallback_artist_genres
        WHERE NOT EXISTS (
            SELECT 1
            FROM primary_artist_genres
            WHERE primary_artist_genres.id = fallback_artist_genres.id
        )
    """


def _primary_artist_genres_query(artist_ids: list[int]) -> str:
    selected_artists = _sql_values_placeholders(artist_ids)
    return f"""
        WITH
        selected_artists(id) AS (
            VALUES {selected_artists}
        ),
        credited_release_groups AS (
            SELECT
                artist_credit_name.artist AS id,
                release_group.id AS release_group_id
            FROM artist_credit_name
            JOIN selected_artists
                ON selected_artists.id = artist_credit_name.artist
            JOIN release_group
                ON release_group.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                artist_credit_name.artist AS id,
                release.release_group AS release_group_id
            FROM artist_credit_name
            JOIN selected_artists
                ON selected_artists.id = artist_credit_name.artist
            JOIN release
                ON release.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                l_artist_release_group.entity0 AS id,
                l_artist_release_group.entity1 AS release_group_id
            FROM l_artist_release_group
            JOIN selected_artists
                ON selected_artists.id = l_artist_release_group.entity0

            UNION

            SELECT
                l_artist_release.entity0 AS id,
                release.release_group AS release_group_id
            FROM l_artist_release
            JOIN selected_artists
                ON selected_artists.id = l_artist_release.entity0
            JOIN release
                ON release.id = l_artist_release.entity1
        ),
        credited_releases AS (
            SELECT
                artist_credit_name.artist AS id,
                release.id AS release_id
            FROM artist_credit_name
            JOIN selected_artists
                ON selected_artists.id = artist_credit_name.artist
            JOIN release
                ON release.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                credited_release_groups.id AS id,
                release.id AS release_id
            FROM credited_release_groups
            JOIN release
                ON release.release_group = credited_release_groups.release_group_id

            UNION

            SELECT
                l_artist_release.entity0 AS id,
                l_artist_release.entity1 AS release_id
            FROM l_artist_release
            JOIN selected_artists
                ON selected_artists.id = l_artist_release.entity0
        )
        SELECT id, genre
        FROM (
            SELECT
                artist_tag.artist AS id,
                genre.name AS genre
            FROM artist_tag
            JOIN selected_artists
                ON selected_artists.id = artist_tag.artist
            JOIN tag
                ON tag.id = artist_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)
            WHERE artist_tag.count > 0

            UNION

            SELECT
                l_artist_genre.entity0 AS id,
                genre.name AS genre
            FROM l_artist_genre
            JOIN selected_artists
                ON selected_artists.id = l_artist_genre.entity0
            JOIN genre
                ON genre.id = l_artist_genre.entity1

            UNION

            SELECT
                credited_release_groups.id AS id,
                genre.name AS genre
            FROM credited_release_groups
            JOIN release_group_tag
                ON release_group_tag.release_group = credited_release_groups.release_group_id
               AND release_group_tag.count > 0
            JOIN tag
                ON tag.id = release_group_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)

            UNION

            SELECT
                credited_releases.id AS id,
                genre.name AS genre
            FROM credited_releases
            JOIN release_tag
                ON release_tag.release = credited_releases.release_id
               AND release_tag.count > 0
            JOIN tag
                ON tag.id = release_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)
        ) primary_artist_genres
    """


def _fallback_artist_genres_query(artist_ids: list[int]) -> str:
    selected_artists = _sql_values_placeholders(artist_ids)
    return f"""
        WITH
        selected_artists(id) AS (
            VALUES {selected_artists}
        ),
        credited_recordings AS (
            SELECT
                artist_credit_name.artist AS id,
                recording.id AS recording_id
            FROM artist_credit_name
            JOIN selected_artists
                ON selected_artists.id = artist_credit_name.artist
            JOIN recording
                ON recording.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                l_artist_recording.entity0 AS id,
                l_artist_recording.entity1 AS recording_id
            FROM l_artist_recording
            JOIN selected_artists
                ON selected_artists.id = l_artist_recording.entity0
        ),
        credited_works AS (
            SELECT
                l_artist_work.entity0 AS id,
                l_artist_work.entity1 AS work_id
            FROM l_artist_work
            JOIN selected_artists
                ON selected_artists.id = l_artist_work.entity0
        )
        SELECT id, genre
        FROM (
            SELECT
                credited_recordings.id AS id,
                genre.name AS genre
            FROM credited_recordings
            JOIN recording_tag
                ON recording_tag.recording = credited_recordings.recording_id
               AND recording_tag.count > 0
            JOIN tag
                ON tag.id = recording_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)

            UNION

            SELECT
                credited_works.id AS id,
                genre.name AS genre
            FROM credited_works
            JOIN work_tag
                ON work_tag.work = credited_works.work_id
               AND work_tag.count > 0
            JOIN tag
                ON tag.id = work_tag.tag
            JOIN genre
                ON LOWER(genre.name) = LOWER(tag.name)
        ) fallback_artist_genres
    """


def _fetch_artist_genres(artist_ids: list[int], conn) -> pd.DataFrame:
    """Return genre names, using recording/work tags only as a last resort."""
    if not artist_ids:
        return pd.DataFrame(columns=["id", "genre"])

    primary_genres = pd.read_sql_query(
        _primary_artist_genres_query(artist_ids),
        conn,
        params=artist_ids,
    )
    found_ids = set(primary_genres["id"].astype(int).tolist())
    missing_ids = [
        artist_id
        for artist_id in artist_ids
        if int(artist_id) not in found_ids
    ]
    if not missing_ids:
        return primary_genres

    fallback_genres = pd.read_sql_query(
        _fallback_artist_genres_query(missing_ids),
        conn,
        params=missing_ids,
    )
    if fallback_genres.empty:
        return primary_genres

    return pd.concat([primary_genres, fallback_genres], ignore_index=True)


def fetch_artist_training_data(conn) -> pd.DataFrame:
    """
    Build artist features for KNN training without dropping artists that lack tags.

    The first query gets artist metadata and artist-level tags. The second query
    keeps every artist that has a genre from artist, release group, or release
    tags.
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

    artist_genres_query = _artist_genres_query()
    artist_genres = pd.read_sql_query(artist_genres_query, conn)

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

    if artist_genres.empty:
        grouped["all_genres"] = ""
    else:
        artist_genres = (
            artist_genres.groupby("id", as_index=False)["genre"]
            .agg(lambda values: " ".join(_ordered_unique(values)))
            .rename(columns={"id": "artist_id", "genre": "all_genres"})
        )
        grouped = grouped.merge(artist_genres, on="artist_id", how="left")
        grouped["all_genres"] = grouped["all_genres"].fillna("")

    grouped["genres"] = (
        grouped["tag_genres"].fillna("")
        + " "
        + grouped["all_genres"].fillna("")
    ).str.split().str.join(" ")
    grouped = grouped[grouped["genres"].str.strip() != ""].copy()
    grouped = grouped.drop(columns=["tag_genres", "all_genres"])
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

    artist_genres = _fetch_artist_genres(artist_ids, conn)
    if artist_genres.empty:
        fallback_genres_by_artist = {}
    else:
        fallback_genres_by_artist = (
            artist_genres.groupby("id")["genre"]
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
                  "genre": agg_genres(int(g.name[0]), g["genre"]),
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

    artist_genres = _fetch_artist_genres(missing_ids, conn)
    if artist_genres.empty:
        return {}

    return (
        artist_genres.groupby("id")["genre"]
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
