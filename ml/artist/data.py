"""
Fetch artist KNN training features from MusicBrainz.

Genre tokens aligned with app inference (predictor.py).
"""

import os
import time
from pathlib import Path

import pandas as pd

from ml.artist.config import (
    ARTIST_ML_GENRE_CHUNK_SIZE,
    ARTIST_ML_MAX_ARTISTS,
    ARTIST_TRAINING_FEATURES_CACHE,
    ML_OUTPUTS_DIR,
)
from ml.artist.features import join_artist_genre_feature_tokens, ordered_unique


def _artist_training_cache_path() -> Path:
    """Prefer artist cache; fall back to legacy training_features.pkl if present."""
    legacy = ML_OUTPUTS_DIR / "training_features.pkl"
    if ARTIST_TRAINING_FEATURES_CACHE.is_file():
        return ARTIST_TRAINING_FEATURES_CACHE
    if legacy.is_file():
        return legacy
    return ARTIST_TRAINING_FEATURES_CACHE

def _sql_values_placeholders(values: list[int]) -> str:
    return ",".join(["(%s)"] * len(values))


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


_SCOPED_ARTIST_JOINS = """
        selected_artists AS (
            SELECT unnest(%s::int[]) AS id
        ),
        credited_release_groups AS (
            SELECT
                artist_credit_name.artist AS id,
                release_group.id AS release_group_id
            FROM artist_credit_name
            JOIN selected_artists ON selected_artists.id = artist_credit_name.artist
            JOIN release_group
                ON release_group.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                artist_credit_name.artist AS id,
                release.release_group AS release_group_id
            FROM artist_credit_name
            JOIN selected_artists ON selected_artists.id = artist_credit_name.artist
            JOIN release
                ON release.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                l_artist_release_group.entity0 AS id,
                l_artist_release_group.entity1 AS release_group_id
            FROM l_artist_release_group
            JOIN selected_artists ON selected_artists.id = l_artist_release_group.entity0

            UNION

            SELECT
                l_artist_release.entity0 AS id,
                release.release_group AS release_group_id
            FROM l_artist_release
            JOIN selected_artists ON selected_artists.id = l_artist_release.entity0
            JOIN release
                ON release.id = l_artist_release.entity1
        ),
        credited_releases AS (
            SELECT
                artist_credit_name.artist AS id,
                release.id AS release_id
            FROM artist_credit_name
            JOIN selected_artists ON selected_artists.id = artist_credit_name.artist
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
            JOIN selected_artists ON selected_artists.id = l_artist_release.entity0
        ),
        credited_recordings AS (
            SELECT
                artist_credit_name.artist AS id,
                recording.id AS recording_id
            FROM artist_credit_name
            JOIN selected_artists ON selected_artists.id = artist_credit_name.artist
            JOIN recording
                ON recording.artist_credit = artist_credit_name.artist_credit

            UNION

            SELECT
                l_artist_recording.entity0 AS id,
                l_artist_recording.entity1 AS recording_id
            FROM l_artist_recording
            JOIN selected_artists ON selected_artists.id = l_artist_recording.entity0
        ),
        credited_works AS (
            SELECT
                l_artist_work.entity0 AS id,
                l_artist_work.entity1 AS work_id
            FROM l_artist_work
            JOIN selected_artists ON selected_artists.id = l_artist_work.entity0
        ),
"""


def _artist_genres_query_training() -> str:
    """
    Lighter genre query for ML training when artist IDs are already scoped.

    - unnest(%s::int[]) instead of thousands of VALUES rows
    - skips artist_tag / l_artist_genre (already loaded in artist_query)
    - UNION ALL + DISTINCT instead of correlated NOT EXISTS
    """
    return (
        _SCOPED_ARTIST_JOINS
        + """
        release_genres AS (
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

            UNION ALL

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
        recording_work_genres AS (
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

            UNION ALL

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
        SELECT DISTINCT id, genre
        FROM (
            SELECT id, genre FROM release_genres
            UNION ALL
            SELECT id, genre FROM recording_work_genres
        ) all_genres
    """
    )


def _fetch_extended_artist_genres(
    conn,
    artist_ids: list[int],
    *,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    if not artist_ids:
        return pd.DataFrame(columns=["id", "genre"])

    chunk_size = chunk_size or ARTIST_ML_GENRE_CHUNK_SIZE
    query = _artist_genres_query_training()
    chunks: list[pd.DataFrame] = []
    total = len(artist_ids)
    n_batches = (total + chunk_size - 1) // chunk_size

    print(
        f"Fetching extended genres for {total:,} artists "
        f"({n_batches} batch(es) of {chunk_size:,})..."
    )
    t0 = time.perf_counter()

    for batch_idx, start in enumerate(range(0, total, chunk_size), start=1):
        batch_ids = artist_ids[start : start + chunk_size]
        batch_t0 = time.perf_counter()
        batch_df = pd.read_sql_query(query, conn, params=[batch_ids])
        chunks.append(batch_df)
        print(
            f"  batch {batch_idx}/{n_batches}: {len(batch_ids):,} artists, "
            f"{len(batch_df):,} genre rows in {time.perf_counter() - batch_t0:.1f}s"
        )

    artist_genres = pd.concat(chunks, ignore_index=True)
    print(f"Extended genres done in {time.perf_counter() - t0:.1f}s total")
    return artist_genres


def _resolve_max_artists(max_artists: int | None) -> int | None:
    if max_artists is not None:
        return max_artists
    env_limit = os.getenv("ARTIST_ML_MAX_ARTISTS") or ARTIST_ML_MAX_ARTISTS
    if not env_limit:
        return None
    return int(env_limit)


def _artist_scope_sql(max_artists: int | None) -> tuple[str, list]:
    if not max_artists:
        return "", []
    return (
        """
          AND artist.id IN (
              SELECT id
              FROM artist
              WHERE name IS NOT NULL
                AND LOWER(name) != 'various artists'
              ORDER BY id
              LIMIT %s
          )
        """,
        [max_artists],
    )


def fetch_artist_knn_training_data_scoped(
    conn,
    *,
    max_artists: int | None = None,
    skip_extended_genres: bool = False,
    use_cache: bool = False,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Build artist features for KNN training without dropping artists that lack tags.

    The first query gets artist metadata and artist-level tags. The second query
    keeps every artist that has a genre from artist, release group, or release
    tags (skipped when skip_extended_genres=True).

    Use max_artists or ARTIST_ML_MAX_ARTISTS to avoid scanning the full MusicBrainz DB.
    """
    max_artists = _resolve_max_artists(max_artists)

    if use_cache and not refresh_cache and _artist_training_cache_path().is_file():
        print(f"Loading cached training data from {_artist_training_cache_path()}")
        return pd.read_pickle(_artist_training_cache_path())

    scope_sql, scope_params = _artist_scope_sql(max_artists)
    artist_query = f"""
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
        {scope_sql}
    """

    if max_artists:
        print(f"Fetching artist tags for at most {max_artists:,} artists...")
    artist_rows = pd.read_sql_query(artist_query, conn, params=scope_params or None)
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

    if skip_extended_genres:
        artist_genres = pd.DataFrame(columns=["id", "genre"])
    elif max_artists:
        artist_ids = artist_rows["artist_id"].drop_duplicates().astype(int).tolist()
        artist_genres = _fetch_extended_artist_genres(conn, artist_ids)
    else:
        print("Fetching extended genres for all artists (slow)...")
        artist_genres_query = _artist_genres_query()
        artist_genres = pd.read_sql_query(artist_genres_query, conn)

    grouped = (
        artist_rows.groupby(
            ["artist_id", "artist_gid", "artist_name", "artist_type", "area_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            tags=("tag", lambda values: " ".join(ordered_unique(values))),
            tag_genres=("genre", join_artist_genre_feature_tokens),
            tag_count_sum=("tag_count", lambda values: values.dropna().clip(lower=0).sum()),
        )
    )

    if artist_genres.empty:
        grouped["all_genres"] = ""
    else:
        artist_genres = (
            artist_genres.groupby("id", as_index=False)["genre"]
            .agg(join_artist_genre_feature_tokens)
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

    if not (use_cache and _artist_training_cache_path().is_file() and not refresh_cache):
        _artist_training_cache_path().parent.mkdir(parents=True, exist_ok=True)
        grouped.to_pickle(_artist_training_cache_path())
        print(f"Cached training features to {_artist_training_cache_path()}")

    return grouped


def fetch_artist_knn_training_data(
    conn,
    *,
    use_cache: bool = False,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Build genre-only artist features for the KNN recommender (temp tables).

    Full DB scan via temp tables (offline training only).
    """
    if use_cache and not refresh_cache and _artist_training_cache_path().is_file():
        print(f"Loading cached training data from {_artist_training_cache_path()}")
        return pd.read_pickle(_artist_training_cache_path())

    setup_steps = [
        """
        DROP TABLE IF EXISTS rec_o_credited_release_groups;
        """,
        """
        CREATE TEMP TABLE rec_o_credited_release_groups ON COMMIT DROP AS
        SELECT artist_credit_name.artist AS id, release_group.id AS release_group_id
        FROM artist_credit_name
        JOIN release_group
            ON release_group.artist_credit = artist_credit_name.artist_credit

        UNION

        SELECT artist_credit_name.artist AS id, release.release_group AS release_group_id
        FROM artist_credit_name
        JOIN release
            ON release.artist_credit = artist_credit_name.artist_credit

        UNION

        SELECT l_artist_release_group.entity0 AS id, l_artist_release_group.entity1 AS release_group_id
        FROM l_artist_release_group

        UNION

        SELECT l_artist_release.entity0 AS id, release.release_group AS release_group_id
        FROM l_artist_release
        JOIN release
            ON release.id = l_artist_release.entity1;
        """,
        """
        CREATE INDEX rec_o_credited_release_groups_id_idx
            ON rec_o_credited_release_groups(id);
        """,
        """
        CREATE INDEX rec_o_credited_release_groups_release_group_idx
            ON rec_o_credited_release_groups(release_group_id);
        """,
        """
        ANALYZE rec_o_credited_release_groups;
        """,
        """
        DROP TABLE IF EXISTS rec_o_credited_releases;
        """,
        """
        CREATE TEMP TABLE rec_o_credited_releases ON COMMIT DROP AS
        SELECT artist_credit_name.artist AS id, release.id AS release_id
        FROM artist_credit_name
        JOIN release
            ON release.artist_credit = artist_credit_name.artist_credit

        UNION

        SELECT rec_o_credited_release_groups.id AS id, release.id AS release_id
        FROM rec_o_credited_release_groups
        JOIN release
            ON release.release_group = rec_o_credited_release_groups.release_group_id

        UNION

        SELECT l_artist_release.entity0 AS id, l_artist_release.entity1 AS release_id
        FROM l_artist_release;
        """,
        """
        CREATE INDEX rec_o_credited_releases_id_idx
            ON rec_o_credited_releases(id);
        """,
        """
        CREATE INDEX rec_o_credited_releases_release_idx
            ON rec_o_credited_releases(release_id);
        """,
        """
        ANALYZE rec_o_credited_releases;
        """,
        """
        DROP TABLE IF EXISTS rec_o_primary_artist_genres;
        """,
        """
        CREATE TEMP TABLE rec_o_primary_artist_genres ON COMMIT DROP AS
        SELECT artist_tag.artist AS id, genre.name AS genre
        FROM artist_tag
        JOIN tag
            ON tag.id = artist_tag.tag
        JOIN genre
            ON LOWER(genre.name) = LOWER(tag.name)
        WHERE artist_tag.count > 0

        UNION

        SELECT l_artist_genre.entity0 AS id, genre.name AS genre
        FROM l_artist_genre
        JOIN genre
            ON genre.id = l_artist_genre.entity1

        UNION

        SELECT rec_o_credited_release_groups.id AS id, genre.name AS genre
        FROM rec_o_credited_release_groups
        JOIN release_group_tag
            ON release_group_tag.release_group = rec_o_credited_release_groups.release_group_id
           AND release_group_tag.count > 0
        JOIN tag
            ON tag.id = release_group_tag.tag
        JOIN genre
            ON LOWER(genre.name) = LOWER(tag.name)

        UNION

        SELECT rec_o_credited_releases.id AS id, genre.name AS genre
        FROM rec_o_credited_releases
        JOIN release_tag
            ON release_tag.release = rec_o_credited_releases.release_id
           AND release_tag.count > 0
        JOIN tag
            ON tag.id = release_tag.tag
        JOIN genre
            ON LOWER(genre.name) = LOWER(tag.name);
        """,
        """
        CREATE INDEX rec_o_primary_artist_genres_id_idx
            ON rec_o_primary_artist_genres(id);
        """,
        """
        ANALYZE rec_o_primary_artist_genres;
        """,
        """
        DROP TABLE IF EXISTS rec_o_artists_without_primary_genres;
        """,
        """
        CREATE TEMP TABLE rec_o_artists_without_primary_genres ON COMMIT DROP AS
        SELECT artist.id
        FROM artist
        WHERE artist.name IS NOT NULL
          AND LOWER(artist.name) != 'various artists'
          AND NOT EXISTS (
              SELECT 1
              FROM rec_o_primary_artist_genres
              WHERE rec_o_primary_artist_genres.id = artist.id
          );
        """,
        """
        CREATE INDEX rec_o_artists_without_primary_genres_id_idx
            ON rec_o_artists_without_primary_genres(id);
        """,
        """
        ANALYZE rec_o_artists_without_primary_genres;
        """,
        """
        DROP TABLE IF EXISTS rec_o_credited_recordings;
        """,
        """
        CREATE TEMP TABLE rec_o_credited_recordings ON COMMIT DROP AS
        SELECT artist_credit_name.artist AS id, recording.id AS recording_id
        FROM artist_credit_name
        JOIN rec_o_artists_without_primary_genres
            ON rec_o_artists_without_primary_genres.id = artist_credit_name.artist
        JOIN recording
            ON recording.artist_credit = artist_credit_name.artist_credit

        UNION

        SELECT l_artist_recording.entity0 AS id, l_artist_recording.entity1 AS recording_id
        FROM l_artist_recording
        JOIN rec_o_artists_without_primary_genres
            ON rec_o_artists_without_primary_genres.id = l_artist_recording.entity0;
        """,
        """
        CREATE INDEX rec_o_credited_recordings_id_idx
            ON rec_o_credited_recordings(id);
        """,
        """
        CREATE INDEX rec_o_credited_recordings_recording_idx
            ON rec_o_credited_recordings(recording_id);
        """,
        """
        ANALYZE rec_o_credited_recordings;
        """,
        """
        DROP TABLE IF EXISTS rec_o_artist_training_genres;
        """,
        """
        CREATE TEMP TABLE rec_o_artist_training_genres ON COMMIT DROP AS
        SELECT id, genre
        FROM rec_o_primary_artist_genres

        UNION

        SELECT rec_o_credited_recordings.id AS id, genre.name AS genre
        FROM rec_o_credited_recordings
        JOIN recording_tag
            ON recording_tag.recording = rec_o_credited_recordings.recording_id
           AND recording_tag.count > 0
        JOIN tag
            ON tag.id = recording_tag.tag
        JOIN genre
            ON LOWER(genre.name) = LOWER(tag.name);
        """,
        """
        CREATE INDEX rec_o_artist_training_genres_id_idx
            ON rec_o_artist_training_genres(id);
        """,
        """
        ANALYZE rec_o_artist_training_genres;
        """,
    ]

    print("Building recommender training temp tables (full DB)...")
    t0 = time.perf_counter()
    for sql in setup_steps:
        conn.execute(sql)

    query = """
        SELECT
            artist.id AS artist_id,
            artist.gid AS artist_gid,
            artist.name AS artist_name,
            artist_type.name AS artist_type,
            area.name AS area_name,
            ''::text AS tags,
            ARRAY_AGG(DISTINCT rec_o_artist_training_genres.genre ORDER BY rec_o_artist_training_genres.genre) AS genre_names,
            0::bigint AS tag_count_sum
        FROM rec_o_artist_training_genres
        JOIN artist
            ON artist.id = rec_o_artist_training_genres.id
        LEFT JOIN artist_type
            ON artist_type.id = artist.type
        LEFT JOIN area
            ON area.id = artist.area
        WHERE artist.name IS NOT NULL
          AND LOWER(artist.name) != 'various artists'
        GROUP BY
            artist.id,
            artist.gid,
            artist.name,
            artist_type.name,
            area.name
    """
    grouped = pd.read_sql_query(query, conn)
    print(f"Recommender training query done in {time.perf_counter() - t0:.1f}s")

    if grouped.empty:
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

    grouped["genres"] = grouped.pop("genre_names").apply(join_artist_genre_feature_tokens)
    grouped = grouped[grouped["genres"].str.strip() != ""].copy()

    _artist_training_cache_path().parent.mkdir(parents=True, exist_ok=True)
    grouped.to_pickle(_artist_training_cache_path())
    print(f"Cached training features to {_artist_training_cache_path()}")

    return grouped


# Backward-compatible aliases
fetch_artist_training_data = fetch_artist_knn_training_data_scoped
fetch_artist_recommender_training_data = fetch_artist_knn_training_data
