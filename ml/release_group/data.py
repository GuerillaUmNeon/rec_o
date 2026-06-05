"""Fetch and prepare release group KNN training features from MusicBrainz."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from ml.release_group.config import (
    LENGTH_MS_FOR_ALBUM,
    LENGTH_MS_FOR_SINGLE,
    RELEASE_GROUP_ML_MAX_ROWS,
    RELEASE_GROUP_ML_TRACK_META_CHUNK_SIZE,
    RELEASE_GROUP_TRAINING_FEATURES_CACHE,
    TRACKS_FOR_ALBUM,
)

RELEASE_GROUP_FEATURES_QUERY = """
WITH all_tag_links AS (
    SELECT
        r.release_group,
        rt.tag AS tag_id
    FROM release r
    JOIN release_tag rt
        ON rt.release = r.id

    UNION ALL

    SELECT
        rgt.release_group,
        rgt.tag AS tag_id
    FROM release_group_tag rgt
),
tag_classified AS (
    SELECT
        atl.release_group,
        atl.tag_id,
        g.id AS genre_id
    FROM all_tag_links atl
    JOIN tag t
        ON t.id = atl.tag_id
    LEFT JOIN genre g
        ON g.name = t.name
),
tag_counts AS (
    SELECT
        release_group,
        tag_id,
        COUNT(*) AS tag_count
    FROM tag_classified
    WHERE genre_id IS NULL
    GROUP BY release_group, tag_id
),
genre_counts AS (
    SELECT
        release_group,
        genre_id,
        COUNT(*) AS genre_count
    FROM tag_classified
    WHERE genre_id IS NOT NULL
    GROUP BY release_group, genre_id
),
tag_buckets AS (
    SELECT
        release_group,
        array_agg(DISTINCT tag_id ORDER BY tag_id) AS tag_ids
    FROM tag_counts
    WHERE tag_count > 1
    GROUP BY release_group
),
genre_buckets AS (
    SELECT
        release_group,
        array_agg(DISTINCT genre_id ORDER BY genre_id) AS genre_ids
    FROM genre_counts
    WHERE genre_count > 0
    GROUP BY release_group
),
release_meta AS (
    SELECT
        r.release_group,
        mode() WITHIN GROUP (ORDER BY r.status) AS status,
        mode() WITHIN GROUP (ORDER BY r.language) AS language,
        mode() WITHIN GROUP (ORDER BY r.script) AS script
    FROM release r
    GROUP BY r.release_group
),
secondary_types AS (
    SELECT
        release_group,
        array_agg(DISTINCT secondary_type ORDER BY secondary_type) AS secondary_type_ids
    FROM release_group_secondary_type_join
    GROUP BY release_group
)
SELECT
    rg.id,
    rg.type,
    rgm.first_release_date_year AS year,
    COALESCE(tb.tag_ids, ARRAY[]::integer[]) AS tag_ids,
    COALESCE(gb.genre_ids, ARRAY[]::integer[]) AS genre_ids,
    rm.status,
    rm.language,
    rm.script,
    COALESCE(st.secondary_type_ids, ARRAY[]::integer[]) AS secondary_type_ids
FROM release_group rg
LEFT JOIN release_group_meta rgm
    ON rgm.id = rg.id
LEFT JOIN tag_buckets tb
    ON tb.release_group = rg.id
LEFT JOIN genre_buckets gb
    ON gb.release_group = rg.id
LEFT JOIN release_meta rm
    ON rm.release_group = rg.id
LEFT JOIN secondary_types st
    ON st.release_group = rg.id
{scope_sql}
"""

TRACK_META_QUERY = """
WITH release_groups_filter AS (
    SELECT rid
    FROM unnest(%s::int[]) AS t(rid)
),
ranked_releases AS (
    SELECT
        r.release_group,
        r.id AS release_id,
        row_number() OVER (
            PARTITION BY r.release_group
            ORDER BY
                CASE
                    WHEN lower(r.name) LIKE '%%super deluxe%%' THEN 1
                    WHEN lower(r.name) LIKE '%%deluxe%%' THEN 1
                    WHEN lower(r.name) LIKE '%%expanded%%' THEN 1
                    WHEN lower(r.name) LIKE '%%anniversary%%' THEN 1
                    WHEN lower(r.name) LIKE '%%bonus%%' THEN 1
                    WHEN lower(r.name) LIKE '%%collector%%' THEN 1
                    ELSE 0
                END,
                COALESCE(rc.date_year, ruc.date_year) NULLS LAST,
                COALESCE(rc.date_month, ruc.date_month) NULLS LAST,
                COALESCE(rc.date_day, ruc.date_day) NULLS LAST,
                r.id
        ) AS rn
    FROM release_groups_filter rgf
    JOIN release r
        ON r.release_group = rgf.rid
    LEFT JOIN release_country rc
        ON rc.release = r.id
    LEFT JOIN release_unknown_country ruc
        ON ruc.release = r.id
        AND rc.release IS NULL
),
chosen_release AS (
    SELECT release_group, release_id
    FROM ranked_releases
    WHERE rn = 1
),
release_track_meta AS (
    SELECT
        cr.release_group,
        COUNT(t.id) AS track_numbers,
        SUM(t.length) AS album_length_ms
    FROM chosen_release cr
    JOIN medium m
        ON m.release = cr.release_id
    JOIN track t
        ON t.medium = m.id
    GROUP BY cr.release_group
)
SELECT
    release_group AS id,
    track_numbers,
    album_length_ms
FROM release_track_meta
"""


def _resolve_max_rows(max_rows: int | None) -> int | None:
    if max_rows is not None:
        return max_rows
    env_limit = os.getenv("RELEASE_GROUP_ML_MAX_ROWS") or RELEASE_GROUP_ML_MAX_ROWS
    if not env_limit:
        return None
    return int(env_limit)


def _scope_sql(max_rows: int | None) -> tuple[str, list]:
    if not max_rows:
        return "", []
    return (
        """
WHERE rg.id IN (
    SELECT id
    FROM release_group
    ORDER BY id
    LIMIT %s
)
""",
        [max_rows],
    )


def _training_cache_path() -> Path:
    legacy = RELEASE_GROUP_TRAINING_FEATURES_CACHE.parent / "rg_training_features.pkl"
    if RELEASE_GROUP_TRAINING_FEATURES_CACHE.is_file():
        return RELEASE_GROUP_TRAINING_FEATURES_CACHE
    if legacy.is_file():
        return legacy
    return RELEASE_GROUP_TRAINING_FEATURES_CACHE


def _normalize_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("tag_ids", "genre_ids", "secondary_type_ids"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))
    return df


def _classify_type(row) -> object:
    tn = row["track_numbers"]
    lm = row["album_length_ms"]

    if tn is None or lm is None or pd.isna(tn) or pd.isna(lm):
        return pd.NA

    tn = int(tn)
    lm = int(lm)

    if tn > TRACKS_FOR_ALBUM or lm > LENGTH_MS_FOR_ALBUM:
        return 1
    if tn == 1 and lm < LENGTH_MS_FOR_SINGLE:
        return 2
    return 3


def _infer_missing_types(conn, data: pd.DataFrame) -> pd.DataFrame:
    null_type = data[data["type"].isna()].copy()
    if null_type.empty:
        return data

    release_group_ids = null_type["id"].astype(int).tolist()
    chunks: list[pd.DataFrame] = []
    chunk_size = RELEASE_GROUP_ML_TRACK_META_CHUNK_SIZE

    t0 = time.perf_counter()
    for start in range(0, len(release_group_ids), chunk_size):
        batch = release_group_ids[start : start + chunk_size]
        batch_df = pd.read_sql_query(TRACK_META_QUERY, conn, params=(batch,))
        chunks.append(batch_df)
        print(
            f"Track meta batch {start // chunk_size + 1}: "
            f"{len(batch)} release groups ({time.perf_counter() - t0:.1f}s)"
        )

    track_meta = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["id", "track_numbers", "album_length_ms"]
    )

    null_type = null_type.merge(track_meta, on="id", how="left")
    null_type["track_numbers"] = null_type["track_numbers"].astype("Int32")
    null_type["album_length_ms"] = null_type["album_length_ms"].astype("Int64")
    null_type["type"] = null_type.apply(_classify_type, axis=1)

    computed_type_map = null_type.set_index("id")["type"]
    data = data.copy()
    data["type"] = data["type"].fillna(data["id"].map(computed_type_map)).astype("Int32")
    return data


def fetch_release_group_knn_training_data(
    conn,
    *,
    max_rows: int | None = None,
    skip_type_inference: bool = False,
    use_cache: bool = False,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """
    Build release group features for KNN training.

    Port of models/note_book_guillaume.ipynb:
    main tag/genre SQL + optional track-meta inference for missing `type`.
    """
    max_rows = _resolve_max_rows(max_rows)
    cache_path = _training_cache_path()

    if use_cache and not refresh_cache and cache_path.is_file():
        print(f"Loading cached training data from {cache_path}")
        return pd.read_pickle(cache_path)

    scope_sql, scope_params = _scope_sql(max_rows)
    query = RELEASE_GROUP_FEATURES_QUERY.format(scope_sql=scope_sql)

    if max_rows:
        print(f"Fetching release group features for at most {max_rows:,} rows...")
    t0 = time.perf_counter()
    df = pd.read_sql_query(query, conn, params=scope_params or None)
    print(f"Main SQL done in {time.perf_counter() - t0:.1f}s — {len(df):,} rows")

    df = df.astype({
        "id": "int64",
        "type": "Int32",
        "year": "Int32",
        "status": "Int32",
        "language": "Int32",
        "script": "Int32",
    })
    df = _normalize_list_columns(df)

    if not skip_type_inference:
        df = _infer_missing_types(conn, df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f"Cached training features to {cache_path}")

    return df
