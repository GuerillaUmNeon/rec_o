"""Fetch and prepare release group KNN training features from MusicBrainz."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from ml.release_group.config import (
    RELEASE_GROUP_ML_MAX_ROWS,
    RELEASE_GROUP_TRAINING_FEATURES_CACHE,
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
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
            )
    return df


def fetch_release_group_knn_training_data(
    conn,
    *,
    max_rows: int | None = None,
    use_cache: bool = False,
    refresh_cache: bool = False,
) -> pd.DataFrame:
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f"Cached training features to {cache_path}")

    return df