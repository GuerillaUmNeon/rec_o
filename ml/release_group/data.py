"""Fetch and prepare release group KNN training features from MusicBrainz."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from gcld3 import NNetLanguageIdentifier

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

RECORDING_GENRES_QUERY = """
WITH artist_tag_links AS (
        SELECT
            r.release_group,
            rt.tag AS tag_id
        FROM tmp_empty_genre_rg t
        JOIN release r
            ON r.release_group = t.release_group_id
        JOIN artist_credit ac
            ON ac.id = r.artist_credit
        JOIN artist_credit_name acn
            ON acn.artist_credit = ac.id
        JOIN artist_tag rt
            ON rt.artist = acn.artist
    ),
    tag_counts AS (
        SELECT
            release_group,
            tag_id,
            COUNT(*) AS tag_count
        FROM artist_tag_links
        GROUP BY release_group, tag_id
    ),
    tag_with_genre AS (
        SELECT
            atl.release_group,
            atl.tag_id,
            g.id AS genre_id
        FROM artist_tag_links atl
        JOIN tag t
            ON t.id = atl.tag_id
        LEFT JOIN genre g
            ON g.name = t.name
    ),
    genre_counts AS (
        SELECT
            release_group,
            genre_id,
            COUNT(*) AS genre_count
        FROM tag_with_genre
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
    )
    SELECT
        t.release_group_id AS id,
        COALESCE(tb.tag_ids, ARRAY[]::integer[]) AS tag_ids,
        COALESCE(gb.genre_ids, ARRAY[]::integer[]) AS genre_ids
    FROM tmp_empty_genre_rg t
    LEFT JOIN tag_buckets tb
        ON tb.release_group = t.release_group_id
    LEFT JOIN genre_buckets gb
        ON gb.release_group = t.release_group_id;
"""

NULL_LANGUAGE_QUERY = """
      SELECT rg.id                          AS release_group_id, \
             COALESCE(rg.name, rg.id::text) AS release_group_title, \
             t.name                         AS track_title
      FROM release_group rg
               LEFT JOIN release r ON r.release_group = rg.id
               LEFT JOIN medium m ON m.release = r.id
               LEFT JOIN track t ON t.medium = m.id
      WHERE rg.id = ANY (%s) \
      """

LANG_ID_MAP = {
    "af": "9", "am": "15", "ar": "18", "bg": "62", "bg-Latn": "62", "bn": "47",
    "bs": "56", "ca": "68", "ceb": "68", "co": "89", "cs": "98", "cy": "455",
    "da": "100", "de": "145", "el": "159", "el-Latn": "159", "en": "120",
    "eo": "122", "es": "393", "et": "123", "eu": "41", "fa": "334", "fi": "131",
    "fil": "189", "fr": "134", "fy": "137", "ga": "149", "gd": "148", "gl": "150",
    "gu": "161", "ha": "165", "haw": "165", "he": "167", "hi": "171", "hi-Latn": "171",
    "hmn": "179", "hr": "366", "ht": "164", "hu": "176", "hy": "21", "id": "189",
    "ig": "179", "is": "180", "it": "195", "iw": "167", "ja": "198", "ja-Latn": "198",
    "jv": "196", "ka": "144", "kk": "211", "km": "215", "kn": "206", "ko": "224",
    "ku": "232", "ky": "219", "la": "238", "lb": "246", "lo": "237", "lt": "243",
    "lv": "239", "mg": "275", "mi": "262", "mk": "254", "ml": "260", "mn": "282",
    "mr": "264", "ms": "266", "mt": "276", "my": "63", "ne": "300", "nl": "113",
    "no": "309", "ny": "313", "pa": "330", "pl": "338", "ps": "343", "pt": "340",
    "ro": "351", "ru": "353", "ru-Latn": "353", "sd": "387", "si": "373", "sk": "377",
    "sl": "378", "sm": "384", "sn": "386", "so": "390", "sq": "12", "sr": "363",
    "st": "392", "su": "399", "sv": "403", "sw": "402", "ta": "407", "te": "409",
    "tg": "413", "th": "415", "tr": "433", "uk": "441", "ur": "444", "uz": "445",
    "vi": "448", "xh": "460", "yi": "463", "yo": "464", "zh": "76", "zh-Latn": "76",
    "zu": "470"
}

SCRIPT_ID_MAP = {
    "af": "28", "am": "86", "ar": "18", "bg": "31", "bg-Latn": "28", "bn": "53",
    "bs": "28", "ca": "28", "ceb": "28", "co": "28", "cs": "28", "cy": "28",
    "da": "28", "de": "28", "el": "22", "el-Latn": "28", "en": "28", "eo": "28",
    "es": "28", "et": "28", "eu": "28", "fa": "18", "fi": "28", "fil": "28",
    "fr": "28", "fy": "28", "ga": "28", "gd": "28", "gl": "28", "gu": "52",
    "ha": "28", "haw": "28", "he": "11", "hi": "50", "hi-Latn": "28", "hmn": "28",
    "hr": "28", "ht": "28", "hu": "28", "hy": "35", "id": "28", "ig": "28",
    "is": "28", "it": "28", "iw": "11", "ja": "85", "ja-Latn": "28", "jv": "28",
    "ka": "36", "kk": "31", "km": "68", "kn": "60", "ko": "43", "ku": "28",
    "ky": "31", "la": "28", "lb": "28", "lo": "69", "lt": "28", "lv": "28",
    "mg": "28", "mi": "28", "mk": "31", "ml": "62", "mn": "31", "mr": "50",
    "ms": "28", "mt": "28", "my": "64", "ne": "50", "nl": "28", "no": "28",
    "ny": "28", "pa": "49", "pl": "28", "ps": "18", "pt": "28", "ro": "28",
    "ru": "31", "ru-Latn": "28", "sd": "18", "si": "63", "sk": "28", "sl": "28",
    "sm": "28", "sn": "28", "so": "28", "sq": "28", "sr": "31", "st": "28",
    "su": "107", "sv": "28", "sw": "28", "ta": "61", "te": "59", "tg": "31",
    "th": "65", "tr": "28", "uk": "31", "ur": "18", "uz": "28", "vi": "28",
    "xh": "28", "yi": "11", "yo": "28", "zh": "92", "zh-Latn": "28", "zu": "28"
}

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

def _is_empty_list(x) -> bool:
    return isinstance(x, list) and len(x) == 0

def _backfill_empty_genres_and_tags(conn, data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    empty_genre_rg_ids = (
        data.loc[data["genre_ids"].apply(_is_empty_list), "id"]
        .dropna()
        .astype(int)
        .unique()
    )

    print(f"Total release groups with empty genre list: {len(empty_genre_rg_ids)}")

    if len(empty_genre_rg_ids) == 0:
        print("No release groups with empty genre list.")
        return data

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS tmp_empty_genre_rg (
                release_group_id INT PRIMARY KEY
            ) ON COMMIT DROP;
        """)
        cur.execute("TRUNCATE TABLE tmp_empty_genre_rg;")
        cur.executemany(
            "INSERT INTO tmp_empty_genre_rg (release_group_id) VALUES (%s) ON CONFLICT DO NOTHING;",
            ((rg_id,) for rg_id in empty_genre_rg_ids),
        )

    df_result = pd.read_sql_query(RECORDING_GENRES_QUERY, conn)
    print(f"Rows returned from fallback genre query: {len(df_result)}")

    if df_result.empty:
        return data

    df_result = _normalize_list_columns(df_result)

    computed_genre_map = df_result.set_index("id")["genre_ids"]
    computed_tag_map = df_result.set_index("id")["tag_ids"]

    updated = data.copy()

    mask_empty_tag = updated["tag_ids"].apply(_is_empty_list)
    mapped_tag = updated.loc[mask_empty_tag, "id"].map(computed_tag_map)
    updated.loc[mask_empty_tag, "tag_ids"] = mapped_tag.apply(
        lambda x: x if isinstance(x, list) else []
    )

    mask_empty_genre = updated["genre_ids"].apply(_is_empty_list)
    mapped_genre = updated.loc[mask_empty_genre, "id"].map(computed_genre_map)
    updated.loc[mask_empty_genre, "genre_ids"] = mapped_genre.apply(
        lambda x: x if isinstance(x, list) else []
    )

    new_empty_tag_ids_count = updated["tag_ids"].apply(_is_empty_list).sum()
    new_empty_genre_ids_count = updated["genre_ids"].apply(_is_empty_list).sum()

    print(f"Empty tag_ids after fallback: {new_empty_tag_ids_count}")
    print(
        f"Empty genre_ids after fallback: {new_empty_genre_ids_count} "
        f"({round(new_empty_genre_ids_count / updated.shape[0] * 100)}%)"
    )

    return updated

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

def _detect_lang(text_value):
    identifier = NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

    if text_value is None:
        return None
    text_value = str(text_value).strip()
    if not text_value:
        return None
    result = identifier.FindLanguage(text_value)
    if not result or not result.language:
        return None
    if hasattr(result, "is_reliable") and not result.is_reliable:
        return None
    return result.language

def _normalize_detected_language(lang_code, original_text):
    if not lang_code:
        return None
    if lang_code == "iw":
        lang_code = "he"
    if lang_code == "tl":
        lang_code = "fil"

    original_text = (original_text or "").strip()
    has_ascii = any(ch.isascii() and ch.isalpha() for ch in original_text)
    has_non_ascii = any(not ch.isascii() for ch in original_text)

    translit_candidates = {"bg", "el", "hi", "ja", "ru", "zh"}
    if lang_code in translit_candidates and has_ascii and not has_non_ascii:
        latn_code = f"{lang_code}-Latn"
        if latn_code in LANG_ID_MAP:
            return latn_code

    return lang_code

def _fetch_release_texts(conn, release_group_ids):
    if not release_group_ids:
        return {}

    with conn.cursor() as cur:
        cur.execute(NULL_LANGUAGE_QUERY, (release_group_ids,))
        rows = cur.fetchall()

    agg = {}
    for release_group_id, release_group_title, track_title in rows:
        key = str(release_group_id)
        item = agg.setdefault(key, {"release_group_title": "", "track_titles": []})
        if release_group_title and not item["release_group_title"]:
            item["release_group_title"] = str(release_group_title).strip()
        if track_title and str(track_title).strip():
            item["track_titles"].append(str(track_title).strip())
    return agg

def _fetch_languages(conn, df):
    pre_language = df["language"].isna().sum()
    pre_script = df["script"].isna().sum()

    print('Getting languages and scripts from titles')

    conn.rollback()

    RELEASE_GROUP_COL = "id"
    LANGUAGE_COL = "language"
    SCRIPT_COL = "script"

    VALID_LANGUAGE_IDS = set(LANG_ID_MAP.values())
    VALID_SCRIPT_IDS = set(SCRIPT_ID_MAP.values())

    if RELEASE_GROUP_COL not in df.columns:
        raise KeyError(f"Missing column '{RELEASE_GROUP_COL}'. Available columns: {list(df.columns)}")

    if LANGUAGE_COL not in df.columns:
        df[LANGUAGE_COL] = None
    if SCRIPT_COL not in df.columns:
        df[SCRIPT_COL] = None

    df[LANGUAGE_COL] = df[LANGUAGE_COL].astype("object")
    df[SCRIPT_COL] = df[SCRIPT_COL].astype("object")

    mask = df[LANGUAGE_COL].isna() | df[SCRIPT_COL].isna()
    release_group_ids = df.loc[mask, RELEASE_GROUP_COL].dropna().astype(str).unique().tolist()
    release_texts = _fetch_release_texts(conn, release_group_ids)

    preds = pd.DataFrame(index=df.loc[mask].index, columns=["pred_language", "pred_script", "detected_code"])

    for idx, row in df.loc[mask].iterrows():
        rgid = str(row[RELEASE_GROUP_COL])
        info = release_texts.get(rgid, {})
        text = " ".join(
            x for x in [info.get("release_group_title", ""), " ".join(info.get("track_titles", []))]
            if str(x).strip()
        ).strip()

        detected = _detect_lang(text)
        norm = _normalize_detected_language(detected, text)

        preds.at[idx, "detected_code"] = norm
        preds.at[idx, "pred_language"] = int(LANG_ID_MAP[norm]) if norm in LANG_ID_MAP else None
        preds.at[idx, "pred_script"] = int(SCRIPT_ID_MAP[norm]) if norm in SCRIPT_ID_MAP else None

    df.loc[df[LANGUAGE_COL].notna() & ~df[LANGUAGE_COL].astype(str).isin(VALID_LANGUAGE_IDS), LANGUAGE_COL] = None
    df.loc[df[SCRIPT_COL].notna() & ~df[SCRIPT_COL].astype(str).isin(VALID_SCRIPT_IDS), SCRIPT_COL] = None

    df.loc[preds.index, LANGUAGE_COL] = preds["pred_language"]
    df.loc[preds.index, SCRIPT_COL] = preds["pred_script"]

    df[LANGUAGE_COL] = df[LANGUAGE_COL].astype("Int32")
    df[SCRIPT_COL] = df[SCRIPT_COL].astype("Int32")

    post_language = df["language"].isna().sum()
    post_script = df["script"].isna().sum()

    updated_languages = pre_language - post_language
    updated_scripts = pre_script - post_script

    print(f"{updated_languages} languages updated, {updated_scripts} scripts updated")
    return df

def fetch_release_group_knn_training_data(
    conn,
    *,
    max_rows: int | None = None,
    skip_type_inference: bool = False,
    use_cache: bool = False,
    refresh_cache: bool = False,
    backfill_empty_genres: bool = True,
    backfill_lang_inference: bool = True,
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

    if backfill_empty_genres:
        df = _backfill_empty_genres_and_tags(conn, df)

    if backfill_lang_inference:
        df = _fetch_languages(conn, df)

    if not skip_type_inference:
        df = _infer_missing_types(conn, df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f"Cached training features to {cache_path}")

    return df