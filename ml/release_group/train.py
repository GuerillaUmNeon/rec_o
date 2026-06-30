"""Train KNN release group recommender (sklearn Pipeline + sparse features)."""

from __future__ import annotations

import random

import pandas as pd
from gcld3 import NNetLanguageIdentifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sqlalchemy import text

from ml.release_group.config import (
    DEFAULT_N_NEIGHBORS,
    LENGTH_MS_FOR_ALBUM,
    LENGTH_MS_FOR_SINGLE,
    RELEASE_GROUP_ML_TRACK_META_CHUNK_SIZE,
    TRACKS_FOR_ALBUM,
)
from ml.release_group.features import ListToSparseTransformer


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
SELECT
    rg.id AS release_group_id,
    COALESCE(rg.name, rg.id::text) AS release_group_title,
    t.name AS track_title
FROM release_group rg
LEFT JOIN release r
    ON r.release_group = rg.id
LEFT JOIN medium m
    ON m.release = r.id
LEFT JOIN track t
    ON t.medium = m.id
WHERE rg.id = ANY (%s)
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


def _normalize_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("tag_ids", "genre_ids", "secondary_type_ids"):
        if col in df.columns:
            vals = df[col].tolist()
            for i, x in enumerate(vals):
                if not isinstance(x, list):
                    try:
                        is_na = pd.isna(x)
                    except (ValueError, TypeError):
                        is_na = False
                    vals[i] = [] if is_na else [x]
            df[col] = vals
    return df


def _is_empty_list(x) -> bool:
    return isinstance(x, list) and len(x) == 0


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

    with conn.connection.cursor() as cur:
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


class BackfillGenresTagsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, conn_factory, chunk_size: int = 50_000):
        self.conn_factory = conn_factory
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data = _normalize_list_columns(data)

        if data.empty or "genre_ids" not in data.columns or "tag_ids" not in data.columns:
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

        chunks = [
            empty_genre_rg_ids[i:i + self.chunk_size]
            for i in range(0, len(empty_genre_rg_ids), self.chunk_size)
        ]
        result_parts = []
        for chunk_idx, chunk_ids in enumerate(chunks, 1):
            print(f"  Backfill genre chunk {chunk_idx}/{len(chunks)} ({len(chunk_ids):,} IDs)...")
            with self.conn_factory() as conn:
                conn.execute(text("""
                    CREATE TEMP TABLE IF NOT EXISTS tmp_empty_genre_rg (
                        release_group_id INT PRIMARY KEY
                    ) ON COMMIT PRESERVE ROWS;
                """))
                conn.execute(text("TRUNCATE TABLE tmp_empty_genre_rg;"))
                conn.execute(
                    text("INSERT INTO tmp_empty_genre_rg (release_group_id) VALUES (:rg_id) ON CONFLICT DO NOTHING;"),
                    [{"rg_id": rg_id} for rg_id in chunk_ids],
                )

                df_chunk = pd.read_sql_query(text(RECORDING_GENRES_QUERY), conn)
                if not df_chunk.empty:
                    result_parts.append(df_chunk)

        df_result = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame(columns=["id", "tag_ids", "genre_ids"])

        print(f"Rows returned from fallback genre query: {len(df_result)}")

        if df_result.empty:
            return data

        df_result = _normalize_list_columns(df_result)

        print("Building lookup dicts...")
        tag_dict = {}
        genre_dict = {}
        for row_id, tags, genres in zip(
            df_result["id"], df_result["tag_ids"], df_result["genre_ids"]
        ):
            tag_dict[row_id] = tags if isinstance(tags, list) else []
            genre_dict[row_id] = genres if isinstance(genres, list) else []

        print("Applying backfilled tags and genres...")
        tag_col = data["tag_ids"].tolist()
        genre_col = data["genre_ids"].tolist()
        id_col = data["id"].tolist()

        for i, row_id in enumerate(id_col):
            cur_tags = tag_col[i]
            if isinstance(cur_tags, list) and len(cur_tags) == 0 and row_id in tag_dict:
                tag_col[i] = tag_dict[row_id]
            cur_genres = genre_col[i]
            if isinstance(cur_genres, list) and len(cur_genres) == 0 and row_id in genre_dict:
                genre_col[i] = genre_dict[row_id]

        data["tag_ids"] = tag_col
        data["genre_ids"] = genre_col
        print("Backfill complete.")

        return data


class LanguageScriptInferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, conn_factory, chunk_size: int = 50_000):
        self.conn_factory = conn_factory
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if df.empty:
            return df

        pre_language = df["language"].isna().sum()
        pre_script = df["script"].isna().sum()

        valid_language_ids = {int(v) for v in LANG_ID_MAP.values()}
        valid_script_ids = {int(v) for v in SCRIPT_ID_MAP.values()}

        df["language"] = df["language"].astype("object")
        df["script"] = df["script"].astype("object")

        mask = df["language"].isna() | df["script"].isna()
        release_group_ids = df.loc[mask, "id"].dropna().astype(int).unique().tolist()

        print(f"Fetching release texts for {len(release_group_ids):,} release groups...")
        release_texts = {}
        for start in range(0, len(release_group_ids), self.chunk_size):
            batch = release_group_ids[start:start + self.chunk_size]
            print(f"  Language chunk {start // self.chunk_size + 1}/{(len(release_group_ids) + self.chunk_size - 1) // self.chunk_size} ({len(batch):,} IDs)...")
            with self.conn_factory() as conn:
                batch_texts = _fetch_release_texts(conn, batch)
            release_texts.update(batch_texts)

        pred_languages = []
        pred_scripts = []
        mask_indices = df.loc[mask].index.tolist()
        mask_ids = df.loc[mask, "id"].tolist()
        total = len(mask_indices)

        print(f"Detecting languages for {total:,} release groups...")
        for i, (idx, row_id) in enumerate(zip(mask_indices, mask_ids)):
            if i > 0 and i % 100_000 == 0:
                print(f"  Language detection progress: {i:,}/{total:,}")

            rgid = str(int(row_id))
            info = release_texts.get(rgid, {})
            text_val = " ".join(
                x for x in [
                    info.get("release_group_title", ""),
                    " ".join(info.get("track_titles", []))
                ]
                if str(x).strip()
            ).strip()

            detected = _detect_lang(text_val)
            norm = _normalize_detected_language(detected, text_val)

            pred_languages.append(int(LANG_ID_MAP[norm]) if norm in LANG_ID_MAP else pd.NA)
            pred_scripts.append(int(SCRIPT_ID_MAP[norm]) if norm in SCRIPT_ID_MAP else pd.NA)

        print(f"Language detection complete for {total:,} rows.")

        bad_language_mask = df["language"].notna() & ~df["language"].isin(valid_language_ids)
        bad_script_mask = df["script"].notna() & ~df["script"].isin(valid_script_ids)

        df.loc[bad_language_mask, "language"] = pd.NA
        df.loc[bad_script_mask, "script"] = pd.NA

        df.loc[mask_indices, "language"] = pred_languages
        df.loc[mask_indices, "script"] = pred_scripts

        df["language"] = df["language"].astype("Int32")
        df["script"] = df["script"].astype("Int32")

        print(f"{pre_language - df['language'].isna().sum()} languages updated, {pre_script - df['script'].isna().sum()} scripts updated")
        return df


class ReleaseGroupTypeInferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, conn_factory, chunk_size: int = RELEASE_GROUP_ML_TRACK_META_CHUNK_SIZE):
        self.conn_factory = conn_factory
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        if data.empty or "type" not in data.columns:
            return data

        null_type = data[data["type"].isna()].copy()
        if null_type.empty:
            return data

        release_group_ids = null_type["id"].dropna().astype(int).tolist()
        chunks = []

        with self.conn_factory() as conn:
            for start in range(0, len(release_group_ids), self.chunk_size):
                batch = release_group_ids[start:start + self.chunk_size]
                batch_df = pd.read_sql_query(TRACK_META_QUERY, conn, params=(batch,))
                chunks.append(batch_df)

        track_meta = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=["id", "track_numbers", "album_length_ms"])
        )

        null_type = null_type.merge(track_meta, on="id", how="left")
        null_type["track_numbers"] = null_type["track_numbers"].astype("Int32")
        null_type["album_length_ms"] = null_type["album_length_ms"].astype("Int64")
        null_type["type"] = null_type.apply(_classify_type, axis=1)

        computed_type_map = null_type.set_index("id")["type"]
        data["type"] = data["type"].fillna(data["id"].map(computed_type_map)).astype("Int32")

        return data


def build_release_group_knn_artifact(
    data: pd.DataFrame,
    conn_factory,
    n_neighbors: int | None = None,
    *,
    backfill_empty_genres: bool = True,
    backfill_lang_inference: bool = True,
    infer_missing_types: bool = True,
) -> dict:
    """
    Train release group KNN from a prepared DataFrame.

    Artifact format matches note_book_guillaume.ipynb bundle keys.
    """

    n_neighbors = n_neighbors or DEFAULT_N_NEIGHBORS

    list_cols = ["tag_ids", "genre_ids", "secondary_type_ids"]
    categorical_cols = ["type", "status", "language", "script"]
    numeric_mean_cols = ["year"]
    exclude_cols = ["id", "artist_credit", "tag", "count"]

    list_cols = [c for c in list_cols if c in data.columns]
    categorical_cols = [c for c in categorical_cols if c in data.columns]
    numeric_mean_cols = [c for c in numeric_mean_cols if c in data.columns]

    scalar_feature_cols = [
        c for c in data.columns
        if c not in exclude_cols + list_cols
    ]

    steps = []

    if backfill_empty_genres:
        steps.append(("backfill_genres_tags", BackfillGenresTagsTransformer(conn_factory=conn_factory)))

    if backfill_lang_inference:
        steps.append(("language_script_inference", LanguageScriptInferenceTransformer(conn_factory=conn_factory)))

    if infer_missing_types:
        steps.append(("type_inference", ReleaseGroupTypeInferenceTransformer(conn_factory=conn_factory)))

    steps.append((
        "preprocess",
        ListToSparseTransformer(
            categorical_cols=categorical_cols,
            numeric_mean_cols=numeric_mean_cols,
            list_cols=list_cols,
        )
    ))

    steps.append((
        "knn",
        NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
        )
    ))

    # Fit each step manually to avoid running heavy transformers twice.
    # (Pipeline.fit would fit_transform all steps, then we'd need to
    #  re-transform to capture intermediate data — doubling the work.)
    transformed_data = data.copy()
    for _name, step in steps[:-2]:
        transformed_data = step.fit_transform(transformed_data)

    preprocess_step = steps[-2][1]
    knn_step = steps[-1][1]
    X_sparse = preprocess_step.fit_transform(transformed_data)
    knn_step.fit(X_sparse)

    pipeline = Pipeline(steps=steps)

    # Clear conn_factory on fitted transformers so the artifact is picklable
    # (lambdas / closures cannot be pickled by joblib).
    for _name, step in pipeline.steps:
        if hasattr(step, "conn_factory"):
            step.conn_factory = None

    id_to_idx = {row_id: idx for idx, row_id in enumerate(transformed_data["id"])}

    return {
        "model_kind": "release_group_knn",
        "pipeline": pipeline,
        "data_model": transformed_data,
        "scalar_feature_cols": scalar_feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_mean_cols": numeric_mean_cols,
        "list_cols": list_cols,
        "id_to_idx": id_to_idx,
        "n_neighbors": n_neighbors,
    }

# TODO: check if this function is useless
def recommend_release_group_ids_from_artifact(
    artifact: dict,
    release_group_ids: list[int],
    top_n: int = 5,
    *,
    exclude_seed: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """Notebook-style inference helper for tests and future app/release_group."""
    pipeline = artifact["pipeline"]
    data_model = artifact["data_model"]
    id_to_idx = artifact["id_to_idx"]

    knn = pipeline.named_steps["knn"]
    preprocessor = pipeline.named_steps["preprocess"]

    if seed is not None:
        random.seed(seed)

    neighbor_lists: list[pd.DataFrame] = []
    for target_id in release_group_ids:
        if target_id not in id_to_idx:
            print(f"Warning: ID {target_id} not found, skipping.")
            continue

        row_idx = id_to_idx[target_id]
        row_data = data_model.iloc[row_idx:row_idx + 1]

        X_row = preprocessor.transform(row_data)
        distances, indices = knn.kneighbors(X_row, n_neighbors=top_n)

        neighbors = data_model.iloc[indices[0]].copy()
        neighbors["distance"] = distances[0]
        neighbors["query_id"] = target_id

        if exclude_seed:
            neighbors = neighbors[neighbors["id"] != target_id]

        neighbor_lists.append(neighbors)

    if not neighbor_lists:
        return pd.DataFrame(columns=list(data_model.columns) + ["distance", "query_id"])

    result = pd.concat(neighbor_lists, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    return result