# ML pipeline (offline)

Train KNN recommenders **locally**, then upload to GCS manually when ready.

Offline training lives under `ml/artist/` and `ml/release_group/`. The API in [`app/`](../app/README_APP.md) loads trained artifacts at runtime. `ml/genre/` is planned next.

Reference notebook: `models/note_book_guillaume.ipynb` (release group).

## Layout

```
ml/
  artist/                        # artist KNN training (implemented)
    config.py                    # ARTIST_MODEL_*, MODEL_BUCKET_NAME
    data.py                      # fetch_artist_knn_training_data* + SQL
    features.py                  # artist genre tokens
    train.py                     # build_artist_knn_artifact (Tfidf + KNN)
    artifact.py                  # save_artist_knn_artifact (local only)
    gcs_upload.py                # upload_artist_knn_model_to_gcs
    scripts/
      train_local.py             # train + save local
      upload_artist.py           # upload .pkl to GCS
  release_group/                 # album KNN training (implemented)
    config.py                    # RELEASE_GROUP_MODEL_*
    data.py                      # fetch_release_group_knn_training_data + SQL
    features.py                  # ListToSparseTransformer
    train.py                     # build_release_group_knn_artifact
    artifact.py                  # save_release_group_knn_artifact
    gcs_upload.py                # upload_release_group_knn_model_to_gcs
    scripts/
      train_local.py
      upload_release_group.py
  genre/                         # genre KNN training (stub)
  outputs/                       # training caches + model copies (gitignored)
```

Each subpackage uses `train_local` + `upload_*` separately (no single pipeline script). The shared bucket is `MODEL_BUCKET_NAME`.

## Prerequisites

- `.env` with Postgres (`DATABASE_URL` or `POSTGRES`, `DATABASE`, etc.)

Optional in `.env`:

```bash
# Faster artist training
# ARTIST_ML_MAX_ARTISTS=20000
# ARTIST_ML_GENRE_CHUNK_SIZE=2000

# Test artifact names (do not overwrite prod)
ARTIST_MODEL_BLOB_NAME=models/knn_model_test_joris_slim.pkl
ARTIST_MODEL_LOCAL_FILENAME=knn_model_test_joris_slim.pkl
```

`MODEL_BUCKET_NAME` is shared across models. Legacy `MODEL_BLOB_NAME`, `MODEL_LOCAL_FILENAME`, `ML_MAX_ARTISTS`, `ML_GENRE_CHUNK_SIZE` still work as fallbacks.

| Variable | Used by | Prod value |
|----------|---------|------------|
| `MODEL_BUCKET_NAME` | API + ML upload (all models) | `rec-o-models` |
| `ARTIST_MODEL_BLOB_NAME` | API download + `upload_artist` | `models/knn_model_test_joris_slim.pkl` |
| `ARTIST_MODEL_LOCAL_FILENAME` | `train_local` saves (`models/`) | `knn_model_test_joris_slim.pkl` |
| `RELEASE_GROUP_MODEL_BLOB_NAME` | API download + `upload_release_group` | `models/release_group_knn_model.pkl` |
| `RELEASE_GROUP_MODEL_LOCAL_FILENAME` | `train_local` saves (`models/`) | `release_group_knn_model.pkl` |

Cloud Run prod reads `ARTIST_MODEL_BLOB_NAME` and `RELEASE_GROUP_MODEL_BLOB_NAME` from Secret Manager — your local `.env` test values do not change prod until you upload to the prod blob path and update the secret.

**Run the API with test models:** after `train_local`, upload with `upload_*` or set `ARTIST_MODEL_LOCAL_PATH` / `RELEASE_GROUP_MODEL_LOCAL_PATH` — see [app/README_APP.md](../app/README_APP.md).

## 1. Train and save locally

**Full DB** — temp tables + genre tokens:

```bash
python -m ml.artist.scripts.train_local
```

**Fast dev** — scoped SQL (`--limit` / `--skip-extended-genres`), same artifact format:

**Fast dev (recommended):**

```bash
python -m ml.artist.scripts.train_local --limit 5000 --skip-extended-genres --use-cache
```

**With extended genres (optimized SQL, batched):**

```bash
python -m ml.artist.scripts.train_local --limit 20000 --use-cache --refresh-cache
```

### CLI options (`train_local`)

| Flag | Effect |
|------|--------|
| `--limit N` | Cap artists; query 2 scoped + batched (see below) |
| `--skip-extended-genres` | Skip query 2 (tags only — fastest) |
| `--use-cache` | Load `ml/outputs/artist_training_features.pkl` if present (skip SQL) |
| `--refresh-cache` | Re-fetch SQL even with `--use-cache` |

### Output files (local only, no GCS)
 
| File | What it is |
|------|------------|
| `ml/outputs/artist_training_features.pkl` | **SQL cache** — see below |
| `models/<ARTIST_MODEL_LOCAL_FILENAME>` | **Trained model** (canonical; used by `upload_artist` by default) |
| `models/<stem>_<timestamp>.pkl` | Timestamped backup per run |

#### `artist_training_features.pkl` (intermediate cache)

**Not** the final model. It is a pandas DataFrame saved after SQL fetch/aggregation, **before** `build_artist_knn_artifact`.

Typical columns: `artist_id`, `artist_name`, `genres`, `tags`, `tag_count_sum`, … — **one row per artist**.

- **Created** at the end of each successful SQL fetch (overwritten on the next fetch).
- **`--use-cache`**: load this file and **skip SQL** on the next `train_local` (still runs clean + train + saves the artist `.pkl`).
- **`--refresh-cache`**: force new SQL and replace the cache.

Use it to iterate on training parameters without re-hitting the database.

#### `knn_model_test_joris_slim.pkl` (final artifact)

Pickled dict for the API:

```python
{
    "vectorizer": TfidfVectorizer,
    "model": NearestNeighbors,
    "artist_names": dict[int, str],  # artist_id -> name
    "data": pd.DataFrame,
    "genre_feature_format": "genre_token_unigram",
}
```

Produced only after training finishes. Upload this one to GCS with `upload_artist`.

## 2. Model usage
Train and save locally; see [app/README_APP.md](../app/README_APP.md) for how the API loads artifacts.

## Release group KNN (`ml/release_group/`)

Port of `models/note_book_guillaume.ipynb` — sparse features (tags, genres, types) + sklearn `Pipeline` + KNN.

### Train locally

**Full DB** (slow — millions of release groups):

```bash
python -m ml.release_group.scripts.train_local
```

**Fast dev (recommended):**

```bash
python -m ml.release_group.scripts.train_local --limit 5000 --skip-type-inference --use-cache
```

| Flag | Effect |
|------|--------|
| `--limit N` | Cap release groups in main SQL |
| `--skip-type-inference` | Skip track-meta SQL for missing `type` |
| `--use-cache` | Load `ml/outputs/release_group_training_features.pkl` |
| `--refresh-cache` | Re-fetch SQL even with `--use-cache` |
| `--n-neighbors N` | KNN neighbors (default `10`) |

### Upload to GCS
Deprecated. Local training only.

### `.env` (release group)

| Variable | Role |
|----------|------|
| `RELEASE_GROUP_MODEL_LOCAL_FILENAME` | Local `.pkl` after `train_local` |
| `RELEASE_GROUP_MODEL_BLOB_NAME` | GCS object path |
| `RELEASE_GROUP_ML_MAX_ROWS` | Default `--limit` when flag omitted in scoped runs |
| `RELEASE_GROUP_ML_TRACK_META_CHUNK_SIZE` | Batch size for type-inference SQL |

### Artifact format

```python
{
    "model_kind": "release_group_knn",
    "pipeline": sklearn Pipeline(preprocess + knn),
    "data_model": pd.DataFrame,
    "id_to_idx": dict[int, int],
    "n_neighbors": 10,
    ...
}
```

---

## Why artist training is slow (full run, no `--limit`)

`fetch_artist_knn_training_data_scoped` runs **2 SQL queries** on the full MusicBrainz DB:

| # | Query | Role | Approx. size |
|---|--------|------|----------------|
| 1 | `artist_query` | Artist metadata + tags | **~3.3M rows** |
| 2 | Extended genres CTE | Release / recording tags | Very large (10+ min) |

Then pandas `groupby`, sklearn fit, `joblib` save. Remote DB adds latency.

### Optimized genre SQL (when `--limit` is set)

Used only in `ml/artist/data.py`:

- `unnest(%s::int[])` instead of thousands of `VALUES` rows
- Skips artist-level tags in query 2 (already in query 1)
- `UNION ALL` + `DISTINCT` instead of heavy `NOT EXISTS`
- Batched by `ML_GENRE_CHUNK_SIZE` (default 2000) with per-batch timing in logs

## Artifact format

```python
{
    "vectorizer": TfidfVectorizer,
    "model": NearestNeighbors,
    "artist_names": list[str],
    "data": pd.DataFrame,  # df_clean: artist_id, genres, ...
}
```

Same structure expected by `app/artist/recommender.py` at inference.

## Reference notebook

`notebooks/note_book_joris.ipynb`
