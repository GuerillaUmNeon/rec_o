# ML pipeline (offline)

Train the KNN artist recommender **locally**, then upload to GCS **manually** when ready.

Offline training lives here. The API in [`app/`](../app/README_APP.md) only loads a trained artifact and serves HTTP routes.

## Layout

```
ml/
  config.py            # paths, MODEL_BUCKET_NAME, ML_MAX_ARTISTS, ML_GENRE_CHUNK_SIZE
  data.py              # fetch_artist_training_data + SQL
  features.py          # genre tokens (aligned with app/predictor.py)
  train.py             # build_artist_recommender_artifact (Tfidf + KNN)
  artifact.py          # local save only (no GCS)
  gcs_upload.py        # GCS upload helper
  scripts/
    run_local.py       # train + save local
    upload_to_gcs.py   # upload an existing .pkl to GCS
  outputs/             # training_features.pkl cache + model copies (gitignored)
```

There is **no** `run_pipeline.py` — use `run_local` and `upload_to_gcs` separately.

## Prerequisites

- `.env` with Postgres (`DATABASE_URL` or `POSTGRES`, `DATABASE`, etc.)
- For upload only:
  - `MODEL_BUCKET_NAME=rec-o-models`
  - `MODEL_BLOB_NAME=models/knn_baseline_model.pkl`
  - `gcloud auth application-default login` on project **rec-o-gcp**
  - IAM: **Storage Object Creator** on bucket `rec-o-models`

Optional in `.env`:

```bash
# Faster training
# ML_MAX_ARTISTS=20000
# ML_GENRE_CHUNK_SIZE=2000

# Test artifact names (do not overwrite prod)
MODEL_LOCAL_FILENAME=knn_baseline_model_test.pkl
MODEL_BLOB_NAME=models/knn_baseline_model_test.pkl
```

| Variable | Used by | Prod value |
|----------|---------|------------|
| `MODEL_LOCAL_FILENAME` | `run_local` saves (`models/`, `ml/outputs/`) | `knn_baseline_model.pkl` |
| `MODEL_BLOB_NAME` | `upload_to_gcs` (+ API download if same `.env`) | `models/knn_baseline_model.pkl` |

Cloud Run prod reads `MODEL_BLOB_NAME` from Secret Manager (default `models/knn_baseline_model.pkl`) — your local `.env` test values do not change prod until you upload to the prod blob path and update the secret.

**Run the API with the test model:** after `run_local`, upload with `upload_to_gcs` or set `MODEL_LOCAL_PATH` — see [app/README_APP.md](../app/README_APP.md).

## 1. Train and save locally

**Full DB** — temp tables + genre tokens:

```bash
python -m ml.scripts.run_local
```

**Fast dev** — scoped SQL (`--limit` / `--skip-extended-genres`), same artifact format:

**Fast dev (recommended):**

```bash
python -m ml.scripts.run_local --limit 5000 --skip-extended-genres --use-cache
```

**With extended genres (optimized SQL, batched):**

```bash
python -m ml.scripts.run_local --limit 20000 --use-cache --refresh-cache
```

### CLI options (`run_local`)

| Flag | Effect |
|------|--------|
| `--limit N` | Cap artists; query 2 scoped + batched (see below) |
| `--skip-extended-genres` | Skip query 2 (tags only — fastest) |
| `--use-cache` | Load `ml/outputs/training_features.pkl` if present (skip SQL) |
| `--refresh-cache` | Re-fetch SQL even with `--use-cache` |

### Output files (local only, no GCS)

| File | What it is |
|------|------------|
| `ml/outputs/training_features.pkl` | **SQL cache** — see below |
| `models/<MODEL_LOCAL_FILENAME>` | **Trained model** (canonical; used by `upload_to_gcs` by default) |
| `ml/outputs/<MODEL_LOCAL_FILENAME>` | Same artifact (convenience copy) |
| `models/<stem>_<timestamp>.pkl` | Timestamped backup per run |

#### `training_features.pkl` (intermediate cache)

**Not** the final model. It is a pandas DataFrame saved after SQL fetch/aggregation, **before** `build_artist_recommender_artifact`.

Typical columns: `artist_id`, `artist_name`, `genres`, `tags`, `tag_count_sum`, … — **one row per artist**.

- **Created** at the end of each successful SQL fetch (overwritten on the next fetch).
- **`--use-cache`**: load this file and **skip SQL** on the next `run_local` (still runs clean + train + saves `knn_baseline_model.pkl`).
- **`--refresh-cache`**: force new SQL and replace the cache.

Use it to iterate on training parameters without re-hitting the database.

#### `knn_baseline_model.pkl` (final artifact)

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

Produced only after training finishes. Upload this one to GCS with `upload_to_gcs`.

## 2. Upload to GCS (manual)

After a successful `run_local`:

```bash
gcloud config set project rec-o-gcp
gcloud auth application-default login
python -m ml.scripts.upload_to_gcs
```

Custom path:

```bash
python -m ml.scripts.upload_to_gcs --path ml/outputs/knn_baseline_model_test.pkl
```

Default lookup (no `--path`): `models/<MODEL_LOCAL_FILENAME>`, then `ml/outputs/<MODEL_LOCAL_FILENAME>` (from `.env`).

### `.env` vs GCP credentials

| Variable | Role |
|----------|------|
| `MODEL_LOCAL_FILENAME` | Local file `upload_to_gcs` sends (and `run_local` writes) |
| `MODEL_BLOB_NAME` | GCS object path, e.g. `models/knn_baseline_model_test.pkl` |
| `MODEL_BUCKET_NAME` | Bucket, e.g. `rec-o-models` |

**Putting only the GCP project name in `.env` does not fix a 403.** Upload uses **Application Default Credentials** (ADC). The project name does not replace a service account that lacks `storage.objects.create` on `rec-o-models` (typical error: `le-wagon-data-bootcamp@airy-cogency-493213-t4...` from another project).

| In `.env` | Fixes 403? |
|-----------|------------|
| `MODEL_*` paths | No — only **where** / **which file** |
| Project name only | No — does not change **who** authenticates |
| `GOOGLE_APPLICATION_CREDENTIALS` + JSON key for **rec-o-gcp** | Yes — if that SA has **Storage Object Creator** on the bucket |

**Recommended fix:**

```bash
echo "$GOOGLE_APPLICATION_CREDENTIALS"
```

If you see a path (e.g. `.../airy-cogency-493213-t4-....json`), every **new terminal** reloads it from `~/.zshrc` — comment out that `export` line, then `source ~/.zshrc`. `unset` alone is not enough until the shell config is fixed.

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project rec-o-gcp
gcloud auth application-default login    # use a Google account with access to rec-o-gcp
python -m ml.scripts.upload_to_gcs
```

Check the success line ends with your test blob, e.g. `gs://rec-o-models/models/knn_baseline_model_test.pkl`. If 403 persists, request **Storage Object Creator** on `rec-o-models` for your user in project **rec-o-gcp**.

Production Cloud Run reads `MODEL_BLOB_NAME` from Secret Manager — update the secret and redeploy a revision to switch models (no image rebuild). See [GCP_SETUP_STEPS.md](../GCP_SETUP_STEPS.md).

## Why training is slow (full run, no `--limit`)

`fetch_artist_training_data` runs **2 SQL queries** on the full MusicBrainz DB:

| # | Query | Role | Approx. size |
|---|--------|------|----------------|
| 1 | `artist_query` | Artist metadata + tags | **~3.3M rows** |
| 2 | Extended genres CTE | Release / recording / work tags | Very large (10+ min) |

Then pandas `groupby`, sklearn fit, `joblib` save. Remote DB adds latency.

### Optimized genre SQL (when `--limit` is set)

Used only in `ml/data.py` (not in `app/predictor.py`):

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

Same structure expected by `app/predictor.py` at inference.

## Reference notebook

`notebooks/note_book_joris.ipynb`
