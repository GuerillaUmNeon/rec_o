# API (`app/`)

FastAPI service for artist recommendations (KNN) and MusicBrainz search.  
**Training and GCS upload live in [`ml/`](../ml/README_ML.md)** — this package only **loads** a pre-trained artifact and serves HTTP routes.

## Layout

```
app/
  main.py       # FastAPI app, routes, auth, rate limits, lifespan (model load at startup)
  predictor.py  # Load artifact (local or GCS), KNN inference, artist enrichment SQL
  database.py   # PostgreSQL connection (psycopg)
  queries.py    # SQL for search endpoints
  schemas.py    # Pydantic request/response models
```

There is **no** training code here (`save_model`, SQL fetch for ML, etc. were moved to `ml/`).

## Run locally

From the project root:

```bash
uvicorn app.main:app --reload
```

- [http://localhost:8000](http://localhost:8000) — health
- [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger
- [http://localhost:8000/model](http://localhost:8000/model) — which model is loaded

Protected routes need header: `X-API-Key: <TOKEN_API_KEY>` (from `.env`).

## Model loading

The recommender `.pkl` is loaded **once at startup** via FastAPI **lifespan** (`main.py` → `load_model()` in `predictor.py`). It is **not** baked into the Docker image.

### Two sources (`.env`)

| Priority | Variable(s) | When to use |
|----------|-------------|-------------|
| **1 — Local file** | `ARTIST_MODEL_LOCAL_PATH` | Dev: test a `.pkl` on disk without GCS |
| **2 — GCS** | `MODEL_BUCKET_NAME` + `ARTIST_MODEL_BLOB_NAME` | Same as prod / Docker / Cloud Run |

If `ARTIST_MODEL_LOCAL_PATH` is set, GCS is **not** used (even if `MODEL_BUCKET_NAME` is set).  
Path can be absolute or relative to the **project root** (e.g. `models/knn_baseline_model_test.pkl`).  
Legacy `MODEL_LOCAL_PATH` / `MODEL_BLOB_NAME` still work as fallbacks.

If `ARTIST_MODEL_LOCAL_PATH` is missing, the API downloads from GCS to a **temp cache**:

```
gs://<MODEL_BUCKET_NAME>/<ARTIST_MODEL_BLOB_NAME>
    →  /tmp/<filename>.pkl   (e.g. /tmp/knn_baseline_model_test2.pkl)
    →  loaded into memory with joblib
```

That `/tmp/...` path is **runtime only** (not in the Docker image; gone when the container stops).

### Example `.env` — GCS (like prod)

```bash
MODEL_BUCKET_NAME=rec-o-models
ARTIST_MODEL_BLOB_NAME=models/knn_baseline_model_test2.pkl
# no ARTIST_MODEL_LOCAL_PATH
```

### Example `.env` — local file only

```bash
ARTIST_MODEL_LOCAL_PATH=models/knn_baseline_model_test2.pkl
```

### ML vs API variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `ARTIST_MODEL_LOCAL_FILENAME` | **`ml/artist/`** only | Name of file written by `train_local` (`models/`, `ml/outputs/`) |
| `ARTIST_MODEL_LOCAL_PATH` | **`app/`** only | Explicit path for the API to load |
| `MODEL_BUCKET_NAME` | **`app/`** + **`ml/`** | Shared GCS bucket (all models) |
| `ARTIST_MODEL_BLOB_NAME` | **`app/`** (download) + **`ml/artist/`** (upload) | Artist KNN object path |

Train and upload first:

```bash
python -m ml.artist.scripts.train_local
python -m ml.artist.scripts.upload_artist
```

See [ml/README_ML.md](../ml/README_ML.md).

## GCP auth (local + GCS)

`.env` tells the API **where** the blob is; it does **not** authenticate to Google.

For local runs that download from GCS:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS   # if pointing to a Le Wagon key in ~/.zshrc
gcloud config set project rec-o-gcp
gcloud auth application-default login
```

On **Cloud Run**, the service account is configured automatically (no `gcloud login`).  
`MODEL_BUCKET_NAME` and `ARTIST_MODEL_BLOB_NAME` come from **Secret Manager** (mounted at deploy) — update the artist blob secret and deploy a new revision to switch models without rebuilding the image.

## Check which model is active

```bash
curl http://localhost:8000/model
```

**Local:**

```json
{
  "loaded": true,
  "source": "local",
  "path": "/home/.../rec_o/models/knn_baseline_model_test2.pkl",
  "filename": "knn_baseline_model_test2.pkl",
  "gcs_uri": null
}
```

**GCS:**

```json
{
  "loaded": true,
  "source": "gcs",
  "path": "/tmp/knn_baseline_model_test2.pkl",
  "filename": "knn_baseline_model_test2.pkl",
  "gcs_uri": "gs://rec-o-models/models/knn_baseline_model_test2.pkl"
}
```

- `gcs_uri` = source of truth on the cloud  
- `path` = local file used to `joblib.load` (disk path or temp cache)

If `loaded` is `false`, check logs at startup, blob name, and GCP credentials.

## Docker

```bash
docker build -t rec-o .
docker run --name rec-o-api -p 8000:8000 --env-file .env rec-o
```

The image contains **only** `app/` + Python deps. No `models/`, no `ml/`, no `.pkl`.  
With `MODEL_BUCKET_NAME` + `ARTIST_MODEL_BLOB_NAME` in `--env-file`, the container downloads from GCS at startup (needs valid credentials in the environment or a service account on Cloud Run).

## Endpoints

| Method | Path | Auth | Role |
|--------|------|------|------|
| GET | `/` | No | Health check |
| GET | `/model` | No | Loaded model metadata (`source`, `path`, `gcs_uri`) |
| POST | `/predict/artist` | Yes | KNN artist recommendations |
| POST | `/predict/album` | Yes | Mock (not wired to ML yet) |
| POST | `/search/album` | Yes | Partial album title search |
| POST | `/search/artist` | Yes | Partial artist name search |
| POST | `/search/genre` | Yes | Partial genre name search |

`/predict/artist` flow:

1. `predict_playlist()` — KNN on loaded artifact  
2. `predict_artist()` — fetch names, genres, URLs from PostgreSQL for recommended IDs  

## Predictor module (`predictor.py`)

Responsibilities:

- **`load_model()`** — resolve local path or GCS download, `joblib.load`, set `model_load_info`
- **`get_model_info()`** — expose load metadata for `GET /model`
- **`predict_playlist()`** — nearest-neighbor artist IDs from seed artist IDs
- **`predict_artist()`** — SQL enrichment for the API response (not used for training)

Training, artifact save, and GCS upload are **not** in this file.

## Related docs

- [README.md](../README.md) — project setup, quick start, GCP overview  
- [ml/README_ML.md](../ml/README_ML.md) — train, upload to GCS, upload 403 troubleshooting  
- [GCP_SETUP_STEPS.md](../GCP_SETUP_STEPS.md) — Cloud Run deploy
