# API (`app/`)

FastAPI service for artist recommendations (KNN) and MusicBrainz search.  
**Training and GCS upload live in [`ml/`](../ml/README_ML.md)** — this package only **loads** a pre-trained artifact and serves HTTP routes.

## Layout

```
app/
  main.py              # FastAPI routes, auth, lifespan → load_models()
  models/              # Registry: load_models(), get_models_info()
    loader.py          # Shared GCS / local path resolution
  artist/              # Artist KNN inference (only model wired today)
    config.py          # ARTIST_MODEL_* env vars
    loader.py          # load_artist_model(), get_artist_model_info()
    recommender.py     # recommend_artist_ids()
    enrichment.py      # enrich_artists_from_db() — SQL response enrichment
  release_group/       # Stub (future album KNN)
  genre/               # Stub (future genre KNN)
  predictor.py         # Deprecated re-exports — use app.artist / app.models
  database.py          # PostgreSQL connection
  queries.py           # SQL for search endpoints
  schemas.py           # Pydantic models
```

There is **no** training code here — offline training lives in `ml/artist/`.

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

Recommender artifacts are loaded **at startup** via FastAPI **lifespan** (`main.py` → `load_models()` in `app/models/`). Only **artist** is implemented today. Artifacts are **not** baked into the Docker image.

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

```json
{
  "artist": {
    "loaded": true,
    "source": "gcs",
    "path": "/tmp/knn_baseline_model_test2.pkl",
    "filename": "knn_baseline_model_test2.pkl",
    "gcs_uri": "gs://rec-o-models/models/knn_baseline_model_test2.pkl"
  },
  "release_group": { "loaded": false, "source": null, "path": null, "filename": null, "gcs_uri": null },
  "genre": { "loaded": false, "source": null, "path": null, "filename": null, "gcs_uri": null }
}
```

If `artist.loaded` is `false`, check logs at startup, blob name, and GCP credentials.

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
| GET | `/model` | No | Load status per model type (`artist`, `release_group`, `genre`) |
| POST | `/predict/artist` | Yes | KNN artist recommendations |
| POST | `/predict/album` | Yes | Mock (not wired to ML yet) |
| POST | `/search/album` | Yes | Partial album title search |
| POST | `/search/artist` | Yes | Partial artist name search |
| POST | `/search/genre` | Yes | Partial genre name search |

`/predict/artist` flow:

1. `recommend_artist_ids()` — KNN on loaded artist artifact  
2. `enrich_artists_from_db()` — fetch names, genres, URLs from PostgreSQL  

## Artist module (`app/artist/`)

| Function | Role |
|----------|------|
| `load_artist_model()` | Local path or GCS download → `joblib.load` |
| `get_artist_model_info()` | Metadata for `GET /model` → `artist` key |
| `recommend_artist_ids()` | KNN nearest-neighbor artist IDs |
| `enrich_artists_from_db()` | SQL enrichment for HTTP response |

## Related docs

- [README.md](../README.md) — project setup, quick start, GCP overview  
- [ml/README_ML.md](../ml/README_ML.md) — train, upload to GCS, upload 403 troubleshooting  
- [GCP_SETUP_STEPS.md](../GCP_SETUP_STEPS.md) — Cloud Run deploy
