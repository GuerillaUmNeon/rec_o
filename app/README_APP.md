# API (`app/`)

FastAPI service for artist and album (release group) recommendations (KNN) and MusicBrainz search.  
**Training and GCS upload live in [`ml/`](../ml/README_ML.md)** — this package only **loads** a pre-trained artifact and serves HTTP routes.

## Layout

```
app/
  main.py              # FastAPI routes, auth, lifespan → load_models()
  models/              # Registry: load_models(), get_models_info()
    loader.py          # Shared GCS / local path resolution
  artist/              # Artist KNN inference
    config.py          # ARTIST_MODEL_* env vars
    loader.py          # load_artist_model(), get_artist_model_info()
    recommender.py     # recommend_artist_ids()
    enrichment.py      # enrich_artists_from_db() — SQL response enrichment
  release_group/       # Release group / album KNN inference
    config.py          # RELEASE_GROUP_MODEL_* env vars
    loader.py          # load_release_group_model(), get_release_group_model_info()
    recommender.py     # recommend_release_group_ids()
    enrichment.py      # enrich_release_groups_from_db() — SQL response enrichment
  genre/               # Stub (future genre KNN)
  predictor.py         # Deprecated re-exports — use app.artist / app.models
  db_models.py         # DB table/ORM models (not ML — see app/models/)
  database.py          # PostgreSQL connection
  queries.py           # SQL for search endpoints
  schemas.py           # Pydantic models
```

There is **no** training code here — offline training lives in `ml/artist/` and `ml/release_group/`.

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

Recommender artifacts are loaded **at startup** via FastAPI **lifespan** (`main.py` → `load_models()` in `app/models/`). **Artist** and **release_group** are wired today. Artifacts are **not** baked into the Docker image.

### Two sources (`.env`)

| Priority | Variable(s) | When to use |
|----------|-------------|-------------|
| **1 — Local file** | `ARTIST_MODEL_LOCAL_PATH`, `RELEASE_GROUP_MODEL_LOCAL_PATH` | Dev: test a `.pkl` on disk without GCS |
| **2 — GCS** | `MODEL_BUCKET_NAME` + `ARTIST_MODEL_BLOB_NAME` / `RELEASE_GROUP_MODEL_BLOB_NAME` | Same as prod / Docker / Cloud Run |

If a local file is resolved, GCS is **not** used for that model (even if `MODEL_BUCKET_NAME` is set).

Local resolution order:

1. `ARTIST_MODEL_LOCAL_PATH` / `RELEASE_GROUP_MODEL_LOCAL_PATH` (explicit)
2. `models/<ARTIST_MODEL_LOCAL_FILENAME>` / `models/<RELEASE_GROUP_MODEL_LOCAL_FILENAME>` if the file exists (dev default after `train_local`)
3. GCS download from `*_MODEL_BLOB_NAME`

Legacy `MODEL_LOCAL_PATH` / `MODEL_BLOB_NAME` still work as fallbacks for **artist** only.

If `*_MODEL_LOCAL_PATH` is missing for a model, the API downloads from GCS to a **temp cache**:

```
gs://<MODEL_BUCKET_NAME>/<ARTIST_MODEL_BLOB_NAME>
    →  /tmp/<filename>.pkl
gs://<MODEL_BUCKET_NAME>/<RELEASE_GROUP_MODEL_BLOB_NAME>
    →  /tmp/<filename>.pkl
```

That `/tmp/...` path is **runtime only** (not in the Docker image; gone when the container stops).

### Example `.env` — GCS (like prod)

```bash
MODEL_BUCKET_NAME=rec-o-models
ARTIST_MODEL_BLOB_NAME=models/knn_model_test_joris_slim.pkl
RELEASE_GROUP_MODEL_BLOB_NAME=models/release_group_knn_model_test.pkl
# no *_MODEL_LOCAL_PATH
```

### Example `.env` — local file only

```bash
ARTIST_MODEL_LOCAL_PATH=models/knn_model_test_joris_slim.pkl
RELEASE_GROUP_MODEL_LOCAL_PATH=models/release_group_knn_model_test.pkl
```

### ML vs API variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `ARTIST_MODEL_LOCAL_FILENAME` | **`ml/artist/`** only | Name of file written by `train_local` (`models/`, `ml/outputs/`) |
| `ARTIST_MODEL_LOCAL_PATH` | **`app/`** only | Explicit path for the API to load |
| `MODEL_BUCKET_NAME` | **`app/`** + **`ml/`** | Shared GCS bucket (all models) |
| `ARTIST_MODEL_BLOB_NAME` | **`app/`** (download) + **`ml/artist/`** (upload) | Artist KNN object path |
| `RELEASE_GROUP_MODEL_LOCAL_PATH` | **`app/`** only | Explicit path for release group API load |
| `RELEASE_GROUP_MODEL_BLOB_NAME` | **`app/`** (download) + **`ml/release_group/`** (upload) | Release group KNN object path |

Train and upload first:

```bash
python -m ml.artist.scripts.train_local
python -m ml.artist.scripts.upload_artist

python -m ml.release_group.scripts.train_local --limit 5000 --skip-type-inference --use-cache
python -m ml.release_group.scripts.upload_release_group
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
`MODEL_BUCKET_NAME`, `ARTIST_MODEL_BLOB_NAME`, and `RELEASE_GROUP_MODEL_BLOB_NAME` come from **Secret Manager** (mounted at deploy) — update a blob secret and deploy a new revision to switch models without rebuilding the image.

## Check which model is active

```bash
curl http://localhost:8000/model
```

```json
{
  "artist": {
    "loaded": true,
    "source": "gcs",
    "path": "/tmp/knn_model_test_joris_slim.pkl",
    "filename": "knn_model_test_joris_slim.pkl",
    "gcs_uri": "gs://rec-o-models/models/knn_model_test_joris_slim.pkl"
  },
  "release_group": {
    "loaded": true,
    "source": "local",
    "path": "models/release_group_knn_model_test.pkl",
    "filename": "release_group_knn_model_test.pkl",
    "gcs_uri": null
  },
  "genre": { "loaded": false, "source": null, "path": null, "filename": null, "gcs_uri": null }
}
```

If `artist.loaded` or `release_group.loaded` is `false`, check logs at startup, blob names, and GCP credentials.

## Docker

```bash
docker build -t rec-o .
docker run --name rec-o-api -p 8000:8000 --env-file .env rec-o
```

The image contains `app/` + a **minimal** `ml/release_group/features.py` stub (so `joblib` can unpickle trained artifacts) + Python deps. No `models/`, no full `ml/` training code, no `.pkl` baked in.  
With `MODEL_BUCKET_NAME` + `ARTIST_MODEL_BLOB_NAME` + `RELEASE_GROUP_MODEL_BLOB_NAME` in `--env-file`, the container downloads from GCS at startup (needs valid credentials in the environment or a service account on Cloud Run).

## Endpoints

| Method | Path | Auth | Role |
|--------|------|------|------|
| GET | `/` | No | Health check |
| GET | `/model` | No | Load status per model type (`artist`, `release_group`, `genre`) |
| POST | `/predict/artist` | Yes | KNN artist recommendations |
| POST | `/predict/album` | Yes | KNN release group (album) recommendations |
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

## Release group module (`app/release_group/`)

| Function | Role |
|----------|------|
| `load_release_group_model()` | Local path or GCS download → `joblib.load` |
| `get_release_group_model_info()` | Metadata for `GET /model` → `release_group` key |
| `recommend_release_group_ids()` | KNN nearest-neighbor release group IDs |
| `enrich_release_groups_from_db()` | SQL enrichment (title, genres, URLs, tracks) |

`/predict/album` flow:

1. `recommend_release_group_ids()` — KNN on loaded release group artifact  
2. `enrich_release_groups_from_db()` — fetch metadata from PostgreSQL  

## Related docs

- [README.md](../README.md) — project setup, quick start, GCP overview  
- [ml/README_ML.md](../ml/README_ML.md) — train, upload to GCS, upload 403 troubleshooting  
- [GCP_SETUP_STEPS.md](../GCP_SETUP_STEPS.md) — Cloud Run deploy
