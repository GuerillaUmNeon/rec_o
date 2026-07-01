# API (`app/`)

FastAPI service for artist and album (release group) recommendations (KNN) and MusicBrainz search.  
**Training lives in [`ml/`](../ml/README_ML.md)** — this package only **loads** a pre-trained artifact and serves HTTP routes.

## Layout

```
app/
  main.py              # FastAPI routes, auth, lifespan → load_models()
  models/              # Registry: load_models(), get_models_info()
    loader.py          # Shared local path resolution
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

### Configuration (`.env`)

| Variable(s) | Purpose |
|-------------|-------------|
| `ARTIST_MODEL_LOCAL_PATH`, `RELEASE_GROUP_MODEL_LOCAL_PATH` | Explicit path to a `.pkl` on disk |

Local resolution order:

1. `ARTIST_MODEL_LOCAL_PATH` / `RELEASE_GROUP_MODEL_LOCAL_PATH` (explicit)
2. `models/<ARTIST_MODEL_LOCAL_FILENAME>` / `models/<RELEASE_GROUP_MODEL_LOCAL_FILENAME>` if the file exists (dev default after `train_local`)

Legacy `MODEL_LOCAL_PATH` still works as a fallback for **artist** only.

### Example `.env`

```bash
ARTIST_MODEL_LOCAL_PATH=models/knn_model_test_joris_slim.pkl
RELEASE_GROUP_MODEL_LOCAL_PATH=models/release_group_knn_model_test.pkl
```

### ML vs API variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `ARTIST_MODEL_LOCAL_FILENAME` | **`ml/artist/`** only | Name of file written by `train_local` (`models/`) |
| `ARTIST_MODEL_LOCAL_PATH` | **`app/`** only | Explicit path for the API to load |
| `RELEASE_GROUP_MODEL_LOCAL_PATH` | **`app/`** only | Explicit path for release group API load |

Train first:

```bash
python -m ml.artist.scripts.train_local
python -m ml.release_group.scripts.train_local --limit 5000 --skip-type-inference --use-cache
```

See [ml/README_ML.md](../ml/README_ML.md).

## Check which model is active

```bash
curl http://localhost:8000/model
```

```json
{
  "artist": {
    "loaded": true,
    "source": "local",
    "path": "models/knn_model_test_joris_slim.pkl",
    "filename": "knn_model_test_joris_slim.pkl"
  },
  "release_group": {
    "loaded": true,
    "source": "local",
    "path": "models/release_group_knn_model_test.pkl",
    "filename": "release_group_knn_model_test.pkl"
  },
  "genre": { "loaded": false, "source": null, "path": null, "filename": null }
}
```

If `artist.loaded` or `release_group.loaded` is `false`, check logs at startup and model file paths.

## Docker

```bash
docker build -t rec-o .
docker run --name rec-o-api -p 8000:8000 --env-file .env rec-o
```

The image contains `app/` + a **minimal** `ml/release_group/features.py` stub (so `joblib` can unpickle trained artifacts) + Python deps. No `models/`, no full `ml/` training code, no `.pkl` baked in.  
Mount or copy model files into the container and set `ARTIST_MODEL_LOCAL_PATH` / `RELEASE_GROUP_MODEL_LOCAL_PATH` in `--env-file`.

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
| `load_artist_model()` | Local path → `joblib.load` |
| `get_artist_model_info()` | Metadata for `GET /model` → `artist` key |
| `recommend_artist_ids()` | KNN nearest-neighbor artist IDs |
| `enrich_artists_from_db()` | SQL enrichment for HTTP response |

## Release group module (`app/release_group/`)

| Function | Role |
|----------|------|
| `load_release_group_model()` | Local path → `joblib.load` |
| `get_release_group_model_info()` | Metadata for `GET /model` → `release_group` key |
| `recommend_release_group_ids()` | KNN nearest-neighbor release group IDs |
| `enrich_release_groups_from_db()` | SQL enrichment (title, genres, URLs, tracks) |

`/predict/album` flow:

1. `recommend_release_group_ids()` — KNN on loaded release group artifact  
2. `enrich_release_groups_from_db()` — fetch metadata from PostgreSQL  

## Related docs

- [README.md](../README.md) — project setup, quick start  
- [ml/README_ML.md](../ml/README_ML.md) — train models
