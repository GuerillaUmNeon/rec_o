# rec_o

FastAPI service for music recommendations and MusicBrainz search (PostgreSQL).

## Project layout

```
app/
  main.py       # routes, auth, rate limiting
  database.py   # PostgreSQL connection (psycopg)
  queries.py    # SQL strings
  schemas.py    # request/response models (Pydantic)
  predictor.py  # KNN prediction
```

## Set up your machine

### Virtualenv

```
cd ~/code
mkdir GuerillaUmNeon
git clone git@github.com:GuerillaUmNeon/rec_o.git
cd rec_o
pyenv install 3.13.13
pyenv virtualenv 3.13.13 rec-o-env
pyenv local rec-o-env
```

### Requirements

```
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment

Copy `.env.sample` to `.env` and fill in values:

- `TOKEN_API_KEY` — API key for protected routes (see below)
- `POSTGRES`, `DB_PORT`, `DATABASE`, `DB_USERNAME`, `DB_PASSWORD` — PostgreSQL (MusicBrainz)
- Optional: `DATABASE_URL` overrides the variables above
- `MODEL_BUCKET_NAME`, `MODEL_BLOB_NAME` — GCS bucket and object path for the recommender artifact (API + upload)
- Optional (local ML only): `MODEL_LOCAL_FILENAME` — local `.pkl` name after `run_local` (use a `*_test*` name to avoid overwriting prod files). See [ml/README_ML.md](ml/README_ML.md).

Generate a token:

```
python -c "import secrets; print(secrets.token_hex(32))"
```

## FastAPI

```
uvicorn app.main:app --reload
```

- [http://localhost:8000](http://localhost:8000) — health check (no API key)
- [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger UI

### Endpoints

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| GET | `/` | No | Health check (60 req/min per IP) |
| POST | `/predict` | Yes | Nearest artist ID prediction (10 req/min per IP) |
| POST | `/search/album` | Yes | Partial album title search (max 20) |
| POST | `/search/artist` | Yes | Partial artist name search (max 20) |

Protected routes require the header:

```
X-API-Key: <TOKEN_API_KEY>
```

In Swagger, use **Authorize** and enter the same value.

## Model training and deployment

Training lives in **`ml/`** (not `app/`). The API only **loads** a pre-trained artifact at runtime (local file or GCS).

```bash
python -m ml.scripts.run_local
python -m ml.scripts.upload_to_gcs
```

See [ml/README_ML.md](ml/README_ML.md). Cloud Run loads from `gs://$MODEL_BUCKET_NAME/$MODEL_BLOB_NAME` when `MODEL_BUCKET_NAME` is set.

### Upload to GCS — `.env` vs GCP credentials

`upload_to_gcs` reads from `.env` **where** to upload (`MODEL_BUCKET_NAME`, `MODEL_BLOB_NAME`) and **which local file** (`MODEL_LOCAL_FILENAME`). Example for a test artifact (does not overwrite prod on GCS if you keep the prod blob path on Cloud Run):

```bash
MODEL_LOCAL_FILENAME=knn_baseline_model_test.pkl
MODEL_BLOB_NAME=models/knn_baseline_model_test.pkl
```

Production Cloud Run still uses `MODEL_BLOB_NAME=models/knn_baseline_model.pkl` from `cloudbuild.yaml` until you change that deploy config.

**You cannot fix a 403 upload by putting only the GCP project name in `.env`.** The Google client library uses **Application Default Credentials** (ADC): your Google account or a service-account key file. A variable such as `GOOGLE_CLOUD_PROJECT=rec-o-gcp` does not switch away from another account (e.g. a Le Wagon service account) that lacks `storage.objects.create` on bucket `rec-o-models`.

| In `.env` | Effect |
|-----------|--------|
| `MODEL_BUCKET_NAME`, `MODEL_BLOB_NAME`, `MODEL_LOCAL_FILENAME` | Destination and local filename — yes |
| GCP project name only | No — does not change which account authenticates |
| `GOOGLE_APPLICATION_CREDENTIALS=/path/to/rec-o-sa-key.json` | Yes — if that key has **Storage Object Creator** (or Admin) on `rec-o-models` |

**Fix 403 (wrong account, e.g. `le-wagon-data-bootcamp@...`):**

```bash
echo "$GOOGLE_APPLICATION_CREDENTIALS"   # must be empty for rec-o upload
```

If this prints a path to a Le Wagon JSON (e.g. `airy-cogency-493213-t4-....json`), a **new terminal will still fail** until you remove that export — often in `~/.zshrc` or `~/.bashrc`:

```bash
grep GOOGLE_APPLICATION_CREDENTIALS ~/.zshrc ~/.bashrc
# Comment out or delete the export line, then: source ~/.zshrc
```

One-off in the current shell:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project rec-o-gcp
gcloud auth application-default login
python -m ml.scripts.upload_to_gcs
```

Expect: `Uploaded ... → gs://rec-o-models/models/knn_baseline_model_test.pkl` (or your `MODEL_BLOB_NAME`). If it still fails, ask a **rec-o-gcp** admin to grant your Google user **Storage Object Creator** on bucket `rec-o-models`.

## Docker

```
docker build -t rec-o .
docker run --name rec-o-api -p 8000:8000 --env-file .env rec-o
```

Remove the container if needed:

```
docker rm rec-o-api
```

## GCP

Deploy to Cloud Run on push to `main` (see `cloudbuild.yaml`). Full setup: [GCP_SETUP_STEPS.md](GCP_SETUP_STEPS.md).

APIs enabled for the project:

- Cloud Build API
- Artifact Registry API
- Cloud Run Admin API
- Secret Manager API
- Cloud Storage API

Production: create matching secrets in Secret Manager (`TOKEN_API_KEY`, `POSTGRES`, `DATABASE`, `DB_USERNAME`, `DB_PASSWORD`, `DB_PORT`, `DATABASE_URL`); `cloudbuild.yaml` mounts them on Cloud Run at deploy time. Do not commit `.env`.
