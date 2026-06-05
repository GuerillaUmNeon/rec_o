# rec_o

FastAPI service for music recommendations and MusicBrainz search (PostgreSQL).

## Project layout

```
app/          # HTTP API — inference only (see app/README_APP.md)
ml/           # Offline training + GCS upload (see ml/README_ML.md)
models/       # Local .pkl artifacts after train_local (not in Docker image)
```

| Doc | Contents |
|-----|----------|
| [app/README_APP.md](app/README_APP.md) | Run API, endpoints, model load (local / GCS), Docker, `GET /model` |
| [ml/README_ML.md](ml/README_ML.md) | Train, upload to GCS, `.env` vars, 403 troubleshooting |
| [GCP_SETUP_STEPS.md](GCP_SETUP_STEPS.md) | Cloud Run, VPC, secrets, production deploy |

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

- `TOKEN_API_KEY` — API key for protected routes
- `POSTGRES`, `DB_PORT`, `DATABASE`, `DB_USERNAME`, `DB_PASSWORD` — PostgreSQL (MusicBrainz)
- Optional: `DATABASE_URL` overrides the variables above
- Model variables — see `.env.sample` and the docs above (`MODEL_*` for API vs ML)

Generate a token:

```
python -c "import secrets; print(secrets.token_hex(32))"
```

## Quick start

**Train and publish models** (offline):

```bash
# Artist KNN
python -m ml.artist.scripts.train_local
python -m ml.artist.scripts.upload_artist

# Release group / album KNN (fast dev)
python -m ml.release_group.scripts.train_local --limit 5000 --skip-type-inference --use-cache
python -m ml.release_group.scripts.upload_release_group
```

Details: [ml/README_ML.md](ml/README_ML.md).

**Run the API** (loads artist + release_group at startup from GCS or `*_MODEL_LOCAL_PATH`):

```bash
uvicorn app.main:app --reload
```

Details: [app/README_APP.md](app/README_APP.md).

**Docker** (API only; model from GCS at runtime):

```bash
docker build -t rec-o .
docker run --name rec-o-api -p 8000:8000 --env-file .env rec-o
```

## GCP production

Deploy to Cloud Run on push to `main` (see `cloudbuild.yaml`). Full setup: [GCP_SETUP_STEPS.md](GCP_SETUP_STEPS.md).

APIs enabled for the project:

- Cloud Build API
- Artifact Registry API
- Cloud Run Admin API
- Secret Manager API
- Cloud Storage API

Production: create matching secrets in Secret Manager (`TOKEN_API_KEY`, `POSTGRES`, `DATABASE`, `DB_USERNAME`, `DB_PASSWORD`, `DB_PORT`, `DATABASE_URL`); `cloudbuild.yaml` mounts them on Cloud Run at deploy time. Do not commit `.env`.

Prod model blobs: Secret Manager `ARTIST_MODEL_BLOB_NAME` and `RELEASE_GROUP_MODEL_BLOB_NAME`. Use test blob names in local `.env` to avoid overwriting prod on GCS.
