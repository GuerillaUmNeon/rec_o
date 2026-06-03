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
- `MODEL_BUCKET_NAME`, `MODEL_BLOB_NAME` — Google Cloud Storage location for the recommender artifact

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
| POST | `/predict` | Yes | Artist/genre prediction (10 req/min per IP) |
| POST | `/search/album` | Yes | Partial album title search (max 20) |
| POST | `/search/artist` | Yes | Partial artist name search (max 20) |

Protected routes require the header:

```
X-API-Key: <TOKEN_API_KEY>
```

In Swagger, use **Authorize** and enter the same value.

## Save the latest model

The API loads the latest KNN recommender artifact from Google Cloud Storage when `MODEL_BUCKET_NAME` is configured.

The final notebook model must be saved as an artifact containing the fitted vectorizer, fitted KNN model, and cleaned dataframe:

```python
from app.predictor import save_model

artifact = {
    "vectorizer": vectorizer,
    "model": knn_model,
    "artist_names": artist_names,
    "data": df_clean,
}

save_model(artifact)
```

This creates a local copy in `models/`, updates `knn_baseline_model.pkl`, and uploads that file to `gs://$MODEL_BUCKET_NAME/$MODEL_BLOB_NAME`.

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

Production uses Secret Manager for `TOKEN_API_KEY`; do not commit `.env`.
