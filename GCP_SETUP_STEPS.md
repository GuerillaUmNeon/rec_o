# GCP setup steps

## Step 1 — GCP project
Create project `rec-o-gcp`, enable billing, select it in the console.

## Step 2 — Enable APIs
Cloud Build, Artifact Registry, Cloud Run, Secret Manager, Cloud Storage.

## Step 3 — Artifact Registry
Create Docker repo `rec-o` in `europe-west1`.

## Step 4 — Cloud Storage
Create bucket `rec-o-models` in `europe-west1` and upload the final notebook artifact to `models/knn_baseline_model.pkl`.

The artifact must contain:
```python
{
    "vectorizer": vectorizer,
    "model": knn_model,
    "artist_names": artist_names,
    "data": df_clean,
}
```

Give the Cloud Run runtime service account `Storage Object Viewer` on the bucket.

## Step 5 — Secret Manager
Create one secret per env var (secret **name** = variable name, value = same as local `.env`):

| Secret | Used for |
|--------|----------|
| `TOKEN_API_KEY` | API auth |
| `POSTGRES` | DB host |
| `DATABASE` | DB name |
| `DB_USERNAME` | DB user |
| `DB_PASSWORD` | DB password |
| `DB_PORT` | DB port (e.g. `5432`) |
| `DATABASE_URL` | Optional full URL; if set at runtime, overrides `POSTGRES` + `DATABASE` + credentials |

Create `DATABASE_URL` in Secret Manager when you switch to a single connection string (placeholder value is fine until then).

## Step 6 — `cloudbuild.yaml`
Pipeline: build image → push → deploy Cloud Run `rec-o-api` with `--set-secrets` for all DB/API secrets (injected at **runtime**, not baked into the Docker image). `MODEL_BUCKET_NAME` / `MODEL_BLOB_NAME` stay as plain env vars. `options.logging: CLOUD_LOGGING_ONLY`. Commit and push to `main`.

## Step 7 — Link GitHub
Cloud Build → Repositories → host connection → Connect GitHub (repo owner authorizes Cloud Build app on `rec_o`) → Link `GuerillaUmNeon/rec_o`.

## Step 8 — Trigger
Create `deploy-main`: push to `^main$`, config `cloudbuild.yaml`, custom service account, logging **Cloud Logging only**. Fix IAM: build SA (Run Admin, Artifact Registry Writer, Secret Accessor); runtime SA `...@compute` needs **Secret Accessor** on every secret listed above (`TOKEN_API_KEY`, `POSTGRES`, etc.).

## Step 9 — Deploy & test
Push to `main` or Run trigger → build Success → URL in Cloud Run → `rec-o-api`.
