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
Create secret `TOKEN_API_KEY` with the API token value.

## Step 6 — `cloudbuild.yaml`
Add pipeline at repo root: build → push image → deploy Cloud Run `rec-o-api` (port 8000, secret `TOKEN_API_KEY`, env var `MODEL_BUCKET_NAME`). Add `options.logging: CLOUD_LOGGING_ONLY`. Commit and push to `main`.

## Step 7 — Link GitHub
Cloud Build → Repositories → host connection → Connect GitHub (repo owner authorizes Cloud Build app on `rec_o`) → Link `GuerillaUmNeon/rec_o`.

## Step 8 — Trigger
Create `deploy-main`: push to `^main$`, config `cloudbuild.yaml`, custom service account, logging **Cloud Logging only**. Fix IAM: build SA (Run Admin, Artifact Registry Writer, Secret Accessor); runtime SA `...@compute` needs Secret Accessor on `TOKEN_API_KEY`.

## Step 9 — Deploy & test
Push to `main` or Run trigger → build Success → URL in Cloud Run → `rec-o-api`.
