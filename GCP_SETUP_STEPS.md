# GCP setup steps

## Step 1 — GCP project
Create project `rec-o-gcp`, enable billing, select it in the console.

## Step 2 — Enable APIs
Cloud Build, Artifact Registry, Cloud Run, Secret Manager.

## Step 3 — Artifact Registry
Create Docker repo `rec-o` in `europe-west1`.

## Step 4 — Secret Manager
Create secret `TOKEN_API_KEY` with the API token value.

## Step 5 — `cloudbuild.yaml`
Add pipeline at repo root: build → push image → deploy Cloud Run `rec-o-api` (port 8000, secret `TOKEN_API_KEY`). Add `options.logging: CLOUD_LOGGING_ONLY`. Commit and push to `main`.

## Step 6 — Link GitHub
Cloud Build → Repositories → host connection → Connect GitHub (repo owner authorizes Cloud Build app on `rec_o`) → Link `GuerillaUmNeon/rec_o`.

## Step 7 — Trigger
Create `deploy-main`: push to `^main$`, config `cloudbuild.yaml`, custom service account, logging **Cloud Logging only**. Fix IAM: build SA (Run Admin, Artifact Registry Writer, Secret Accessor); runtime SA `...@compute` needs Secret Accessor on `TOKEN_API_KEY`.

## Step 8 — Deploy & test
Push to `main` or Run trigger → build Success → URL in Cloud Run → `rec-o-api`.
