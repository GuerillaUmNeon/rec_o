# GCP setup steps

API project: **`rec-o-gcp`** · region: **`europe-west1`** · service: **`rec-o-api`**

Postgres usually runs in **another GCP project** (public IP). Cloud Run must use a **fixed egress IP** (NAT) so the DB firewall can allow it.

---

## Step 1 — GCP project
Create project `rec-o-gcp`, enable billing, select it in the console.

## Step 2 — Enable APIs
Cloud Build, Artifact Registry, Cloud Run, Secret Manager, Cloud Storage, **Compute Engine**, **Serverless VPC Access**.

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

| Secret | Used for | Example value |
|--------|----------|---------------|
| `TOKEN_API_KEY` | API auth | (generated token) |
| `POSTGRES` | DB host | VM IP |
| `DATABASE` | DB name | `musicbrainz` |
| `DB_USERNAME` | DB user | `rec_o` |
| `DB_PASSWORD` | DB password | (password) |
| `DB_PORT` | DB port | `5432` |
| `DATABASE_URL` | Full URL; if set at runtime, overrides the vars above | `postgresql://...` |
| `MODEL_BUCKET_NAME` | GCS bucket for the recommender artifact | `rec-o-models` |
| `ARTIST_MODEL_BLOB_NAME` | Artist KNN GCS object path (change to switch model without rebuilding the image) | `models/knn_baseline_model.pkl` |

`DATABASE_URL` must be a valid `postgresql://user:pass@host:5432/db` string (no placeholder `ip` in the host).

Secrets are mounted at **Cloud Run runtime** via `cloudbuild.yaml` — not baked into the Docker image.

`MODEL_*` values are configuration (not credentials), but Secret Manager lets you change the active model path without editing `cloudbuild.yaml` or rebuilding the Docker image.

**Switch model in production:**

1. Upload the new `.pkl` to GCS (`python -m ml.artist.scripts.upload_artist`).
2. Update secret `ARTIST_MODEL_BLOB_NAME` in Secret Manager (e.g. `models/knn_baseline_model_v2.pkl`).
3. Deploy a new Cloud Run revision (re-run the Cloud Build trigger, or **Edit & deploy new revision** in the console — no image rebuild required).

## Step 6 — VPC, connector, Cloud NAT (fixed egress IP)

One-time setup in **`rec-o-gcp`** so outbound traffic to the DB uses a stable IP.

| Resource | Name |
|----------|------|
| VPC | `rec-o-vpc` |
| Subnet | `rec-o-subnet` (`10.8.0.0/24`, `europe-west1`) |
| Serverless VPC connector | `rec-o-connector` (`10.8.1.0/28`) |
| Static regional IP | `rec-o-nat-ip` |
| Router | `rec-o-router` |
| Cloud NAT | `rec-o-nat` (pool: `rec-o-nat-ip`, all subnet ranges) |

Example CLI (skip steps if resources already exist):

```bash
gcloud config set project rec-o-gcp

gcloud compute networks create rec-o-vpc --subnet-mode=custom
gcloud compute networks subnets create rec-o-subnet \
  --network=rec-o-vpc --region=europe-west1 --range=10.8.0.0/24

gcloud compute networks vpc-access connectors create rec-o-connector \
  --region=europe-west1 --network=rec-o-vpc --range=10.8.1.0/28 \
  --min-instances=2 --max-instances=3

gcloud compute addresses create rec-o-nat-ip --region=europe-west1
gcloud compute routers create rec-o-router --network=rec-o-vpc --region=europe-west1
gcloud compute routers nats create rec-o-nat \
  --router=rec-o-router --region=europe-west1 \
  --nat-external-ip-pool=rec-o-nat-ip --nat-all-subnet-ip-ranges

gcloud run services update rec-o-api \
  --region=europe-west1 \
  --vpc-connector=rec-o-connector \
  --vpc-egress=all-traffic
```

**Get the egress IP** (give this to the DB team for firewall whitelist):

```bash
gcloud compute addresses describe rec-o-nat-ip \
  --region=europe-west1 --format='get(address)'
```

Console: **VPC network** → **IP addresses** → `rec-o-nat-ip` → **External IP** (`europe-west1`).

Example: `34.140.60.217` → firewall rule source: **`34.140.60.217/32`**, TCP **5432**.

Cloud Run **ingress** URL is still `*.run.app` — that is not the IP used for DB whitelist.

## Step 7 — Database project firewall (other GCP project)

On the Postgres VM / VPC (DB project), allow **ingress** from the NAT IP only:

- Source: `NAT_IP/32` (from Step 6)
- Protocol: `tcp:5432`
- Target: Postgres VM (network tag or service account)

`DATABASE_URL` on Cloud Run still points at the DB host (e.g. `34.14.1.32`), not at the NAT IP.

## Step 8 — `cloudbuild.yaml`
Pipeline: build image → push → deploy Cloud Run `rec-o-api` with:

- `--vpc-connector=rec-o-connector` and `--vpc-egress=all-traffic` (keep NAT on every deploy)
- `--set-secrets` for all secrets in Step 5 (including `MODEL_BUCKET_NAME` and `ARTIST_MODEL_BLOB_NAME`)
- `options.logging: CLOUD_LOGGING_ONLY`

Commit and push to `main`.

## Step 9 — Link GitHub
Cloud Build → Repositories → host connection → Connect GitHub (repo owner authorizes Cloud Build app on `rec_o`) → Link `GuerillaUmNeon/rec_o`.

## Step 10 — Trigger
Create `deploy-main`: push to `^main$`, config `cloudbuild.yaml`, custom service account, logging **Cloud Logging only**.

IAM:

- **Cloud Build SA**: Run Admin, Artifact Registry Writer, Secret Accessor
- **Cloud Run runtime SA** (`...@compute`): Secret Accessor on every secret in Step 5

## Step 11 — Deploy & test

1. Push to `main` or run trigger → build success.
2. Service URL:
   ```bash
   gcloud run services describe rec-o-api --region=europe-west1 --format='value(status.url)'
   ```
3. `GET /` (no API key).
4. `POST /search/artist` with header `X-API-Key` — confirms DB + NAT + firewall.

If DB calls fail: check secret `DATABASE_URL`, NAT IP whitelisted on DB project, and Cloud Run annotations include `vpc-connector` + `vpc-egress=all-traffic`.
