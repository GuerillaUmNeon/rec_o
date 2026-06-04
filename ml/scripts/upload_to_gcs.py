#!/usr/bin/env python3
"""
Upload the local trained model to GCS (manual step after run_local).

Run from project root:
  python -m ml.scripts.upload_to_gcs
  python -m ml.scripts.upload_to_gcs --path ml/outputs/knn_baseline_model.pkl

Requires MODEL_BUCKET_NAME / MODEL_BLOB_NAME in .env and valid GCP credentials
(gcloud auth application-default login on project rec-o-gcp).
"""

import argparse

from ml.gcs_upload import resolve_model_path, upload_model_to_gcs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local model artifact to GCS.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to .pkl (default: MODEL_LOCAL_FILENAME at project root or ml/outputs/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.path)
    print(f"Uploading {model_path}...")
    upload_model_to_gcs(model_path)


if __name__ == "__main__":
    main()
