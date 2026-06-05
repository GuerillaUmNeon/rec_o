#!/usr/bin/env python3
"""
Upload the local artist KNN model to GCS (manual step after train_local).

Run from project root:
  python -m ml.artist.scripts.upload_artist
  python -m ml.artist.scripts.upload_artist --path ml/outputs/knn_model_test_joris_slim.pkl

Requires MODEL_BUCKET_NAME and ARTIST_MODEL_BLOB_NAME in .env (legacy MODEL_BLOB_NAME still works)
and valid GCP credentials (gcloud auth application-default login on project rec-o-gcp).
"""

import argparse

from ml.artist.gcs_upload import (
    resolve_artist_knn_model_path,
    upload_artist_knn_model_to_gcs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload artist KNN artifact to GCS.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to .pkl (default: models/ then ml/outputs/ ARTIST_MODEL_LOCAL_FILENAME).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_artist_knn_model_path(args.path)
    print(f"Uploading {model_path}...")
    upload_artist_knn_model_to_gcs(model_path)


if __name__ == "__main__":
    main()
