#!/usr/bin/env python3
"""
Upload the local release group KNN model to GCS (manual step after train_local).

Run from project root:
  python -m ml.release_group.scripts.upload_release_group
  python -m ml.release_group.scripts.upload_release_group --path models/release_group_knn_model.pkl
"""

import argparse

from ml.release_group.gcs_upload import (
    resolve_release_group_knn_model_path,
    upload_release_group_knn_model_to_gcs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload release group KNN artifact to GCS.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to .pkl (default: models/ then ml/outputs/ RELEASE_GROUP_MODEL_LOCAL_FILENAME).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_release_group_knn_model_path(args.path)
    print(f"Uploading {model_path}...")
    upload_release_group_knn_model_to_gcs(model_path)


if __name__ == "__main__":
    main()
