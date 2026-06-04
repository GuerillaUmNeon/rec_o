#!/usr/bin/env python3
"""
Train KNN and save artifacts locally only (no GCS).

Run from project root:
  python -m ml.scripts.run_local
  python -m ml.scripts.run_local --limit 5000 --skip-extended-genres --use-cache
"""

import argparse

from app.database import get_connection
from ml.artifact import save_artifact
from ml.data import fetch_artist_training_data
from ml.train import clean_training_frame, train_knn_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KNN and save locally.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max artists to load (faster). Overrides ML_MAX_ARTISTS from .env.",
    )
    parser.add_argument(
        "--skip-extended-genres",
        action="store_true",
        help="Skip the heavy 2nd SQL query (release/recording genres).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse ml/outputs/training_features.pkl if present.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-fetch SQL even when --use-cache is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with get_connection() as conn:
        raw_df = fetch_artist_training_data(
            conn,
            max_artists=args.limit,
            skip_extended_genres=args.skip_extended_genres,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
        )

    if raw_df.empty:
        raise SystemExit("No training rows returned from the database.")

    df_clean = clean_training_frame(raw_df)
    print(f"Training rows after cleaning: {len(df_clean):,}")

    artifact = train_knn_artifact(df_clean)
    save_artifact(artifact)
    print("Done. Upload to GCS manually: python -m ml.scripts.upload_to_gcs")


if __name__ == "__main__":
    main()
