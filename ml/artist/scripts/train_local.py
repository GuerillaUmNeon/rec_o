#!/usr/bin/env python3
"""
Train artist KNN and save artifacts locally (no GCS).

Run from project root:
  python -m ml.artist.scripts.train_local
  python -m ml.artist.scripts.train_local --limit 5000 --skip-extended-genres --use-cache
"""

import argparse

from app.database import engine
from ml.artist.artifact import save_artist_knn_artifact
from ml.artist.data import (
    fetch_artist_knn_training_data,
    fetch_artist_knn_training_data_scoped,
)
from ml.artist.train import build_artist_knn_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train artist KNN and save locally.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max artists to load (faster). Overrides ARTIST_ML_MAX_ARTISTS / ML_MAX_ARTISTS from .env.",
    )
    parser.add_argument(
        "--skip-extended-genres",
        action="store_true",
        help="Skip the heavy 2nd SQL query (release/recording genres).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse ml/outputs/artist_training_features.pkl if present.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-fetch SQL even when --use-cache is set.",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=1,
        help="Minimum MusicBrainz tag count kept in training SQL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_scoped_fetch = args.limit is not None or args.skip_extended_genres

    with engine.connect() as conn:
        if use_scoped_fetch:
            raw_df = fetch_artist_knn_training_data_scoped(
                conn,
                max_artists=args.limit,
                skip_extended_genres=args.skip_extended_genres,
                use_cache=args.use_cache,
                refresh_cache=args.refresh_cache,
                min_tag_count=args.min_tag_count,
            )
        else:
            raw_df = fetch_artist_knn_training_data(
                conn,
                use_cache=args.use_cache,
                refresh_cache=args.refresh_cache,
                min_tag_count=args.min_tag_count,
            )

    if raw_df.empty:
        raise SystemExit("No artist training rows returned from the database.")

    print(f"Artist training rows: {len(raw_df):,}")
    artifact = build_artist_knn_artifact(raw_df)
    print(f"Artifact rows (with genres): {len(artifact['data']):,}")

    save_artist_knn_artifact(artifact)
    print("Done. Upload to GCS manually: python -m ml.artist.scripts.upload_artist")


if __name__ == "__main__":
    main()
