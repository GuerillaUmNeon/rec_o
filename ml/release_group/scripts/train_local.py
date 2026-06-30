#!/usr/bin/env python3
"""
Train release group KNN and save artifacts locally (no GCS).

Run from project root:
  python -m ml.release_group.scripts.train_local
  python -m ml.release_group.scripts.train_local --limit 5000 --skip-type-inference --use-cache
"""

import argparse

from app.database import engine
from ml.release_group.artifact import save_release_group_knn_artifact
from ml.release_group.config import DEFAULT_N_NEIGHBORS
from ml.release_group.data import fetch_release_group_knn_training_data
from ml.release_group.train import build_release_group_knn_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train release group KNN and save locally.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max release groups (faster). Overrides RELEASE_GROUP_ML_MAX_ROWS from .env.",
    )
    parser.add_argument(
        "--skip-type-inference",
        action="store_true",
        help="Skip track-meta SQL used to infer missing release_group.type.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse ml/outputs/release_group_training_features.pkl if present.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-fetch SQL even when --use-cache is set.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_N_NEIGHBORS,
        help=f"KNN neighbors (default: {DEFAULT_N_NEIGHBORS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with engine.connect() as conn:
        raw_df = fetch_release_group_knn_training_data(
            conn,
            max_rows=args.limit,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
        )

    if raw_df.empty:
        raise SystemExit("No release group training rows returned from the database.")

    print(f"Release group training rows: {len(raw_df):,}")

    artifact = build_release_group_knn_artifact(
        raw_df,
        conn_factory=lambda: engine.connect(),
        n_neighbors=args.n_neighbors,
    )

    print(f"Artifact rows: {len(artifact['data_model']):,}")

    save_release_group_knn_artifact(artifact)

if __name__ == "__main__":
    main()
