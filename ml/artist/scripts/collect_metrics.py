#!/usr/bin/env python3
"""Collect comparable metrics for one artist recommender artifact."""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from app.artist.recommender import _recommend_artist_ids_from_artifact


DEFAULT_METRICS_PATH = Path("ml/artist_model_metrics.csv")
LATEST_JSON_PATH = Path("ml/outputs/artist_model_metrics_latest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect artist model metrics.")
    parser.add_argument(
        "--model-path",
        default="models/knn_model_test_joris_slim.pkl",
        help="Path to the .pkl/.joblib artifact to measure.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable ID for this run. Existing rows with the same ID are replaced.",
    )
    parser.add_argument(
        "--metrics-path",
        default=str(DEFAULT_METRICS_PATH),
        help="CSV table where the metric row is written.",
    )
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def _artifact_data(artifact: dict) -> pd.DataFrame:
    data = artifact.get("data")
    if data is None:
        data = artifact.get("df_clean")
    if data is None:
        raise RuntimeError("Artifact has no data dataframe.")
    return data.copy()


def _prediction_metrics(
    artifact: dict,
    df: pd.DataFrame,
    *,
    seed_count: int,
    top_n: int,
) -> dict:
    seed_rows = df[df["genres"].str.strip() != ""].head(seed_count)
    seed_ids = [int(value) for value in seed_rows["artist_id"].tolist()]

    id_to_genres = {
        int(row.artist_id): set(str(row.genres).split())
        for row in df[["artist_id", "genres"]].itertuples(index=False)
    }

    latencies = []
    recommended_lengths = []
    seed_leak_count = 0
    recommendation_duplicate_count = 0
    jaccards = []
    all_rec_genres = []

    for seed_id in seed_ids:
        t0 = time.perf_counter()
        recs = _recommend_artist_ids_from_artifact(artifact, [seed_id], top_n)
        latencies.append(time.perf_counter() - t0)
        recommended_lengths.append(len(recs))

        if seed_id in recs:
            seed_leak_count += 1
        recommendation_duplicate_count += len(recs) - len(set(recs))

        seed_genres = id_to_genres.get(seed_id, set())
        for rec_id in recs:
            rec_genres = id_to_genres.get(int(rec_id), set())
            all_rec_genres.extend(rec_genres)
            union = seed_genres | rec_genres
            if union:
                jaccards.append(len(seed_genres & rec_genres) / len(union))

    latency_series = pd.Series(latencies)
    return {
        "prediction_seed_count": len(seed_ids),
        "prediction_top_n": top_n,
        "prediction_latency_mean_ms": round(1000 * float(latency_series.mean()), 3),
        "prediction_latency_p95_ms": round(
            1000 * float(latency_series.quantile(0.95)),
            3,
        ),
        "recommended_count_mean": round(
            sum(recommended_lengths) / len(recommended_lengths),
            3,
        )
        if recommended_lengths
        else 0,
        "seed_leak_count": seed_leak_count,
        "recommendation_duplicate_count": recommendation_duplicate_count,
        "genre_jaccard_mean": round(sum(jaccards) / len(jaccards), 4)
        if jaccards
        else None,
        "recommended_unique_genres": len(set(all_rec_genres)),
    }


def collect_metrics(args: argparse.Namespace) -> dict:
    model_path = Path(args.model_path)
    run_id = args.run_id or f"{model_path.stem}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

    load_t0 = time.perf_counter()
    artifact = joblib.load(model_path)
    load_seconds = time.perf_counter() - load_t0

    if not isinstance(artifact, dict):
        raise RuntimeError("Expected a dict artifact.")

    df = _artifact_data(artifact)
    df["genres"] = df["genres"].fillna("").astype(str)
    genre_token_counts = df["genres"].str.split().apply(len)

    vectorizer = artifact.get("vectorizer")
    model = artifact.get("model")
    model_n_neighbors = getattr(model, "n_neighbors", None)

    row = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "genre_feature_format": artifact.get("genre_feature_format", ""),
        "data_rows": len(df),
        "duplicate_artist_ids": int(df.duplicated(subset=["artist_id"]).sum())
        if "artist_id" in df.columns
        else None,
        "empty_genres": int(df["genres"].str.strip().eq("").sum()),
        "nan_total": int(df.isna().sum().sum()),
        "tag_count_sum_zero_or_less": int(df["tag_count_sum"].fillna(0).le(0).sum())
        if "tag_count_sum" in df.columns
        else None,
        "vocab_size": len(getattr(vectorizer, "vocabulary_", {})),
        "model_n_neighbors": int(model_n_neighbors)
        if model_n_neighbors is not None
        else None,
        "genre_tokens_mean": round(float(genre_token_counts.mean()), 3),
        "genre_tokens_median": round(float(genre_token_counts.median()), 3),
        "genre_tokens_p95": round(float(genre_token_counts.quantile(0.95)), 3),
        "genre_tokens_max": int(genre_token_counts.max()),
        "load_seconds": round(load_seconds, 3),
        **_prediction_metrics(
            artifact,
            df,
            seed_count=args.seed_count,
            top_n=args.top_n,
        ),
        "notes": args.notes,
    }
    return row


def write_metrics(row: dict, metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if metrics_path.is_file():
        with metrics_path.open(newline="") as file:
            reader = csv.DictReader(file)
            existing_rows = [item for item in reader if item.get("run_id") != row["run_id"]]

    fieldnames = list(row.keys())
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)

    LATEST_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JSON_PATH.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    row = collect_metrics(args)
    write_metrics(row, Path(args.metrics_path))
    print(pd.DataFrame([row]).T.to_string(header=False))


if __name__ == "__main__":
    main()
