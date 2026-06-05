"""Train KNN release group recommender (sklearn Pipeline + sparse features)."""

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from ml.release_group.config import DEFAULT_N_NEIGHBORS
from ml.release_group.features import ListToSparseTransformer


def build_release_group_knn_artifact(
    data: pd.DataFrame,
    n_neighbors: int | None = None,
) -> dict:
    """
    Train release group KNN from a prepared DataFrame.

    Artifact format matches note_book_guillaume.ipynb bundle keys.
    """
    n_neighbors = n_neighbors or DEFAULT_N_NEIGHBORS
    data_model = data.copy()

    list_cols = ["tag_ids", "genre_ids", "secondary_type_ids"]
    categorical_cols = ["type", "status", "language", "script"]
    numeric_mean_cols = ["year"]
    exclude_cols = ["id", "artist_credit", "tag", "count"]

    list_cols = [c for c in list_cols if c in data_model.columns]
    categorical_cols = [c for c in categorical_cols if c in data_model.columns]
    numeric_mean_cols = [c for c in numeric_mean_cols if c in data_model.columns]

    scalar_feature_cols = [
        c for c in data_model.columns if c not in exclude_cols + list_cols
    ]

    preprocessor = ListToSparseTransformer(
        categorical_cols=categorical_cols,
        numeric_mean_cols=numeric_mean_cols,
        list_cols=list_cols,
    )

    knn = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(data_model)),
        metric="cosine",
        algorithm="brute",
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("knn", knn),
    ])
    pipeline.fit(data_model)

    id_to_idx = {int(row_id): idx for idx, row_id in enumerate(data_model["id"])}

    return {
        "model_kind": "release_group_knn",
        "pipeline": pipeline,
        "data_model": data_model,
        "scalar_feature_cols": scalar_feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_mean_cols": numeric_mean_cols,
        "list_cols": list_cols,
        "id_to_idx": id_to_idx,
        "n_neighbors": n_neighbors,
    }


def recommend_release_group_ids_from_artifact(
    artifact: dict,
    release_group_ids: list[int],
    top_n: int = 5,
    *,
    exclude_seed: bool = True,
) -> list[int]:
    """Notebook-style inference helper for tests and future app/release_group."""
    pipeline = artifact["pipeline"]
    data_model = artifact["data_model"]
    id_to_idx = artifact["id_to_idx"]

    knn = pipeline.named_steps["knn"]
    preprocessor = pipeline.named_steps["preprocess"]
    seed_ids = {int(rg_id) for rg_id in release_group_ids}

    neighbor_lists: list[list[int]] = []
    for target_id in seed_ids:
        if target_id not in id_to_idx:
            raise RuntimeError(f"Unknown release group ID: {target_id}")

        row_idx = id_to_idx[target_id]
        row_data = data_model.iloc[row_idx : row_idx + 1]
        X_row = preprocessor.transform(row_data)
        distances, indices = knn.kneighbors(
            X_row,
            n_neighbors=min(len(data_model), top_n + len(seed_ids) + 20),
        )

        batch: list[int] = []
        for idx in indices[0]:
            rg_id = int(data_model.iloc[idx]["id"])
            if exclude_seed and rg_id in seed_ids:
                continue
            if rg_id in batch:
                continue
            batch.append(rg_id)
            if len(batch) >= top_n:
                break
        neighbor_lists.append(batch)

    if not neighbor_lists:
        return []

    # Merge results from multiple seed IDs (first-seen order)
    merged: list[int] = []
    seen = set(seed_ids) if exclude_seed else set()
    for batch in neighbor_lists:
        for rg_id in batch:
            if rg_id in seen:
                continue
            seen.add(rg_id)
            merged.append(rg_id)
            if len(merged) >= top_n:
                return merged
    return merged
