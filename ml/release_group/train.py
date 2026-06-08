"""Train KNN release group recommender (sklearn Pipeline + sparse features)."""

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import random
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

    print(data)
    n_neighbors = n_neighbors or DEFAULT_N_NEIGHBORS
    list_cols = ["tag_ids", "genre_ids", "secondary_type_ids"]
    categorical_cols = ["type", "status", "language", "script"]
    numeric_mean_cols = ["year"]
    exclude_cols = ["id", "artist_credit", "tag", "count"]

    list_cols = [c for c in list_cols if c in data.columns]
    categorical_cols = [c for c in categorical_cols if c in data.columns]
    numeric_mean_cols = [c for c in numeric_mean_cols if c in data.columns]

    scalar_feature_cols = [
        c for c in data.columns
        if c not in exclude_cols + list_cols
    ]

    preprocessor = ListToSparseTransformer(
        categorical_cols=categorical_cols,
        numeric_mean_cols=numeric_mean_cols,
        list_cols=list_cols
    )

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute"
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("knn", knn)
    ])

    pipeline.fit(data)

    id_to_idx = {row_id: idx for idx, row_id in enumerate(data["id"])}

    return {
        "model_kind": "release_group_knn",
        "pipeline": pipeline,
        "data_model": data,
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
    seed: int = 42
) -> list[int]:
    """Notebook-style inference helper for tests and future app/release_group."""
    pipeline = artifact["pipeline"]
    data_model = artifact["data_model"]
    id_to_idx = artifact["id_to_idx"]

    knn = pipeline.named_steps["knn"]
    preprocessor = pipeline.named_steps["preprocess"]

    if seed is not None:
        random.seed(seed)

    neighbor_lists: list[list[int]] = []
    for target_id in release_group_ids:
        if target_id not in id_to_idx:
            print(f"Warning: ID {target_id} not found, skipping.")
            continue

        row_idx = id_to_idx[target_id]
        row_data = data_model.iloc[row_idx:row_idx+1]

        # Transform only through preprocessor, then query KNN
        X_row = preprocessor.transform(row_data)
        distances, indices = knn.kneighbors(X_row, n_neighbors=top_n)

        neighbors = data_model.iloc[indices[0]].copy()
        neighbors["distance"] = distances[0]
        neighbors["query_id"] = target_id

        if exclude_seed:
            neighbors = neighbors[neighbors["id"] != target_id]

        neighbor_lists.append(neighbors)

    if not neighbor_lists:
        return []

    # Merge results from multiple seed IDs (first-seen order)
    # merged: list[int] = []
    # seen = set(seed_ids) if exclude_seed else set()
    # for batch in neighbor_lists:
    #     for rg_id in batch:
    #         if rg_id in seen:
    #             continue
    #         seen.add(rg_id)
    #         merged.append(rg_id)
    #         if len(merged) >= top_n:
    #             return merged
    # return merged

    result = pd.concat(neighbor_lists, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    return result
