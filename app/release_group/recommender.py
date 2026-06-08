"""Release group KNN recommendations from a loaded artifact."""

from app.release_group.loader import get_release_group_model, load_release_group_model


def _recommend_release_group_ids_from_artifact(
    artifact: dict,
    release_group_ids: list[int],
    top_n: int,
    *,
    blacklist: list[int] | None = None,
    genre_ids: list[int] | None = None,
) -> list[int]:
    pipeline = artifact["pipeline"]
    data_model = artifact["data_model"]
    id_to_idx = artifact["id_to_idx"]

    knn = pipeline.named_steps["knn"]
    preprocessor = pipeline.named_steps["preprocess"]

    seed_ids = {int(rg_id) for rg_id in release_group_ids}
    exclude_ids = set(seed_ids)
    if blacklist:
        exclude_ids.update(int(rg_id) for rg_id in blacklist)

    missing_ids = sorted(seed_ids - set(id_to_idx))
    if missing_ids:
        raise RuntimeError(f"Unknown release group IDs: {missing_ids}")

    neighbor_lists: list[list[int]] = []
    for target_id in seed_ids:
        row_idx = id_to_idx[target_id]
        row_data = data_model.iloc[row_idx : row_idx + 1]
        X_row = preprocessor.transform(row_data)
        distances, indices = knn.kneighbors(
            X_row,
            n_neighbors=min(len(data_model), top_n + len(exclude_ids) + 50),
        )

        batch: list[int] = []
        for idx in indices[0]:
            rg_id = int(data_model.iloc[idx]["id"])
            if rg_id in exclude_ids:
                continue
            if rg_id in batch:
                continue
            batch.append(rg_id)
            if len(batch) >= top_n:
                break
        neighbor_lists.append(batch)

    if not neighbor_lists:
        return []

    merged: list[int] = []
    seen = set(exclude_ids)
    for batch in neighbor_lists:
        for rg_id in batch:
            if rg_id in seen:
                continue
            seen.add(rg_id)
            merged.append(rg_id)
            if len(merged) >= top_n:
                break
        if len(merged) >= top_n:
            break

    if not genre_ids:
        return merged

    genre_set = {int(genre_id) for genre_id in genre_ids}
    filtered: list[int] = []
    for rg_id in merged:
        row = data_model.loc[data_model["id"] == rg_id]
        if row.empty:
            continue
        row_genres = row.iloc[0].get("genre_ids") or []
        if set(row_genres) & genre_set:
            filtered.append(rg_id)
    return filtered[:top_n]


def recommend_release_group_ids(
    release_group_ids: list[int],
    top_n: int = 10,
    *,
    blacklist: list[int] | None = None,
    genre_ids: list[int] | None = None,
) -> list[int]:
    model = get_release_group_model()
    if model is None:
        load_release_group_model()
        model = get_release_group_model()

    if model is None:
        raise RuntimeError(
            "No release group model available. Train with: "
            "python -m ml.release_group.scripts.train_local "
            "then upload: python -m ml.release_group.scripts.upload_release_group"
        )

    if not isinstance(model, dict) or "pipeline" not in model:
        raise RuntimeError(
            "Release group artifact must be a dict with 'pipeline' and 'data_model' keys."
        )

    recommendations = _recommend_release_group_ids_from_artifact(
        model,
        release_group_ids,
        top_n,
        blacklist=blacklist,
        genre_ids=genre_ids,
    )

    recommendations = [x for x in recommendations if x not in blacklist]

    if not recommendations:
        raise RuntimeError("No recommendation found for these release group IDs.")
    return recommendations
