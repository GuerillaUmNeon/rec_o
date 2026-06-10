"""Release group KNN recommendations from a loaded artifact."""
import pandas as pd
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

    seed_ids = list(dict.fromkeys(int(rg_id) for rg_id in release_group_ids))
    exclude_ids = set(seed_ids)
    if blacklist:
        exclude_ids.update(int(rg_id) for rg_id in blacklist)

    missing_ids = sorted(set(seed_ids) - set(id_to_idx))
    if missing_ids:
        raise RuntimeError(f"Unknown release group IDs: {missing_ids}")

    if not seed_ids:
        return []

    seed_rows = data_model.iloc[[id_to_idx[rg_id] for rg_id in seed_ids]].copy()

    def _mode_or_na(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return pd.NA
        mode = s.mode()
        return mode.iloc[0] if not mode.empty else pd.NA

    def _list_union(series: pd.Series) -> list[int]:
        values = set()
        for item in series:
            if isinstance(item, list):
                values.update(int(x) for x in item)
        return sorted(values)

    query_row = {}

    for col in data_model.columns:
        if col == "id":
            query_row[col] = -1
        elif col in {"tag_ids", "genre_ids", "secondary_type_ids"}:
            query_row[col] = _list_union(seed_rows[col]) if col in seed_rows.columns else []
        elif col == "year":
            s = seed_rows[col].dropna()
            query_row[col] = int(round(s.mean())) if not s.empty else pd.NA
        elif col in {"type", "status", "language", "script"}:
            query_row[col] = _mode_or_na(seed_rows[col]) if col in seed_rows.columns else pd.NA
        else:
            # Preserve unused columns if they exist in data_model
            s = seed_rows[col].dropna()
            query_row[col] = s.iloc[0] if not s.empty else pd.NA

    query_df = pd.DataFrame([query_row], columns=data_model.columns)

    X_query = preprocessor.transform(query_df)

    distances, indices = knn.kneighbors(
        X_query,
        n_neighbors=min(len(data_model), top_n + len(exclude_ids) + 200),
    )

    genre_set = {int(g) for g in genre_ids} if genre_ids else None

    results: list[int] = []
    for idx in indices[0]:
        row = data_model.iloc[idx]
        rg_id = int(row["id"])

        if rg_id in exclude_ids:
            continue

        if genre_set is not None:
            row_genres = row.get("genre_ids") or []
            if not (set(int(g) for g in row_genres) & genre_set):
                continue

        results.append(rg_id)
        if len(results) >= top_n:
            break

    return results


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
