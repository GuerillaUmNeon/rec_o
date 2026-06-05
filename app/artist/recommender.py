"""Artist KNN recommendations from a loaded artifact."""

from app.artist.loader import get_artist_model, load_artist_model


def _recommend_artist_ids_from_artifact(
    artifact: dict,
    artist_ids: list[int],
    top_n: int,
    blacklist_artist_ids: list[int] | None = None,
) -> list[int]:
    recommender = artifact["model"]
    vectorizer = artifact["vectorizer"]
    data = artifact.get("data")
    if data is None:
        data = artifact.get("df_clean")

    if data is None:
        raise RuntimeError(
            "Artist artifact must contain df_clean as 'data'. "
            "Save {'vectorizer': vectorizer, 'model': knn_model, 'data': df_clean}."
        )

    df_clean = data.copy()
    df_clean["genres"] = df_clean["genres"].fillna("")

    seed_ids = {int(artist_id) for artist_id in artist_ids}
    blacklist_ids = {
        int(artist_id)
        for artist_id in (blacklist_artist_ids or [])
    }
    excluded_ids = seed_ids | blacklist_ids
    matches = df_clean[df_clean["artist_id"].isin(seed_ids)]
    found_ids = {int(artist_id) for artist_id in matches["artist_id"].tolist()}
    missing_ids = sorted(seed_ids - found_ids)
    if missing_ids:
        raise RuntimeError(
            f"Artiste inconnu dans la base de données: {missing_ids}"
        )

    query_vectors = vectorizer.transform(matches["genres"].astype(str))
    if len(matches) == 1:
        query_vector = query_vectors[0]
    else:
        query_vector = query_vectors.mean(axis=0).A

    if len(excluded_ids) >= len(df_clean):
        raise RuntimeError(
            "Pas assez de recommandations disponibles après application de la blacklist."
        )

    n_neighbors = min(len(df_clean), top_n + len(excluded_ids) + 50)
    recommendations = []

    while True:
        _, indices = recommender.kneighbors(
            query_vector,
            n_neighbors=n_neighbors,
        )

        for idx in indices[0]:
            row = df_clean.iloc[idx]
            artist_id = int(row["artist_id"])

            if artist_id in excluded_ids or artist_id in recommendations:
                continue

            recommendations.append(artist_id)
            if len(recommendations) >= top_n:
                return recommendations

        if n_neighbors >= len(df_clean):
            break

        n_neighbors = min(
            len(df_clean),
            max(n_neighbors * 2, n_neighbors + top_n + len(excluded_ids) + 50),
        )

    return recommendations


def recommend_artist_ids(
    artist_ids: list[int],
    top_n: int = 5,
    blacklist_artist_ids: list[int] | None = None,
) -> list[int]:
    model = get_artist_model()
    if model is None:
        load_artist_model()
        model = get_artist_model()

    if model is None:
        raise RuntimeError(
            "No artist model available. Train with: python -m ml.artist.scripts.train_local "
            "then upload: python -m ml.artist.scripts.upload_artist"
        )

    if isinstance(model, dict):
        recommendations = _recommend_artist_ids_from_artifact(
            model,
            artist_ids,
            top_n,
            blacklist_artist_ids=blacklist_artist_ids,
        )
        if len(recommendations) < top_n:
            raise RuntimeError(
                "Pas assez de recommandations disponibles après application de la blacklist."
            )
        return recommendations

    pred_result = model.predict([artist_ids])[0]
    if hasattr(pred_result, "tolist"):
        pred_result = pred_result.tolist()
    if not isinstance(pred_result, list):
        pred_result = list(pred_result)
    excluded_ids = {
        int(artist_id)
        for artist_id in [*artist_ids, *(blacklist_artist_ids or [])]
    }
    recommendations = [
        int(artist_id)
        for artist_id in pred_result
        if int(artist_id) not in excluded_ids
    ]
    if len(recommendations) < top_n:
        raise RuntimeError(
            "Pas assez de recommandations disponibles après application de la blacklist."
        )
    return recommendations[:top_n]
