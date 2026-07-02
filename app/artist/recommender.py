"""Artist KNN recommendations from a loaded artifact."""

from app.database import engine
from app.artist.loader import get_artist_model, load_artist_model


def _filter_recommendable_artist_ids(candidate_ids: list[int]) -> list[int]:
    """
    Keep only artists that are recommendable as primary music artists.

    MusicBrainz stores people like engineers, photographers, designers, and
    managers in the artist table. For recommendation output, keep only
    Person/Group entities that are credited as primary artists on a release
    group or release, preserving the KNN order.
    """
    if not candidate_ids:
        return []

    with engine.connect() as conn:
        # Build ARRAY constructor to avoid parameter binding issues with unnest()
        # Use raw cursor to bypass SQLAlchemy parameter processing
        ids_str = ", ".join(str(cid) for cid in candidate_ids)
        
        with conn.connection.cursor() as cursor:
            query = f"""
                WITH candidate_ids(id, ord) AS (
                    SELECT *
                    FROM unnest(ARRAY[{ids_str}]::int[]) WITH ORDINALITY
                ),
                primary_artists AS (
                    SELECT DISTINCT acn.artist AS id
                    FROM release_group rg
                    JOIN artist_credit_name acn
                        ON acn.artist_credit = rg.artist_credit
                    JOIN candidate_ids
                        ON candidate_ids.id = acn.artist

                    UNION

                    SELECT DISTINCT acn.artist AS id
                    FROM release r
                    JOIN artist_credit_name acn
                        ON acn.artist_credit = r.artist_credit
                    JOIN candidate_ids
                        ON candidate_ids.id = acn.artist
                )
                SELECT candidate_ids.id
                FROM candidate_ids
                JOIN primary_artists
                    ON primary_artists.id = candidate_ids.id
                JOIN artist
                    ON artist.id = candidate_ids.id
                JOIN artist_type
                    ON artist_type.id = artist.type
                WHERE artist_type.name IN ('Person', 'Group')
                ORDER BY candidate_ids.ord
            """
            cursor.execute(query)
            return [int(row[0]) for row in cursor.fetchall()]


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
        data = model.get("data") or model.get("df_clean")
        if data is None:
            raise RuntimeError(
                "Artist artifact must contain df_clean as 'data'. "
                "Save {'vectorizer': vectorizer, 'model': knn_model, 'data': df_clean}."
            )
        
        # Type check: data is guaranteed to be non-None here and has a length
        candidate_count = min(
            len(data),
            max(top_n * 80, top_n + len(blacklist_artist_ids or []) + 200, 800),
        )
        candidates = _recommend_artist_ids_from_artifact(
            model,
            artist_ids,
            candidate_count,
            blacklist_artist_ids=blacklist_artist_ids,
        )
        recommendations = _filter_recommendable_artist_ids(candidates)[:top_n]
        if len(recommendations) < top_n:
            raise RuntimeError(
                "Pas assez de recommandations disponibles après application des filtres."
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
