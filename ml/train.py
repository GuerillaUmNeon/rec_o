"""
Train TF-IDF + KNN artist recommender (aligned with app/predictor.py).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ml.features import GENRE_FEATURE_FORMAT, GENRE_TOKEN_PATTERN


def build_artist_recommender_artifact(
    df: pd.DataFrame,
    n_neighbors: int = 20,
) -> dict:
    """
    Train the artist recommender from genre tokens.

    The genres column must contain one whitespace-separated token per
    MusicBrainz genre, for example: "east_coast_hip_hop hip_hop".
    """
    df_clean = df.copy()
    df_clean["genres"] = df_clean["genres"].fillna("").astype(str).str.split().str.join(" ")
    df_clean = df_clean[df_clean["genres"].str.strip() != ""].copy()

    if df_clean.empty:
        raise ValueError("Cannot train artist recommender without genre data.")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 1),
        token_pattern=GENRE_TOKEN_PATTERN,
    )
    vectors = vectorizer.fit_transform(df_clean["genres"])

    knn_model = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(df_clean)),
        metric="cosine",
        algorithm="brute",
    )
    knn_model.fit(vectors)

    artist_names = (
        df_clean[["artist_id", "artist_name"]]
        .drop_duplicates("artist_id")
        .set_index("artist_id")["artist_name"]
        .to_dict()
    )

    return {
        "vectorizer": vectorizer,
        "model": knn_model,
        "artist_names": artist_names,
        "data": df_clean,
        "genre_feature_format": GENRE_FEATURE_FORMAT,
    }


def train_artist_recommender(conn, n_neighbors: int = 20) -> dict:
    from ml.data import fetch_artist_recommender_training_data

    df = fetch_artist_recommender_training_data(conn)
    return build_artist_recommender_artifact(df, n_neighbors=n_neighbors)
