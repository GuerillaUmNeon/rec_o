"""
Train TF-IDF + KNN artist recommender (baseline from note_book_joris.ipynb).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook cleaning: dedupe, drop empty genres and Various Artists."""
    df_clean = df.copy()

    for col in ("artist_name", "artist_type", "area_name", "tags", "genres"):
        if col in df_clean.columns:
            fill = "Unknown" if col in ("artist_type", "area_name") else ""
            df_clean[col] = df_clean[col].fillna(fill)

    df_clean = df_clean.drop_duplicates()
    if "artist_id" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["artist_id"])

    if "artist_name" in df_clean.columns:
        df_clean = df_clean[
            df_clean["artist_name"].str.lower() != "various artists"
        ].copy()

    if "tag_count_sum" in df_clean.columns:
        df_clean["tag_count_clean"] = df_clean["tag_count_sum"].clip(lower=0)

    df_clean = df_clean[df_clean["genres"].str.strip() != ""].copy()
    return df_clean


def train_knn_artifact(
    df_clean: pd.DataFrame,
    *,
    max_features: int = 5000,
    n_neighbors: int = 20,
) -> dict:
    """
    Fit vectorizer + NearestNeighbors on genre text; return API-compatible artifact.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(df_clean["genres"].astype(str))

    knn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    knn_model.fit(X)

    artist_names = (
        df_clean["artist_name"].tolist()
        if "artist_name" in df_clean.columns
        else []
    )

    return {
        "vectorizer": vectorizer,
        "model": knn_model,
        "artist_names": artist_names,
        "data": df_clean,
    }
