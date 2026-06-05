"""Artist genre feature helpers for KNN training (aligned with app inference)."""

import re

import pandas as pd

ARTIST_GENRE_FEATURE_FORMAT = "genre_token_unigram"
ARTIST_GENRE_TOKEN_PATTERN = r"(?u)\b\w+\b"

# Backward-compatible aliases (artist-only pipeline today)
GENRE_FEATURE_FORMAT = ARTIST_GENRE_FEATURE_FORMAT
GENRE_TOKEN_PATTERN = ARTIST_GENRE_TOKEN_PATTERN


def ordered_unique(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if pd.isna(value):
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def artist_genre_feature_token(genre: str) -> str:
    return re.sub(r"\W+", "_", str(genre).strip().lower()).strip("_")


def join_artist_genre_feature_tokens(values) -> str:
    tokens = []
    for genre in ordered_unique(values):
        token = artist_genre_feature_token(genre)
        if token:
            tokens.append(token)
    return " ".join(tokens)
