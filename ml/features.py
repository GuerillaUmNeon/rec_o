"""Genre feature helpers (aligned with app/predictor.py)."""

import re

import pandas as pd

GENRE_FEATURE_FORMAT = "genre_token_unigram"
GENRE_TOKEN_PATTERN = r"(?u)\b\w+\b"


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


def genre_feature_token(genre: str) -> str:
    return re.sub(r"\W+", "_", str(genre).strip().lower()).strip("_")


def join_genre_feature_tokens(values) -> str:
    tokens = []
    for genre in ordered_unique(values):
        token = genre_feature_token(genre)
        if token:
            tokens.append(token)
    return " ".join(tokens)
