"""Release group KNN inference: load artifact, recommend IDs, enrich from DB."""

from app.release_group.enrichment import enrich_release_groups_from_db
from app.release_group.loader import (
    get_release_group_model_info,
    load_release_group_model,
)
from app.release_group.recommender import recommend_release_group_ids

__all__ = [
    "load_release_group_model",
    "get_release_group_model_info",
    "recommend_release_group_ids",
    "enrich_release_groups_from_db",
]
