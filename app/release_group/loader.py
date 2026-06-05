"""Load the release group KNN artifact at startup (local file or GCS)."""

import sys
import types

import joblib

from app.models.loader import build_load_info, resolve_artifact_path
from app.release_group.config import (
    APP_ROOT,
    MODEL_BUCKET_NAME,
    RELEASE_GROUP_MODEL_BLOB_NAME,
    release_group_local_path_raw,
)

_release_group_model = None
_release_group_model_load_info: dict = build_load_info(loaded=False)


def _register_legacy_ml_shim() -> None:
    """
    Artifacts trained before this move pickle ListToSparseTransformer as
    ml.release_group.features.* — the Docker image only ships app/.
    """
    from app.release_group import features as app_features

    if "ml.release_group.features" in sys.modules:
        return

    ml_pkg = sys.modules.get("ml")
    if ml_pkg is None:
        ml_pkg = types.ModuleType("ml")
        sys.modules["ml"] = ml_pkg

    rg_pkg = sys.modules.get("ml.release_group")
    if rg_pkg is None:
        rg_pkg = types.ModuleType("ml.release_group")
        sys.modules["ml.release_group"] = rg_pkg
        ml_pkg.release_group = rg_pkg

    sys.modules["ml.release_group.features"] = app_features
    rg_pkg.features = app_features


def get_release_group_model_info() -> dict:
    return dict(_release_group_model_load_info)


def load_release_group_model():
    """Load release group artifact from RELEASE_GROUP_MODEL_LOCAL_PATH or GCS."""
    global _release_group_model, _release_group_model_load_info

    resolved = resolve_artifact_path(
        app_root=APP_ROOT,
        local_path=release_group_local_path_raw() or None,
        local_path_env="RELEASE_GROUP_MODEL_LOCAL_PATH",
        bucket_name=MODEL_BUCKET_NAME,
        blob_name=RELEASE_GROUP_MODEL_BLOB_NAME,
    )

    if resolved is None:
        _release_group_model = None
        _release_group_model_load_info = build_load_info(loaded=False)
        return None

    model_path, source = resolved
    _register_legacy_ml_shim()
    _release_group_model = joblib.load(model_path)
    _release_group_model_load_info = build_load_info(
        loaded=True,
        source=source,
        path=model_path,
        bucket_name=MODEL_BUCKET_NAME,
        blob_name=RELEASE_GROUP_MODEL_BLOB_NAME,
    )
    return _release_group_model


def get_release_group_model():
    return _release_group_model
