"""Shared artifact resolution and compatibility helpers for model loading."""

import importlib
import sys
import types
from pathlib import Path


def resolve_local_path(
    app_root: Path,
    raw_path: str,
    *,
    env_name: str,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = app_root / path
    if not path.is_file():
        raise RuntimeError(f"{env_name} not found: {path}")
    return path


def resolve_artifact_path(
    *,
    app_root: Path,
    local_path: str | None,
    local_path_env: str,
) -> tuple[Path, str] | None:
    if local_path:
        return resolve_local_path(app_root, local_path, env_name=local_path_env), "local"

    return None


def build_load_info(
    *,
    loaded: bool,
    source: str | None = None,
    path: Path | None = None,
) -> dict:
    if not loaded:
        return {
            "loaded": False,
            "source": None,
            "path": None,
            "filename": None,
        }

    return {
        "loaded": True,
        "source": source,
        "path": str(path),
        "filename": path.name if path else None,
    }


def ensure_legacy_back_package_aliases() -> None:
    """
    Expose compatibility aliases for artifacts pickled with `back.*` module paths.

    The repository used to be importable as a top-level `back` package. Existing
    joblib artifacts still reference classes from modules such as
    `back.ml.release_group.features`. When the API is started from `back/`,
    Python exposes `app` and `ml` directly, but not `back`. Register aliases in
    ``sys.modules`` before unpickling so legacy artifacts keep loading.
    """
    if "back" not in sys.modules:
        back_pkg = types.ModuleType("back")
        back_pkg.__path__ = []  # mark as package-like for import machinery
        sys.modules["back"] = back_pkg

    alias_map = {
        "back.app": "app",
        "back.ml": "ml",
        "back.ml.artist": "ml.artist",
        "back.ml.artist.config": "ml.artist.config",
        "back.ml.artist.features": "ml.artist.features",
        "back.ml.artist.train": "ml.artist.train",
        "back.ml.release_group": "ml.release_group",
        "back.ml.release_group.config": "ml.release_group.config",
        "back.ml.release_group.features": "ml.release_group.features",
        "back.ml.release_group.train": "ml.release_group.train",
    }

    for legacy_name, current_name in alias_map.items():
        if legacy_name in sys.modules:
            continue
        sys.modules[legacy_name] = importlib.import_module(current_name)
