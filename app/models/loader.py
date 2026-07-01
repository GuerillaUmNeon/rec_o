"""Shared artifact resolution: local path lookup."""

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
