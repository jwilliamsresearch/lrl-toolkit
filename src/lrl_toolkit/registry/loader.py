"""Discovery and loading of language / model / compute profiles.

Profiles are looked up by slug across a search path so users can override or add
profiles without editing the installed package. Search order (first match wins):

1. Directories listed in the ``LRL_CONFIG_PATH`` environment variable.
2. ``<cwd>/configs`` in the current working directory.
3. The profiles bundled with the package (``lrl_toolkit/configs``).

Each of those roots is expected to contain ``languages/``, ``models/``, and
``compute/`` subdirectories of ``<slug>.yaml`` files.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .profiles import ComputeProfile, LanguageProfile, ModelProfile

_BUNDLED_CONFIGS = Path(__file__).resolve().parent.parent / "configs"

_KIND_TO_SUBDIR = {
    "language": "languages",
    "model": "models",
    "compute": "compute",
}


def config_search_path() -> list[Path]:
    """Return config root directories in priority order."""
    roots: list[Path] = []
    env = os.environ.get("LRL_CONFIG_PATH")
    if env:
        roots.extend(Path(p) for p in env.split(os.pathsep) if p.strip())
    roots.append(Path.cwd() / "configs")
    roots.append(_BUNDLED_CONFIGS)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        rp = root.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(root)
    return unique


class ProfileNotFoundError(FileNotFoundError):
    """Raised when a profile slug cannot be resolved on the search path."""


def _find_profile_file(kind: str, slug: str) -> Path:
    subdir = _KIND_TO_SUBDIR[kind]
    for root in config_search_path():
        candidate = root / subdir / f"{slug}.yaml"
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(r / subdir) for r in config_search_path())
    raise ProfileNotFoundError(
        f"No {kind} profile '{slug}' found. Searched: {searched}"
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Profile file {path} must contain a YAML mapping.")
    return data


def load_language(slug: str) -> LanguageProfile:
    data = _load_yaml(_find_profile_file("language", slug))
    data.setdefault("name", slug)
    return LanguageProfile.model_validate(data)


def load_model(slug: str) -> ModelProfile:
    data = _load_yaml(_find_profile_file("model", slug))
    data.setdefault("name", slug)
    return ModelProfile.model_validate(data)


def load_compute(slug: str) -> ComputeProfile:
    data = _load_yaml(_find_profile_file("compute", slug))
    data.setdefault("name", slug)
    return ComputeProfile.model_validate(data)


def _list_slugs(kind: str) -> list[str]:
    subdir = _KIND_TO_SUBDIR[kind]
    slugs: set[str] = set()
    for root in config_search_path():
        d = root / subdir
        if d.is_dir():
            slugs.update(p.stem for p in d.glob("*.yaml"))
    return sorted(slugs)


def list_languages() -> list[str]:
    return _list_slugs("language")


def list_models() -> list[str]:
    return _list_slugs("model")


def list_compute() -> list[str]:
    return _list_slugs("compute")
