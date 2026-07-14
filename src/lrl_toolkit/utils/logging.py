"""Rich-based logging for the toolkit, with a plain fallback."""

from __future__ import annotations

import logging

try:  # Rich is a core dependency, but degrade gracefully if unavailable.
    from rich.console import Console
    from rich.logging import RichHandler

    _console = Console(stderr=True)
except Exception:  # pragma: no cover - defensive fallback
    Console = None  # type: ignore[assignment]
    RichHandler = None  # type: ignore[assignment]
    _console = None

_CONFIGURED = False


def get_console():
    """Return the shared Rich console (or ``None`` if Rich is unavailable)."""
    return _console


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler: logging.Handler
    if RichHandler is not None:
        handler = RichHandler(
            console=_console, show_time=True, show_path=False, rich_tracebacks=True
        )
        fmt = "%(message)s"
    else:  # pragma: no cover
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[handler], force=True)
    # Quiet chatty third-party libraries (HTTP requests, Hub, dataset builders).
    for noisy in (
        "httpx", "httpcore", "urllib3", "filelock", "fsspec",
        "huggingface_hub", "datasets", "requests",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str = "lrl") -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
