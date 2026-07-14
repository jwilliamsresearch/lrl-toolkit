"""Shared utilities: logging and small IO helpers."""

from .io import read_json, write_json, write_text
from .logging import get_console, get_logger

__all__ = ["get_console", "get_logger", "read_json", "write_json", "write_text"]
