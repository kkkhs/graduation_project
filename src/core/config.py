"""Compatibility wrapper for legacy src.core.config."""

from src.application.dto import RuntimeConfig  # re-export
from src.infrastructure.config_loader import load_runtime_config  # re-export

__all__ = ["RuntimeConfig", "load_runtime_config"]
