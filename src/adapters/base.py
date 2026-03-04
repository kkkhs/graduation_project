"""Compatibility wrapper for legacy src.adapters.base."""

from src.application.dto import ModelRuntimeConfig as ModelConfig
from src.infrastructure.adapters.base import BaseAdapter

__all__ = ["ModelConfig", "BaseAdapter"]
