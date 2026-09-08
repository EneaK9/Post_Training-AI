"""Shared, IO-free schemas for Outlier AI.

Everything the backend, trainer, and API agree on lives here: enums, the section 3
domain objects as pydantic models, and the typed configuration tree.
"""

from outlier_schemas import config, enums, models

__all__ = ["config", "enums", "models"]
