"""Typed exceptions. Everything raised on purpose inherits from OutlierError."""


class OutlierError(Exception):
    """Base class for all deliberate errors."""


class ConfigError(OutlierError):
    """Missing or invalid configuration or secret."""


class SafetyError(OutlierError):
    """A guard refused an action that would spend money or leave dry-run."""


class NotFoundError(OutlierError):
    """An object id did not resolve."""


class ValidationError(OutlierError):
    """Domain-level validation failed (distinct from pydantic's)."""
