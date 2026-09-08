"""Typed Meta errors. The sync jobs map these onto account state (scenario table row 12)."""

from outlier_ai.core.errors import OutlierError


class MetaError(OutlierError):
    """Base for anything the Meta client raises on purpose."""


class MetaAuthError(MetaError):
    """Token expired or revoked. The account is marked `needs_reauth`; nothing else stops."""


class MetaRateLimitError(MetaError):
    """Transient. Jobs back off and retry."""


class MetaApiVersionError(MetaError):
    """The pinned API version is no longer served. Treated like an auth error operationally."""


class MetaObjectNotFoundError(MetaError):
    """An ad, ad set, creative, or post id did not resolve."""


class MetaValidationError(MetaError):
    """Meta rejected a create call (bad spec, policy pre-check, missing permission)."""
