"""Framework exception hierarchy.

Every error deliberately raised by SolarCast derives from `SolarCastError`,
which lets calling layers distinguish an internal failure from an
unanticipated third-party error (httpx, SQLAlchemy).
"""

from __future__ import annotations


class SolarCastError(Exception):
    """Base class for all SolarCast errors."""


class ConfigError(SolarCastError):
    """Configuration missing, malformed, or inconsistent."""


class IngestionError(SolarCastError):
    """Generic error occurring during data acquisition."""


class ProviderError(IngestionError):
    """The remote provider responded, but with an error.

    Parameters
    ----------
    provider:
        Name of the client involved (``open-meteo``, ``pvgis``...).
    message:
        Human-readable message.
    status_code:
        HTTP status code, if available.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"[{provider}] {message}{suffix}")


class RetryableProviderError(ProviderError):
    """Transient error: the request can be replayed as-is.

    Covers 429s, 5xxs, and network/timeout errors.
    """


class ValidationError(IngestionError):
    """The received data does not match the expected contract."""


class StorageError(SolarCastError):
    """Persistence error."""
