"""Externalized configuration.

Three levels, from weakest to strongest:

1. defaults declared in the Pydantic models;
2. YAML file (``config/settings.yaml`` by default);
3. environment variables prefixed with ``SOLARCAST_``.

No secrets should live in the YAML: API keys and base URLs go exclusively
through the environment (or a ``.env`` file loaded by the orchestrator).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from solarcast.core.exceptions import ConfigError

ENV_PREFIX = "SOLARCAST_"
DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


class LocationConfig(BaseModel):
    """Geographic site tracked by the framework."""

    name: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    altitude_m: float | None = None
    timezone: str = "UTC"

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("site name cannot be empty")
        return v.strip()


class RateLimitConfig(BaseModel):
    """Rate-limiter settings applied to a client."""

    max_concurrency: int = Field(default=4, ge=1)
    requests_per_second: float = Field(default=2.0, gt=0)


class RetryConfig(BaseModel):
    """Retry policy on transient errors."""

    max_attempts: int = Field(default=4, ge=1)
    initial_backoff_s: float = Field(default=1.0, gt=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_backoff_s: float = Field(default=30.0, gt=0)
    jitter: bool = True


class ProviderConfig(BaseModel):
    """Configuration for a single data provider."""

    enabled: bool = True
    base_url: str
    timeout_s: float = Field(default=30.0, gt=0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    options: dict[str, Any] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    """Connection to the time-series database."""

    url: str = "sqlite+aiosqlite:///data/solarcast.db"
    echo: bool = False
    pool_size: int = Field(default=5, ge=1)

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = False

    @field_validator("level")
    @classmethod
    def _known_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"unknown log level: {v}")
        return upper


class Settings(BaseModel):
    """Root of the application configuration."""

    environment: str = "dev"
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    locations: list[LocationConfig] = Field(default_factory=list)

    def provider(self, name: str) -> ProviderConfig:
        """Return a provider's configuration or raise `ConfigError`."""
        try:
            return self.providers[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.providers)) or "none"
            raise ConfigError(
                f"provider '{name}' not found in configuration (known: {known})"
            ) from exc

    def location(self, name: str) -> LocationConfig:
        for loc in self.locations:
            if loc.name.lower() == name.lower():
                return loc
        known = ", ".join(loc.name for loc in self.locations) or "none"
        raise ConfigError(f"unknown site '{name}' (declared: {known})")


def _coerce(raw: str) -> Any:
    """Interpret an environment value as a Python type."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply `SOLARCAST_*` variables onto the configuration tree.

    A double underscore separates levels:
    ``SOLARCAST_DATABASE__URL`` overrides ``database.url``.
    """
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        cursor: dict[str, Any] = data
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce(value)
    return data


def load_settings(path: str | Path | None = None) -> Settings:
    """Load configuration from YAML, then apply environment overrides.

    A missing file is not an error: defaults plus the environment are
    enough to start.
    """
    config_path = Path(path or os.getenv(f"{ENV_PREFIX}CONFIG", DEFAULT_CONFIG_PATH))
    data: dict[str, Any] = {}

    if config_path.is_file():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(f"{config_path} must contain a mapping at the root")
            data = loaded

    data = _apply_env_overrides(data)

    try:
        return Settings.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError and subclasses
        raise ConfigError(f"invalid configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Memoized instance for regular application use."""
    return load_settings()
