"""Generic acquisition client.

`BaseProviderClient` factors out everything that doesn't depend on the
provider: shared HTTP session, rate limiting, exponential backoff with
jitter, error classification. A concrete client only has to implement
request construction and translating the response into
`ObservationPoint` objects.

The `fetch_historical` / `fetch_forecast` split is deliberate: only some
sources publish forecasts. PVGIS and NASA POWER serve historical data
only, and a forecast attempt should fail explicitly rather than silently
return the past.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from datetime import datetime
from types import TracebackType
from typing import Any

import httpx

from solarcast.core.config import ProviderConfig
from solarcast.core.exceptions import ProviderError, RetryableProviderError
from solarcast.core.logging import get_logger
from solarcast.core.types import ObservationPoint, Provider, Variable
from solarcast.ingestion.ratelimit import RateLimiter

logger = get_logger(__name__)

#: HTTP status codes considered transient.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class BaseProviderClient(ABC):
    """Common foundation for every ingestion client."""

    #: Provider identifier, to override in each subclass.
    provider: Provider

    #: Does the provider publish forecasts?
    supports_forecast: bool = False

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._limiter = RateLimiter(
            requests_per_second=config.rate_limit.requests_per_second,
            max_concurrency=config.rate_limit.max_concurrency,
        )
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------- lifecycle

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            headers={"User-Agent": "solarcast/0.1 (+research)"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                f"{type(self).__name__} must be used as an async context "
                "manager (`async with`)"
            )
        return self._client

    # --------------------------------------------------------------- requests

    def _backoff_delay(self, attempt: int) -> float:
        """Delay before attempt `attempt` (1-indexed)."""
        policy = self.config.retry
        delay = policy.initial_backoff_s * (policy.backoff_multiplier ** (attempt - 1))
        delay = min(delay, policy.max_backoff_s)
        if policy.jitter:
            # Full jitter: prevents multiple tasks from resynchronizing on
            # the same slot after a shared 429.
            delay = random.uniform(0.0, delay)
        return delay

    async def request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        method: str = "GET",
    ) -> httpx.Response:
        """Execute a rate-limited, retried request.

        Raises `RetryableProviderError` if every attempt fails on a
        transient error, `ProviderError` on a permanent one.
        """
        policy = self.config.retry
        last_error: Exception | None = None

        for attempt in range(1, policy.max_attempts + 1):
            try:
                async with self._limiter:
                    response = await self.http.request(method, path, params=params)

                if response.status_code in RETRYABLE_STATUS:
                    raise RetryableProviderError(
                        self.provider.value,
                        "transient server response",
                        response.status_code,
                    )
                if response.status_code >= 400:
                    raise ProviderError(
                        self.provider.value,
                        response.text[:300].strip() or "no error detail",
                        response.status_code,
                    )
                return response

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = RetryableProviderError(
                    self.provider.value, f"network error: {exc}"
                )
            except RetryableProviderError as exc:
                last_error = exc
            except ProviderError:
                raise  # permanent: no point retrying

            if attempt < policy.max_attempts:
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "attempt failed, retry scheduled",
                    extra={
                        "context": {
                            "provider": self.provider.value,
                            "attempt": attempt,
                            "max_attempts": policy.max_attempts,
                            "delay_s": round(delay, 2),
                            "error": str(last_error),
                        }
                    },
                )
                await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error

    async def get_json(self, path: str, params: dict[str, Any]) -> Any:
        """GET request returning decoded JSON."""
        response = await self.request(path, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                self.provider.value, f"unreadable JSON response: {exc}"
            ) from exc

    # ---------------------------------------------------------------- contract

    @abstractmethod
    async def fetch_historical(
        self,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
        variables: list[Variable] | None = None,
    ) -> list[ObservationPoint]:
        """Fetch past data over the requested window."""

    async def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        horizon_days: int = 3,
        variables: list[Variable] | None = None,
    ) -> list[ObservationPoint]:
        """Fetch a forecast. Not available by default."""
        raise NotImplementedError(
            f"{self.provider.value} does not publish forecasts; "
            "use a client with supports_forecast set to True"
        )
