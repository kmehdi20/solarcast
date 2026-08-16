"""Asynchronous rate limiter.

The public APIs in use (PVGIS, NASA POWER, Open-Meteo) cap the number of
calls: NASA POWER's documentation explicitly states that its endpoints
throttle requests to avoid overload from rapid repeated calls. Without
client-side regulation, a multi-site ingestion run in parallel gets cut
off within seconds.

Two combined safeguards:

* a **token bucket** that smooths the average rate (requests per second);
* a **semaphore** that bounds the number of concurrent requests.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType


class TokenBucket:
    """Token bucket: at most `rate` acquisitions per second on average.

    The capacity allows a short burst at startup, avoiding unnecessary
    serialization of the first few requests.
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be strictly positive")
        self._rate = rate
        self._capacity = capacity if capacity is not None else max(rate, 1.0)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until the requested number of tokens is available."""
        if tokens > self._capacity:
            raise ValueError("requested amount exceeds bucket capacity")

        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                deficit = tokens - self._tokens
                wait_for = deficit / self._rate

            await asyncio.sleep(wait_for)


class RateLimiter:
    """Combines a token bucket with a concurrency bound.

    Used as an async context manager:

    >>> async with limiter:
    ...     await client.get(url)
    """

    def __init__(self, requests_per_second: float, max_concurrency: int) -> None:
        self._bucket = TokenBucket(requests_per_second)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self) -> "RateLimiter":
        await self._semaphore.acquire()
        try:
            await self._bucket.acquire()
        except BaseException:
            self._semaphore.release()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._semaphore.release()
