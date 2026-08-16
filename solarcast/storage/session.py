"""Engine and async session management.

The engine is created once per process. On SQLite, two settings are
applied on every connection:

* ``journal_mode=WAL`` — allows concurrent reads during a write, which
  matters as soon as a scheduler is ingesting while a notebook reads;
* ``foreign_keys=ON`` — SQLite disables foreign keys by default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from solarcast.core.config import DatabaseConfig
from solarcast.core.logging import get_logger
from solarcast.storage.models import Base

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record) -> None:  # noqa: ANN001
    """Apply SQLite PRAGMAs when each connection opens."""
    module = type(dbapi_connection).__module__
    if "sqlite" not in module:
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


def _ensure_sqlite_directory(url: str) -> None:
    """Create the SQLite file's parent directory if needed."""
    marker = ":///"
    if marker not in url:
        return
    path_part = url.split(marker, 1)[1].split("?", 1)[0]
    if not path_part or path_part == ":memory:":
        return
    Path(path_part).expanduser().parent.mkdir(parents=True, exist_ok=True)


def create_engine(config: DatabaseConfig) -> AsyncEngine:
    """Build an async engine from the configuration."""
    kwargs: dict[str, object] = {"echo": config.echo, "future": True}
    if config.is_sqlite:
        _ensure_sqlite_directory(config.url)
    else:
        kwargs["pool_size"] = config.pool_size
        kwargs["pool_pre_ping"] = True
    return create_async_engine(config.url, **kwargs)


def init_engine(config: DatabaseConfig) -> AsyncEngine:
    """Initialize the global engine (idempotent)."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine(config)
        _session_factory = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
        logger.info(
            "database engine initialized",
            extra={"context": {"dialect": _engine.dialect.name}},
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("init_engine() must be called before get_session_factory()")
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session: commit on success, rollback on exception."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def create_schema(engine: AsyncEngine | None = None) -> None:
    """Create any missing tables.

    Good enough for development. In production, use Alembic instead to get
    versioned, reversible migrations.
    """
    target = engine or _engine
    if target is None:
        raise RuntimeError("no engine initialized")
    async with target.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("schema verified", extra={"context": {"tables": len(Base.metadata.tables)}})


async def healthcheck(engine: AsyncEngine | None = None) -> bool:
    """Check that the database responds."""
    target = engine or _engine
    if target is None:
        return False
    try:
        async with target.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - depends on infrastructure
        logger.error("database healthcheck failed", extra={"context": {"error": str(exc)}})
        return False


async def dispose_engine() -> None:
    """Cleanly close the connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
