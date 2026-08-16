"""Acquisition: async clients and orchestration."""

from solarcast.ingestion.base import BaseProviderClient
from solarcast.ingestion.openmeteo import OpenMeteoClient
from solarcast.ingestion.pvgis import PVGISClient
from solarcast.ingestion.service import (
    IngestionResult,
    build_client,
    ingest_all,
    ingest_historical,
)

__all__ = [
    "BaseProviderClient",
    "OpenMeteoClient",
    "PVGISClient",
    "IngestionResult",
    "build_client",
    "ingest_historical",
    "ingest_all",
]
