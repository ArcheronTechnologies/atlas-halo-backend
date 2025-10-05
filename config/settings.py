"""
Centralized configuration with lightweight env parsing (no extra deps).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List


def _get_list(var: str, default: List[str]) -> List[str]:
    val = os.getenv(var)
    if not val:
        return default
    # Split by comma and strip
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass
class Settings:
    rate_capacity_default: int
    rate_capacity_analytics: int
    cors_allowed_origins: List[str]
    retention_scheduler_enabled: bool
    lakehouse_enabled: bool
    lakehouse_path: str
    # Ingestion: Polisen API
    polisen_base_url: str
    polisen_events_path: str
    polisen_user_agent: str
    polisen_min_interval_seconds: float
    polisen_hourly_cap: int
    polisen_daily_cap: int
    polisen_timeout_seconds: float
    # PxWeb config (Brå/SCB)
    bra_pxweb_base_url: str
    pxweb_timeout_seconds: float


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(
            rate_capacity_default=int(os.getenv("RATE_CAPACITY_DEFAULT", "60")),
            rate_capacity_analytics=int(os.getenv("RATE_CAPACITY_ANALYTICS", "30")),
            cors_allowed_origins=_get_list(
                "CORS_ALLOWED_ORIGINS",
                [
                    "https://dashboard.atlasai.internal",
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:8000",
                    "http://127.0.0.1:8000",
                ],
            ),
            retention_scheduler_enabled=os.getenv("RETENTION_SCHEDULER_ENABLED", "1")
            == "1",
            lakehouse_enabled=os.getenv("LAKEHOUSE_ENABLED", "0") == "1",
            lakehouse_path=os.getenv("LAKEHOUSE_PATH", "data_lake"),
            # Polisen API
            polisen_base_url=os.getenv("POLISEN_BASE_URL", "https://polisen.se"),
            polisen_events_path=os.getenv("POLISEN_EVENTS_PATH", "/api/events"),
            polisen_user_agent=os.getenv(
                "POLISEN_USER_AGENT",
                "Atlas AI Crime Analysis (contact: ops@atlasai.local)",
            ),
            polisen_min_interval_seconds=float(
                os.getenv("POLISEN_MIN_INTERVAL_SECONDS", "10.0")
            ),
            polisen_hourly_cap=int(os.getenv("POLISEN_HOURLY_CAP", "60")),
            polisen_daily_cap=int(os.getenv("POLISEN_DAILY_CAP", "1440")),
            polisen_timeout_seconds=float(os.getenv("POLISEN_TIMEOUT_SECONDS", "30.0")),
            # PxWeb
            bra_pxweb_base_url=os.getenv(
                "BRA_PXWEB_BASE_URL",
                # Fallback to SCB PxWeb base when Brå endpoint is unavailable
                os.getenv("SCB_PXWEB_BASE_URL", "https://api.scb.se/OV0104/v1/doris/sv"),
            ),
            pxweb_timeout_seconds=float(os.getenv("PXWEB_TIMEOUT_SECONDS", "30.0")),
        )
    return _settings
