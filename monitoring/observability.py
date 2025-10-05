"""Lightweight observability helpers for Phase 0 baseline."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional


_logger = logging.getLogger("observability")


@dataclass
class MetricEvent:
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


def record_metric(name: str, value: float = 1.0, **labels: str) -> None:
    """Record a metric data point.

    Phase 0 baseline pushes metrics to the log stream for later ingestion.
    """

    event = MetricEvent(name=name, value=value, labels=labels)
    _logger.info("metric", extra={"metric": event.__dict__})


@contextmanager
def trace_span(span_name: str, **labels: str):
    """Context manager to trace execution time of a span."""

    start = time.perf_counter()
    try:
        yield
    except Exception:
        record_metric(f"{span_name}.exceptions", 1, **labels)
        raise
    finally:
        duration = time.perf_counter() - start
        record_metric(f"{span_name}.duration_seconds", duration, **labels)


def log_sla(service: str, status: str, detail: Optional[str] = None) -> None:
    """Emit SLA compliance information for dashboards."""

    payload = {"service": service, "status": status}
    if detail:
        payload["detail"] = detail
    _logger.info("sla", extra={"sla": payload})

