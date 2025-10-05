"""
Atlas AI Monitoring Module
Provides metrics, dashboards, and alerting
"""

from .prometheus_metrics import router as metrics_router
from .admin_dashboard import router as admin_router

__all__ = ['metrics_router', 'admin_router']
