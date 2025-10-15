"""
Real-time Notifications System

Production-ready WebSocket-based notification system with event routing,
user targeting, persistence, and delivery guarantees.
"""

from .manager import notification_manager, NotificationManager
from .models import *
from .channels import *

__all__ = [
    'notification_manager',
    'NotificationManager'
]