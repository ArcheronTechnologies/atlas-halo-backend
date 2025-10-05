"""Observability metrics"""
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class MockCounter:
    """Mock counter for metrics"""
    def __init__(self, name: str, description: str, labels: Tuple[str, ...] = ()):
        self.name = name
        self.description = description
        self._labels = {}

    def labels(self, **kwargs):
        """Set labels"""
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = MockCounter(self.name, self.description)
        return self._labels[key]

    def inc(self, amount: float = 1.0):
        """Increment counter"""
        pass

class MockMetrics:
    """Mock metrics system"""
    def counter(self, name: str, description: str, labels: Tuple[str, ...] = ()) -> MockCounter:
        return MockCounter(name, description, labels)

    def gauge(self, name: str, description: str, labels: Tuple[str, ...] = ()) -> MockCounter:
        return MockCounter(name, description, labels)

    def histogram(self, name: str, description: str, labels: Tuple[str, ...] = ()) -> MockCounter:
        return MockCounter(name, description, labels)

metrics = MockMetrics()
