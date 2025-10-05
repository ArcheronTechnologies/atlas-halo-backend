"""
Geocoding utilities for the backend
Wrapper around the atlas_mvp geocoding module
"""

import sys
from pathlib import Path

# Add atlas_mvp to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "atlas_mvp" / "backend" / "src"))

from pipelines.geocoding import geocode, reverse_geocode

__all__ = ["geocode", "reverse_geocode"]
