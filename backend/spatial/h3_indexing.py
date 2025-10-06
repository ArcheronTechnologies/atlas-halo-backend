"""H3 spatial indexing service"""
import logging
import h3
from typing import List, Tuple, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class H3Cell(BaseModel):
    """H3 cell representation"""
    h3_index: str
    latitude: float
    longitude: float
    resolution: int = 8

class H3SpatialIndex:
    """H3 spatial indexing for geospatial queries"""

    def __init__(self, resolution: int = 8):
        self.resolution = resolution
        logger.info(f"H3SpatialIndex initialized (resolution={resolution})")

    def latlon_to_h3(self, lat: float, lon: float) -> str:
        """Convert lat/lon to H3 cell"""
        try:
            return h3.geo_to_h3(lat, lon, self.resolution)
        except:
            return None

    def h3_to_latlon(self, h3_address: str) -> Tuple[float, float]:
        """Convert H3 cell to lat/lon"""
        try:
            return h3.h3_to_geo(h3_address)
        except:
            return (None, None)

    def get_neighbors(self, h3_address: str, k: int = 1) -> List[str]:
        """Get neighboring H3 cells"""
        try:
            return list(h3.k_ring(h3_address, k))
        except:
            return []

    def get_cell(self, lat: float, lon: float) -> H3Cell:
        """Get H3 cell for coordinates"""
        h3_index = self.latlon_to_h3(lat, lon)
        return H3Cell(
            h3_index=h3_index,
            latitude=lat,
            longitude=lon,
            resolution=self.resolution
        )

class H3IndexingService(H3SpatialIndex):
    """Alias for backwards compatibility"""
    pass

_h3_indexing = None

def get_h3_indexing(resolution: int = 8) -> H3IndexingService:
    global _h3_indexing
    if _h3_indexing is None:
        _h3_indexing = H3IndexingService(resolution)
    return _h3_indexing
