"""H3 utility functions for spatial indexing"""
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

def latlon_to_h3(lat: float, lon: float, resolution: int = 8) -> str:
    """Convert lat/lon to H3 cell ID"""
    if not H3_AVAILABLE:
        return None
    try:
        return h3.geo_to_h3(lat, lon, resolution)
    except:
        return None

def h3_to_latlon(h3_address: str) -> tuple:
    """Convert H3 cell ID to lat/lon"""
    if not H3_AVAILABLE:
        return (None, None)
    try:
        lat, lon = h3.h3_to_geo(h3_address)
        return (lat, lon)
    except:
        return (None, None)
