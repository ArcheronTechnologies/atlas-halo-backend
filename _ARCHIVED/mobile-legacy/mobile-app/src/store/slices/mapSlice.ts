/**
 * Map-related Redux slice for Atlas AI Mobile App
 */

import {createSlice, createAsyncThunk, PayloadAction} from '@reduxjs/toolkit';
import {
  RiskLevel,
  LocationInfo,
  SafetyZone,
  UserLocation,
  Coordinates,
  SafetyAssessment,
  SafetyAlert,
} from '@types/safety';
import {LocationSearchResult} from '@types/api';

// Async thunks for API calls
export const fetchSafetyZones = createAsyncThunk(
  'map/fetchSafetyZones',
  async (params: {center: Coordinates; radius: number}) => {
    // This would call the actual API
    // For now, return mock data
    const response = await fetch(
      `/api/mobile/safety-zones?lat=${params.center.latitude}&lng=${params.center.longitude}&radius=${params.radius}`
    );
    return response.json();
  }
);

export const searchLocations = createAsyncThunk(
  'map/searchLocations',
  async (query: string) => {
    const response = await fetch(
      `/api/mobile/location-search?query=${encodeURIComponent(query)}`
    );
    return response.json();
  }
);

export const fetchLocationSafety = createAsyncThunk(
  'map/fetchLocationSafety',
  async (location: LocationInfo) => {
    const response = await fetch(
      `/api/mobile/location-safety?lat=${location.coordinates.latitude}&lng=${location.coordinates.longitude}`
    );
    return response.json();
  }
);

// Map state interface
interface MapState {
  // Current map view
  region: {
    latitude: number;
    longitude: number;
    latitudeDelta: number;
    longitudeDelta: number;
  };

  // User location
  userLocation: UserLocation | null;
  isLocationEnabled: boolean;
  locationPermissionStatus: 'granted' | 'denied' | 'pending';

  // Safety zones and risk data
  safetyZones: SafetyZone[];
  currentRiskLevel: RiskLevel;

  // Location search
  searchQuery: string;
  searchResults: LocationSearchResult[];
  isSearching: boolean;

  // Selected location for safety check
  selectedLocation: LocationInfo | null;
  selectedLocationSafety: SafetyAssessment | null;

  // Map display settings
  mapStyle: 'standard' | 'satellite' | 'hybrid';
  showHistoricalData: boolean;
  showCommunityReports: boolean;
  zoomLevel: number;

  // Loading states
  isLoadingSafetyZones: boolean;
  isLoadingLocationSafety: boolean;

  // Safety alerts
  safetyAlerts: SafetyAlert[];
  lastZoneUpdate: Date | null;

  // Error handling
  error: string | null;
  lastUpdated: Date | null;
}

// Initial state
const initialState: MapState = {
  region: {
    latitude: 59.3293, // Stockholm center
    longitude: 18.0686,
    latitudeDelta: 0.05,
    longitudeDelta: 0.05,
  },
  userLocation: null,
  isLocationEnabled: false,
  locationPermissionStatus: 'pending',
  safetyZones: [],
  currentRiskLevel: RiskLevel.SAFE,
  searchQuery: '',
  searchResults: [],
  isSearching: false,
  selectedLocation: null,
  selectedLocationSafety: null,
  mapStyle: 'standard',
  showHistoricalData: true,
  showCommunityReports: true,
  zoomLevel: 12,
  isLoadingSafetyZones: false,
  isLoadingLocationSafety: false,
  safetyAlerts: [],
  lastZoneUpdate: null,
  error: null,
  lastUpdated: null,
};

// Map slice
const mapSlice = createSlice({
  name: 'map',
  initialState,
  reducers: {
    // User location actions
    setUserLocation: (state, action: PayloadAction<UserLocation>) => {
      state.userLocation = action.payload;
      state.isLocationEnabled = true;

      // Update current risk level based on user location
      const currentZone = state.safetyZones.find(zone =>
        isPointInZone(action.payload.coordinates, zone.coordinates)
      );
      if (currentZone) {
        state.currentRiskLevel = currentZone.riskLevel;
      }
    },

    setLocationPermissionStatus: (state, action: PayloadAction<'granted' | 'denied' | 'pending'>) => {
      state.locationPermissionStatus = action.payload;
      if (action.payload === 'denied') {
        state.isLocationEnabled = false;
        state.userLocation = null;
      }
    },

    // Map region actions
    setMapRegion: (state, action: PayloadAction<typeof initialState.region>) => {
      state.region = action.payload;
      state.zoomLevel = calculateZoomLevel(action.payload.latitudeDelta);
    },

    // Search actions
    setSearchQuery: (state, action: PayloadAction<string>) => {
      state.searchQuery = action.payload;
      if (!action.payload.trim()) {
        state.searchResults = [];
      }
    },

    clearSearchResults: (state) => {
      state.searchResults = [];
      state.searchQuery = '';
    },

    // Location selection actions
    setSelectedLocation: (state, action: PayloadAction<LocationInfo | null>) => {
      state.selectedLocation = action.payload;
      if (!action.payload) {
        state.selectedLocationSafety = null;
      }
    },

    // Map display settings
    setMapStyle: (state, action: PayloadAction<'standard' | 'satellite' | 'hybrid'>) => {
      state.mapStyle = action.payload;
    },

    toggleHistoricalData: (state) => {
      state.showHistoricalData = !state.showHistoricalData;
    },

    toggleCommunityReports: (state) => {
      state.showCommunityReports = !state.showCommunityReports;
    },

    // Risk level updates
    updateCurrentRiskLevel: (state, action: PayloadAction<RiskLevel>) => {
      state.currentRiskLevel = action.payload;
    },

    setCurrentRiskLevel: (state, action: PayloadAction<RiskLevel>) => {
      state.currentRiskLevel = action.payload;
    },

    updateUserLocation: (state, action: PayloadAction<UserLocation>) => {
      state.userLocation = action.payload;
      state.isLocationEnabled = true;
    },

    // Safety zones management
    updateSafetyZone: (state, action: PayloadAction<SafetyZone>) => {
      const index = state.safetyZones.findIndex(zone => zone.id === action.payload.id);
      if (index >= 0) {
        state.safetyZones[index] = action.payload;
      } else {
        state.safetyZones.push(action.payload);
      }
    },

    removeSafetyZone: (state, action: PayloadAction<string>) => {
      state.safetyZones = state.safetyZones.filter(zone => zone.id !== action.payload);
    },

    // Safety alerts management
    addSafetyAlert: (state, action: PayloadAction<SafetyAlert>) => {
      state.safetyAlerts.unshift(action.payload); // Add to beginning
      // Keep only last 20 alerts
      if (state.safetyAlerts.length > 20) {
        state.safetyAlerts = state.safetyAlerts.slice(0, 20);
      }
    },

    dismissSafetyAlert: (state, action: PayloadAction<string>) => {
      state.safetyAlerts = state.safetyAlerts.filter(alert => alert.id !== action.payload);
    },

    clearAllSafetyAlerts: (state) => {
      state.safetyAlerts = [];
    },

    // Error handling
    clearError: (state) => {
      state.error = null;
    },

    // Manual refresh
    refreshMapData: (state) => {
      state.lastUpdated = new Date();
    },
  },

  extraReducers: (builder) => {
    // Fetch safety zones
    builder
      .addCase(fetchSafetyZones.pending, (state) => {
        state.isLoadingSafetyZones = true;
        state.error = null;
      })
      .addCase(fetchSafetyZones.fulfilled, (state, action) => {
        state.isLoadingSafetyZones = false;
        state.safetyZones = action.payload.zones || [];
        state.lastUpdated = new Date();
      })
      .addCase(fetchSafetyZones.rejected, (state, action) => {
        state.isLoadingSafetyZones = false;
        state.error = action.error.message || 'Failed to load safety zones';
      });

    // Search locations
    builder
      .addCase(searchLocations.pending, (state) => {
        state.isSearching = true;
        state.error = null;
      })
      .addCase(searchLocations.fulfilled, (state, action) => {
        state.isSearching = false;
        state.searchResults = action.payload.results || [];
      })
      .addCase(searchLocations.rejected, (state, action) => {
        state.isSearching = false;
        state.error = action.error.message || 'Search failed';
      });

    // Fetch location safety
    builder
      .addCase(fetchLocationSafety.pending, (state) => {
        state.isLoadingLocationSafety = true;
        state.error = null;
      })
      .addCase(fetchLocationSafety.fulfilled, (state, action) => {
        state.isLoadingLocationSafety = false;
        state.selectedLocationSafety = action.payload.assessment;
      })
      .addCase(fetchLocationSafety.rejected, (state, action) => {
        state.isLoadingLocationSafety = false;
        state.error = action.error.message || 'Failed to load location safety';
      });
  },
});

// Helper functions
function isPointInZone(point: Coordinates, zoneCoordinates: Coordinates[]): boolean {
  // Simple point-in-polygon algorithm
  // In a real app, you'd use a more robust geospatial library
  let inside = false;
  const x = point.longitude;
  const y = point.latitude;

  for (let i = 0, j = zoneCoordinates.length - 1; i < zoneCoordinates.length; j = i++) {
    const xi = zoneCoordinates[i].longitude;
    const yi = zoneCoordinates[i].latitude;
    const xj = zoneCoordinates[j].longitude;
    const yj = zoneCoordinates[j].latitude;

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
}

function calculateZoomLevel(latitudeDelta: number): number {
  // Convert latitude delta to approximate zoom level
  return Math.round(Math.log2(360 / latitudeDelta));
}

// Export actions and reducer
export const {
  setUserLocation,
  setLocationPermissionStatus,
  setMapRegion,
  setSearchQuery,
  clearSearchResults,
  setSelectedLocation,
  setMapStyle,
  toggleHistoricalData,
  toggleCommunityReports,
  updateCurrentRiskLevel,
  setCurrentRiskLevel,
  updateUserLocation,
  updateSafetyZone,
  removeSafetyZone,
  addSafetyAlert,
  dismissSafetyAlert,
  clearAllSafetyAlerts,
  clearError,
  refreshMapData,
} = mapSlice.actions;

export default mapSlice.reducer;

// Selectors
export const selectMapRegion = (state: {map: MapState}) => state.map.region;
export const selectUserLocation = (state: {map: MapState}) => state.map.userLocation;
export const selectCurrentRiskLevel = (state: {map: MapState}) => state.map.currentRiskLevel;
export const selectSafetyZones = (state: {map: MapState}) => state.map.safetyZones;
export const selectSearchResults = (state: {map: MapState}) => state.map.searchResults;
export const selectSelectedLocation = (state: {map: MapState}) => state.map.selectedLocation;
export const selectSelectedLocationSafety = (state: {map: MapState}) => state.map.selectedLocationSafety;
export const selectIsLoadingSafetyZones = (state: {map: MapState}) => state.map.isLoadingSafetyZones;
export const selectMapError = (state: {map: MapState}) => state.map.error;
export const selectSafetyAlerts = (state: {map: MapState}) => state.map.safetyAlerts;
export const selectIsSearching = (state: {map: MapState}) => state.map.isSearching;