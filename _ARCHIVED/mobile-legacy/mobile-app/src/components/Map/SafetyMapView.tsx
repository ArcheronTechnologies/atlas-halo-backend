/**
 * Main safety map component with crime hotspot visualization
 */

import React, {useEffect, useRef, useState, useCallback} from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  Alert,
  Platform,
} from 'react-native';
import MapView, {
  PROVIDER_GOOGLE,
  Region,
  LatLng,
  Polygon,
  Marker,
  Circle,
} from 'react-native-maps';
import {useDispatch, useSelector} from 'react-redux';
import Geolocation from 'react-native-geolocation-service';

import {
  setUserLocation,
  setMapRegion,
  fetchSafetyZones,
  updateCurrentRiskLevel,
  selectMapRegion,
  selectUserLocation,
  selectSafetyZones,
  selectCurrentRiskLevel,
  selectIsLoadingSafetyZones,
} from '@store/slices/mapSlice';
import {RiskLevel, SafetyZone, UserLocation} from '@types/safety';
import {AppDispatch} from '@store/index';

import RiskZoneOverlay from './RiskZoneOverlay';
import UserLocationMarker from './UserLocationMarker';
import SafetyLegend from './SafetyLegend';
import LoadingOverlay from '@components/Common/LoadingOverlay';

const {width, height} = Dimensions.get('window');

interface SafetyMapViewProps {
  onLocationSelect?: (location: LatLng) => void;
  showUserLocation?: boolean;
  showSafetyZones?: boolean;
  interactive?: boolean;
  style?: any;
}

const SafetyMapView: React.FC<SafetyMapViewProps> = ({
  onLocationSelect,
  showUserLocation = true,
  showSafetyZones = true,
  interactive = true,
  style,
}) => {
  const dispatch = useDispatch<AppDispatch>();
  const mapRef = useRef<MapView>(null);

  // Redux state
  const mapRegion = useSelector(selectMapRegion);
  const userLocation = useSelector(selectUserLocation);
  const safetyZones = useSelector(selectSafetyZones);
  const currentRiskLevel = useSelector(selectCurrentRiskLevel);
  const isLoadingSafetyZones = useSelector(selectIsLoadingSafetyZones);

  // Local state
  const [isMapReady, setIsMapReady] = useState(false);
  const [userLocationCircle, setUserLocationCircle] = useState<{
    center: LatLng;
    radius: number;
  } | null>(null);

  // Initialize location tracking
  useEffect(() => {
    if (showUserLocation) {
      requestLocationPermission();
    }
  }, [showUserLocation]);

  // Fetch safety zones when map region changes
  useEffect(() => {
    if (isMapReady && showSafetyZones) {
      const timer = setTimeout(() => {
        dispatch(fetchSafetyZones({
          center: {
            latitude: mapRegion.latitude,
            longitude: mapRegion.longitude,
          },
          radius: calculateRadius(mapRegion),
        }));
      }, 500); // Debounce API calls

      return () => clearTimeout(timer);
    }
  }, [mapRegion, isMapReady, showSafetyZones, dispatch]);

  // Request location permission and start tracking
  const requestLocationPermission = useCallback(async () => {
    try {
      if (Platform.OS === 'android') {
        const granted = await Geolocation.requestAuthorization('whenInUse');
        if (granted === 'granted') {
          startLocationTracking();
        } else {
          Alert.alert(
            'Location Permission',
            'Location access is required to show your current position and nearby safety information.',
            [
              {text: 'Cancel', style: 'cancel'},
              {text: 'Settings', onPress: () => {/* Open settings */}},
            ]
          );
        }
      } else {
        // iOS
        Geolocation.requestAuthorization('whenInUse');
        startLocationTracking();
      }
    } catch (error) {
      console.error('Location permission error:', error);
    }
  }, []);

  // Start location tracking
  const startLocationTracking = useCallback(() => {
    Geolocation.getCurrentPosition(
      (position) => {
        const userLocationData: UserLocation = {
          coordinates: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
          accuracy: position.coords.accuracy,
          timestamp: new Date(),
          isMoving: false,
          speed: position.coords.speed || undefined,
          heading: position.coords.heading || undefined,
        };

        dispatch(setUserLocation(userLocationData));

        // Center map on user location (first time only)
        if (!userLocation) {
          const newRegion = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            latitudeDelta: 0.01,
            longitudeDelta: 0.01,
          };
          dispatch(setMapRegion(newRegion));
          mapRef.current?.animateToRegion(newRegion, 1000);
        }

        // Set user location circle for accuracy visualization
        setUserLocationCircle({
          center: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
          radius: position.coords.accuracy,
        });
      },
      (error) => {
        console.error('Location error:', error);
        Alert.alert(
          'Location Error',
          'Unable to get your current location. Please check your location settings.',
        );
      },
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 10000,
        distanceFilter: 10, // Update every 10 meters
      }
    );

    // Watch position for continuous updates
    const watchId = Geolocation.watchPosition(
      (position) => {
        const userLocationData: UserLocation = {
          coordinates: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
          accuracy: position.coords.accuracy,
          timestamp: new Date(),
          isMoving: (position.coords.speed || 0) > 0.5, // Moving if speed > 0.5 m/s
          speed: position.coords.speed || undefined,
          heading: position.coords.heading || undefined,
        };

        dispatch(setUserLocation(userLocationData));

        // Update user location circle
        setUserLocationCircle({
          center: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
          radius: position.coords.accuracy,
        });

        // Check if user entered a new risk zone
        checkUserRiskZone(userLocationData.coordinates);
      },
      (error) => {
        console.error('Location watch error:', error);
      },
      {
        enableHighAccuracy: true,
        distanceFilter: 10,
        interval: 5000, // Update every 5 seconds
        fastestInterval: 2000,
      }
    );

    return () => {
      Geolocation.clearWatch(watchId);
    };
  }, [dispatch, userLocation]);

  // Check if user entered a new risk zone
  const checkUserRiskZone = useCallback((location: LatLng) => {
    if (!safetyZones.length) return;

    for (const zone of safetyZones) {
      if (isPointInPolygon(location, zone.coordinates)) {
        if (zone.riskLevel !== currentRiskLevel) {
          dispatch(updateCurrentRiskLevel(zone.riskLevel));

          // Trigger alert for high-risk zones
          if (zone.riskLevel === RiskLevel.HIGH || zone.riskLevel === RiskLevel.CRITICAL) {
            Alert.alert(
              'Safety Alert',
              `You have entered a ${zone.riskLevel} risk area. Stay alert and consider moving to a safer location.`,
              [
                {text: 'OK', style: 'default'},
                {text: 'View Details', onPress: () => {/* Show zone details */}},
              ]
            );
          }
        }
        break;
      }
    }
  }, [safetyZones, currentRiskLevel, dispatch]);

  // Handle map region changes
  const handleRegionChangeComplete = useCallback((region: Region) => {
    dispatch(setMapRegion(region));
  }, [dispatch]);

  // Handle map press
  const handleMapPress = useCallback((event: any) => {
    if (onLocationSelect && interactive) {
      onLocationSelect(event.nativeEvent.coordinate);
    }
  }, [onLocationSelect, interactive]);

  // Handle map ready
  const handleMapReady = useCallback(() => {
    setIsMapReady(true);
  }, []);

  // Center map on user location
  const centerOnUserLocation = useCallback(() => {
    if (userLocation && mapRef.current) {
      const region = {
        latitude: userLocation.coordinates.latitude,
        longitude: userLocation.coordinates.longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      };
      mapRef.current.animateToRegion(region, 1000);
    }
  }, [userLocation]);

  return (
    <View style={[styles.container, style]}>
      <MapView
        ref={mapRef}
        provider={PROVIDER_GOOGLE}
        style={styles.map}
        initialRegion={mapRegion}
        onRegionChangeComplete={handleRegionChangeComplete}
        onPress={handleMapPress}
        onMapReady={handleMapReady}
        showsUserLocation={false} // We'll use custom user location marker
        showsMyLocationButton={false}
        showsCompass={true}
        showsScale={true}
        rotateEnabled={interactive}
        scrollEnabled={interactive}
        zoomEnabled={interactive}
        pitchEnabled={interactive}
        toolbarEnabled={false}
        loadingEnabled={true}
        mapType="standard"
      >
        {/* Safety zone overlays */}
        {showSafetyZones && safetyZones.map((zone) => (
          <RiskZoneOverlay
            key={zone.id}
            zone={zone}
          />
        ))}

        {/* User location accuracy circle */}
        {showUserLocation && userLocationCircle && (
          <Circle
            center={userLocationCircle.center}
            radius={userLocationCircle.radius}
            strokeColor="rgba(0, 122, 255, 0.3)"
            fillColor="rgba(0, 122, 255, 0.1)"
            strokeWidth={1}
          />
        )}

        {/* User location marker */}
        {showUserLocation && userLocation && (
          <UserLocationMarker
            location={userLocation}
            riskLevel={currentRiskLevel}
          />
        )}
      </MapView>

      {/* Safety legend */}
      <SafetyLegend style={styles.legend} />

      {/* Loading overlay */}
      {isLoadingSafetyZones && (
        <LoadingOverlay
          visible={true}
          message="Loading safety data..."
        />
      )}
    </View>
  );
};

// Helper functions
function calculateRadius(region: Region): number {
  // Calculate approximate radius in meters based on map region
  const latRad = region.latitude * Math.PI / 180;
  const degLen = 111320 * Math.cos(latRad);
  const deltaLng = region.longitudeDelta;
  const deltaLat = region.latitudeDelta;

  const radiusLng = (deltaLng * degLen) / 2;
  const radiusLat = (deltaLat * 111320) / 2;

  return Math.max(radiusLng, radiusLat);
}

function isPointInPolygon(point: LatLng, polygon: LatLng[]): boolean {
  let inside = false;
  const x = point.longitude;
  const y = point.latitude;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].longitude;
    const yi = polygon[i].latitude;
    const xj = polygon[j].longitude;
    const yj = polygon[j].latitude;

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    width: width,
    height: height,
  },
  legend: {
    position: 'absolute',
    top: 60,
    right: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 8,
    padding: 12,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
});

export default SafetyMapView;