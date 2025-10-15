/**
 * Main safety map screen with geofencing alerts and threat capture
 */

import React, {useEffect, useRef} from 'react';
import {
  View,
  StyleSheet,
  StatusBar,
  Platform,
} from 'react-native';
import {useDispatch, useSelector} from 'react-redux';

import {
  selectUserLocation,
  selectCurrentRiskLevel,
} from '@store/slices/mapSlice';
import {AppDispatch} from '@store/index';

import SafetyMapView from '@components/Map/SafetyMapView';
import LocationSearchBar from '@components/Search/LocationSearchBar';
import ThreatCaptureButton from '@components/Capture/ThreatCaptureButton';
import SafetyAlertSystem from '@components/Alerts/SafetyAlertSystem';
import ConnectionIndicator from '@components/Common/ConnectionIndicator';
import GeofencingService from '@services/GeofencingService';
import {useWebSocket} from '@hooks/useWebSocket';

const SafetyMapScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const userLocation = useSelector(selectUserLocation);
  const currentRiskLevel = useSelector(selectCurrentRiskLevel);

  const geofencingService = useRef(GeofencingService.getInstance());

  // Initialize WebSocket connection for real-time updates
  const {isConnected, connectionStatus} = useWebSocket({
    autoConnect: true,
    reconnectOnAppStateChange: true,
    locationUpdateInterval: 30000,
  });

  // Initialize geofencing when component mounts
  useEffect(() => {
    const initializeGeofencing = async () => {
      try {
        await geofencingService.current.startMonitoring();
        console.log('Geofencing service initialized successfully');
      } catch (error) {
        console.error('Failed to initialize geofencing service:', error);
      }
    };

    initializeGeofencing();

    // Cleanup on unmount
    return () => {
      geofencingService.current.stopMonitoring();
    };
  }, []);

  // Handle location selection from search
  const handleLocationSelect = (location: any) => {
    // Location selection handled by SafetyMapView
    console.log('Location selected:', location);
  };

  return (
    <View style={styles.container}>
      <StatusBar
        backgroundColor="transparent"
        barStyle="dark-content"
        translucent={true}
      />

      {/* Main map view */}
      <SafetyMapView
        showUserLocation={true}
        showSafetyZones={true}
        interactive={true}
        onLocationSelect={handleLocationSelect}
        style={styles.mapView}
      />

      {/* Safety alert system */}
      <SafetyAlertSystem style={styles.alertSystem} />

      {/* Location search bar */}
      <LocationSearchBar
        placeholder="Search for places to check safety..."
        onLocationSelect={handleLocationSelect}
        style={styles.searchBar}
      />

      {/* Connection indicator */}
      <ConnectionIndicator
        compact={true}
        style={styles.connectionIndicator}
      />

      {/* Threat capture button */}
      <ThreatCaptureButton
        size={60}
        emergencyMode={currentRiskLevel === 'critical'}
        style={styles.captureButton}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  mapView: {
    flex: 1,
  },
  alertSystem: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 1000,
  },
  searchBar: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 120 : 100,
    left: 16,
    right: 16,
    zIndex: 999,
  },
  connectionIndicator: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 60 : 40,
    right: 16,
    zIndex: 1001,
  },
  captureButton: {
    position: 'absolute',
    bottom: 30,
    right: 20,
    zIndex: 998,
  },
});

export default SafetyMapScreen;