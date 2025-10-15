/**
 * React hook for managing WebSocket connection and real-time updates
 */

import {useEffect, useRef, useState, useCallback} from 'react';
import {useDispatch, useSelector} from 'react-redux';
import {AppState} from 'react-native';

import WebSocketService from '@services/WebSocketService';
import {selectUserLocation} from '@store/slices/mapSlice';
import {AppDispatch} from '@store/index';

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectOnAppStateChange?: boolean;
  locationUpdateInterval?: number;
}

interface WebSocketHookReturn {
  connectionStatus: string;
  isConnected: boolean;
  connect: () => Promise<void>;
  disconnect: () => void;
  sendMessage: (type: string, data: any) => void;
  subscribe: (messageType: string, callback: (data: any) => void) => () => void;
  updateLocation: (latitude: number, longitude: number) => void;
  subscribeToLocationAlerts: (latitude: number, longitude: number, radius?: number) => void;
  requestRiskAssessment: (latitude: number, longitude: number) => void;
}

export const useWebSocket = (options: UseWebSocketOptions = {}): WebSocketHookReturn => {
  const {
    autoConnect = true,
    reconnectOnAppStateChange = true,
    locationUpdateInterval = 30000, // 30 seconds
  } = options;

  const dispatch = useDispatch<AppDispatch>();
  const userLocation = useSelector(selectUserLocation);

  // State
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [isConnected, setIsConnected] = useState(false);

  // Refs
  const wsService = useRef(WebSocketService.getInstance());
  const locationUpdateTimer = useRef<NodeJS.Timeout | null>(null);
  const appStateRef = useRef(AppState.currentState);

  // Connection status updates
  const handleConnectionStatusChange = useCallback(() => {
    const status = wsService.current.getConnectionStatus();
    setConnectionStatus(status);
    setIsConnected(status === 'connected');
  }, []);

  // Initialize WebSocket service
  useEffect(() => {
    const service = wsService.current;

    // Set up event listeners
    const unsubscribeConnected = service.onConnected(() => {
      console.log('WebSocket connected');
      handleConnectionStatusChange();
      startLocationUpdates();
    });

    const unsubscribeDisconnected = service.onDisconnected(() => {
      console.log('WebSocket disconnected');
      handleConnectionStatusChange();
      stopLocationUpdates();
    });

    const unsubscribeError = service.onError((error) => {
      console.error('WebSocket error:', error);
      handleConnectionStatusChange();
    });

    // Auto-connect if enabled
    if (autoConnect) {
      service.connect().catch(error => {
        console.error('Failed to auto-connect WebSocket:', error);
      });
    }

    // Initial status
    handleConnectionStatusChange();

    return () => {
      unsubscribeConnected();
      unsubscribeDisconnected();
      unsubscribeError();
      stopLocationUpdates();
    };
  }, [autoConnect, handleConnectionStatusChange]);

  // Handle app state changes
  useEffect(() => {
    if (!reconnectOnAppStateChange) return;

    const handleAppStateChange = (nextAppState: string) => {
      const currentState = appStateRef.current;
      appStateRef.current = nextAppState;

      if (currentState === 'background' && nextAppState === 'active') {
        // App came to foreground
        if (connectionStatus === 'disconnected') {
          console.log('Reconnecting WebSocket after app activation');
          wsService.current.connect().catch(error => {
            console.error('Failed to reconnect on app activation:', error);
          });
        }
      } else if (currentState === 'active' && nextAppState === 'background') {
        // App went to background - keep connection alive for notifications
        console.log('App went to background, maintaining WebSocket connection');
      }
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);

    return () => subscription?.remove();
  }, [connectionStatus, reconnectOnAppStateChange]);

  // Location-based subscriptions
  useEffect(() => {
    if (isConnected && userLocation) {
      // Subscribe to alerts for current location
      subscribeToLocationAlerts(
        userLocation.coordinates.latitude,
        userLocation.coordinates.longitude,
        1000 // 1km radius
      );
    }
  }, [isConnected, userLocation]);

  // Start periodic location updates
  const startLocationUpdates = useCallback(() => {
    if (locationUpdateTimer.current) {
      clearInterval(locationUpdateTimer.current);
    }

    locationUpdateTimer.current = setInterval(() => {
      if (userLocation && isConnected) {
        updateLocation(
          userLocation.coordinates.latitude,
          userLocation.coordinates.longitude
        );
      }
    }, locationUpdateInterval);
  }, [userLocation, isConnected, locationUpdateInterval]);

  // Stop location updates
  const stopLocationUpdates = useCallback(() => {
    if (locationUpdateTimer.current) {
      clearInterval(locationUpdateTimer.current);
      locationUpdateTimer.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(async (): Promise<void> => {
    try {
      await wsService.current.connect();
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      throw error;
    }
  }, []);

  // Disconnect from WebSocket
  const disconnect = useCallback((): void => {
    wsService.current.disconnect();
    stopLocationUpdates();
  }, [stopLocationUpdates]);

  // Send message
  const sendMessage = useCallback((type: string, data: any): void => {
    wsService.current.send({type, data});
  }, []);

  // Subscribe to message type
  const subscribe = useCallback((messageType: string, callback: (data: any) => void): (() => void) => {
    return wsService.current.subscribe(messageType, callback);
  }, []);

  // Update location
  const updateLocation = useCallback((latitude: number, longitude: number): void => {
    wsService.current.updateLocation(latitude, longitude);
  }, []);

  // Subscribe to location alerts
  const subscribeToLocationAlerts = useCallback((
    latitude: number,
    longitude: number,
    radius: number = 1000
  ): void => {
    wsService.current.subscribeToLocationAlerts(latitude, longitude, radius);
  }, []);

  // Request risk assessment
  const requestRiskAssessment = useCallback((latitude: number, longitude: number): void => {
    wsService.current.requestRiskAssessment(latitude, longitude);
  }, []);

  return {
    connectionStatus,
    isConnected,
    connect,
    disconnect,
    sendMessage,
    subscribe,
    updateLocation,
    subscribeToLocationAlerts,
    requestRiskAssessment,
  };
};

// Specialized hooks for specific use cases

/**
 * Hook for live incident monitoring
 */
export const useLiveIncidents = () => {
  const {subscribe} = useWebSocket();
  const [liveIncidents, setLiveIncidents] = useState<any[]>([]);

  useEffect(() => {
    const unsubscribe = subscribe('live_incident', (incident) => {
      setLiveIncidents(prev => [incident, ...prev.slice(0, 19)]); // Keep last 20
    });

    return unsubscribe;
  }, [subscribe]);

  return {liveIncidents};
};

/**
 * Hook for emergency broadcasts
 */
export const useEmergencyBroadcasts = () => {
  const {subscribe} = useWebSocket();
  const [emergencyAlerts, setEmergencyAlerts] = useState<any[]>([]);

  useEffect(() => {
    const unsubscribe = subscribe('emergency_broadcast', (alert) => {
      setEmergencyAlerts(prev => [alert, ...prev]);
    });

    return unsubscribe;
  }, [subscribe]);

  const clearEmergencyAlerts = useCallback(() => {
    setEmergencyAlerts([]);
  }, []);

  return {
    emergencyAlerts,
    clearEmergencyAlerts,
  };
};

/**
 * Hook for watchlist alerts
 */
export const useWatchlistAlerts = () => {
  const {subscribe} = useWebSocket();
  const [watchlistAlerts, setWatchlistAlerts] = useState<any[]>([]);

  useEffect(() => {
    const unsubscribe = subscribe('watchlist_alert', (alert) => {
      setWatchlistAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep last 10
    });

    return unsubscribe;
  }, [subscribe]);

  const clearWatchlistAlert = useCallback((alertId: string) => {
    setWatchlistAlerts(prev => prev.filter(alert => alert.id !== alertId));
  }, []);

  return {
    watchlistAlerts,
    clearWatchlistAlert,
  };
};

/**
 * Hook for real-time risk level monitoring
 */
export const useRiskLevelMonitoring = () => {
  const {subscribe, requestRiskAssessment} = useWebSocket();
  const userLocation = useSelector(selectUserLocation);
  const [currentRiskLevel, setCurrentRiskLevel] = useState<string>('safe');
  const [riskHistory, setRiskHistory] = useState<Array<{level: string, timestamp: Date}>>([]);

  useEffect(() => {
    const unsubscribeRiskUpdate = subscribe('risk_level_update', (update) => {
      const newLevel = update.riskLevel;
      setCurrentRiskLevel(newLevel);
      setRiskHistory(prev => [
        {level: newLevel, timestamp: new Date()},
        ...prev.slice(0, 23) // Keep last 24 updates
      ]);
    });

    const unsubscribeRiskResponse = subscribe('risk_assessment_response', (response) => {
      if (response.assessment?.riskLevel) {
        setCurrentRiskLevel(response.assessment.riskLevel);
      }
    });

    return () => {
      unsubscribeRiskUpdate();
      unsubscribeRiskResponse();
    };
  }, [subscribe]);

  // Request immediate risk assessment
  const refreshRiskLevel = useCallback(() => {
    if (userLocation) {
      requestRiskAssessment(
        userLocation.coordinates.latitude,
        userLocation.coordinates.longitude
      );
    }
  }, [userLocation, requestRiskAssessment]);

  return {
    currentRiskLevel,
    riskHistory,
    refreshRiskLevel,
  };
};

export default useWebSocket;