/**
 * Geofencing service for monitoring user location and triggering safety alerts
 */

import Geolocation from '@react-native-community/geolocation';
import BackgroundJob from 'react-native-background-job';
import PushNotification from 'react-native-push-notification';
import {store} from '@store/index';
import {
  setCurrentRiskLevel,
  addSafetyAlert,
  updateUserLocation,
} from '@store/slices/mapSlice';
import {UserLocation, SafetyZone, RiskLevel, SafetyAlert} from '@types/safety';

interface GeofenceRegion {
  id: string;
  latitude: number;
  longitude: number;
  radius: number;
  riskLevel: RiskLevel;
  isActive: boolean;
  enteredAt?: Date;
}

interface RiskZonePredict {
  zoneId: string;
  currentRisk: RiskLevel;
  predictedRisk: RiskLevel;
  timeToChange: number; // minutes
  confidence: number;
}

class GeofencingService {
  private static instance: GeofencingService;
  private watchId: number | null = null;
  private activeGeofences: Map<string, GeofenceRegion> = new Map();
  private lastKnownLocation: UserLocation | null = null;
  private alertCooldowns: Map<string, number> = new Map();
  private backgroundJobId: string | null = null;

  // Configuration
  private readonly LOCATION_UPDATE_INTERVAL = 30000; // 30 seconds
  private readonly GEOFENCE_CHECK_INTERVAL = 60000; // 1 minute
  private readonly ALERT_COOLDOWN_DURATION = 300000; // 5 minutes
  private readonly RISK_PREDICTION_RADIUS = 1000; // 1km radius for predictions

  public static getInstance(): GeofencingService {
    if (!GeofencingService.instance) {
      GeofencingService.instance = new GeofencingService();
    }
    return GeofencingService.instance;
  }

  private constructor() {
    this.initializePushNotifications();
  }

  /**
   * Initialize push notification configuration
   */
  private initializePushNotifications(): void {
    PushNotification.configure({
      onNotification: (notification) => {
        console.log('Safety notification received:', notification);
      },
      requestPermissions: true,
    });

    PushNotification.createChannel(
      {
        channelId: 'safety-alerts',
        channelName: 'Safety Alerts',
        channelDescription: 'Critical safety alerts and zone warnings',
        importance: 4,
        vibrate: true,
      },
      (created) => console.log(`Safety channel created: ${created}`)
    );
  }

  /**
   * Start location monitoring and geofencing
   */
  public async startMonitoring(): Promise<void> {
    try {
      // Request location permissions
      const hasPermission = await this.requestLocationPermission();
      if (!hasPermission) {
        throw new Error('Location permission denied');
      }

      // Start location updates
      this.startLocationTracking();

      // Start background monitoring
      this.startBackgroundMonitoring();

      // Initial safety zone setup
      await this.initializeSafetyZones();

      console.log('Geofencing service started successfully');
    } catch (error) {
      console.error('Failed to start geofencing service:', error);
      throw error;
    }
  }

  /**
   * Stop all monitoring
   */
  public stopMonitoring(): void {
    if (this.watchId !== null) {
      Geolocation.clearWatch(this.watchId);
      this.watchId = null;
    }

    if (this.backgroundJobId) {
      BackgroundJob.stop({
        jobKey: this.backgroundJobId,
      });
      this.backgroundJobId = null;
    }

    this.activeGeofences.clear();
    console.log('Geofencing service stopped');
  }

  /**
   * Request location permission
   */
  private async requestLocationPermission(): Promise<boolean> {
    return new Promise((resolve) => {
      Geolocation.requestAuthorization(
        () => resolve(true),
        () => resolve(false)
      );
    });
  }

  /**
   * Start continuous location tracking
   */
  private startLocationTracking(): void {
    this.watchId = Geolocation.watchPosition(
      (position) => {
        const location: UserLocation = {
          coordinates: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
          accuracy: position.coords.accuracy,
          timestamp: new Date(position.timestamp),
        };

        this.handleLocationUpdate(location);
      },
      (error) => {
        console.error('Location tracking error:', error);
      },
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 10000,
        distanceFilter: 10, // Update every 10 meters
      }
    );
  }

  /**
   * Start background monitoring for when app is not active
   */
  private startBackgroundMonitoring(): void {
    this.backgroundJobId = BackgroundJob.start({
      jobKey: 'safetyMonitoring',
      period: this.GEOFENCE_CHECK_INTERVAL,
      callback: () => {
        this.performBackgroundSafetyCheck();
      },
    });
  }

  /**
   * Handle location updates
   */
  private async handleLocationUpdate(location: UserLocation): Promise<void> {
    this.lastKnownLocation = location;

    // Update Redux store
    store.dispatch(updateUserLocation(location));

    // Check geofences
    await this.checkGeofences(location);

    // Check for risk predictions
    await this.checkRiskPredictions(location);

    // Update safety zones if needed
    await this.updateSafetyZones(location);
  }

  /**
   * Initialize safety zones around user's location
   */
  private async initializeSafetyZones(): Promise<void> {
    if (!this.lastKnownLocation) return;

    try {
      const response = await fetch('/api/mobile/safety-zones', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: this.lastKnownLocation.coordinates.latitude,
          longitude: this.lastKnownLocation.coordinates.longitude,
          radius: 5000, // 5km radius
        }),
      });

      const result = await response.json();

      if (result.success) {
        result.data.safetyZones.forEach((zone: SafetyZone) => {
          this.addGeofence(zone);
        });
      }
    } catch (error) {
      console.error('Failed to initialize safety zones:', error);
    }
  }

  /**
   * Add a geofence for monitoring
   */
  private addGeofence(zone: SafetyZone): void {
    const geofence: GeofenceRegion = {
      id: zone.id,
      latitude: zone.coordinates.latitude,
      longitude: zone.coordinates.longitude,
      radius: zone.radius,
      riskLevel: zone.riskLevel,
      isActive: true,
    };

    this.activeGeofences.set(zone.id, geofence);
  }

  /**
   * Check if user has entered/exited any geofences
   */
  private async checkGeofences(location: UserLocation): Promise<void> {
    const currentTime = Date.now();

    for (const [zoneId, geofence] of this.activeGeofences) {
      const distance = this.calculateDistance(
        location.coordinates.latitude,
        location.coordinates.longitude,
        geofence.latitude,
        geofence.longitude
      );

      const isInside = distance <= geofence.radius;
      const wasInside = geofence.enteredAt !== undefined;

      // Entered geofence
      if (isInside && !wasInside) {
        geofence.enteredAt = new Date();
        await this.handleGeofenceEntry(geofence, location);
      }
      // Exited geofence
      else if (!isInside && wasInside) {
        delete geofence.enteredAt;
        await this.handleGeofenceExit(geofence, location);
      }
    }
  }

  /**
   * Handle geofence entry
   */
  private async handleGeofenceEntry(
    geofence: GeofenceRegion,
    location: UserLocation
  ): Promise<void> {
    console.log(`Entered geofence: ${geofence.id} (Risk: ${geofence.riskLevel})`);

    // Update current risk level
    store.dispatch(setCurrentRiskLevel(geofence.riskLevel));

    // Send alert for high-risk zones
    if (this.shouldSendAlert(geofence.id, geofence.riskLevel)) {
      await this.sendRiskZoneAlert(geofence, 'entry');
    }
  }

  /**
   * Handle geofence exit
   */
  private async handleGeofenceExit(
    geofence: GeofenceRegion,
    location: UserLocation
  ): Promise<void> {
    console.log(`Exited geofence: ${geofence.id}`);

    // Send alert for leaving high-risk zones
    if (geofence.riskLevel === RiskLevel.HIGH || geofence.riskLevel === RiskLevel.CRITICAL) {
      await this.sendRiskZoneAlert(geofence, 'exit');
    }
  }

  /**
   * Check for risk predictions (zones about to become dangerous)
   */
  private async checkRiskPredictions(location: UserLocation): Promise<void> {
    try {
      const response = await fetch('/api/mobile/risk-predictions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: location.coordinates.latitude,
          longitude: location.coordinates.longitude,
          radius: this.RISK_PREDICTION_RADIUS,
          timeHorizon: 60, // 1 hour prediction
        }),
      });

      const result = await response.json();

      if (result.success) {
        for (const prediction of result.data.predictions) {
          await this.handleRiskPrediction(prediction, location);
        }
      }
    } catch (error) {
      console.error('Failed to check risk predictions:', error);
    }
  }

  /**
   * Handle risk prediction alerts
   */
  private async handleRiskPrediction(
    prediction: RiskZonePredict,
    location: UserLocation
  ): Promise<void> {
    // Only alert for significant risk increases
    if (this.getRiskLevelValue(prediction.predictedRisk) <=
        this.getRiskLevelValue(prediction.currentRisk)) {
      return;
    }

    // Check if we should send prediction alert
    if (this.shouldSendAlert(`prediction_${prediction.zoneId}`, prediction.predictedRisk)) {
      await this.sendPredictionAlert(prediction);
    }
  }

  /**
   * Send risk zone alert
   */
  private async sendRiskZoneAlert(
    geofence: GeofenceRegion,
    type: 'entry' | 'exit'
  ): Promise<void> {
    const alertId = `${geofence.id}_${type}`;

    const alert: SafetyAlert = {
      id: alertId,
      type: type === 'entry' ? 'zone_entry' : 'zone_exit',
      severity: geofence.riskLevel,
      title: type === 'entry' ? 'Entered High-Risk Zone' : 'Exited High-Risk Zone',
      message: this.getZoneAlertMessage(geofence, type),
      location: {
        latitude: geofence.latitude,
        longitude: geofence.longitude,
      },
      timestamp: new Date(),
    };

    // Add to Redux store
    store.dispatch(addSafetyAlert(alert));

    // Send push notification
    PushNotification.localNotification({
      channelId: 'safety-alerts',
      title: alert.title,
      message: alert.message,
      priority: geofence.riskLevel === RiskLevel.CRITICAL ? 'high' : 'default',
      vibrate: true,
      playSound: true,
    });

    // Set cooldown
    this.alertCooldowns.set(geofence.id, Date.now());
  }

  /**
   * Send prediction alert
   */
  private async sendPredictionAlert(prediction: RiskZonePredict): Promise<void> {
    const alert: SafetyAlert = {
      id: `prediction_${prediction.zoneId}`,
      type: 'risk_prediction',
      severity: prediction.predictedRisk,
      title: 'Risk Level Increasing',
      message: `This area's risk level may increase to ${prediction.predictedRisk.toUpperCase()} in ${prediction.timeToChange} minutes.`,
      timestamp: new Date(),
    };

    store.dispatch(addSafetyAlert(alert));

    PushNotification.localNotification({
      channelId: 'safety-alerts',
      title: alert.title,
      message: alert.message,
      priority: 'default',
    });

    this.alertCooldowns.set(`prediction_${prediction.zoneId}`, Date.now());
  }

  /**
   * Check if we should send an alert (respects cooldowns)
   */
  private shouldSendAlert(alertKey: string, riskLevel: RiskLevel): boolean {
    const lastAlert = this.alertCooldowns.get(alertKey);
    const now = Date.now();

    // No cooldown for critical alerts
    if (riskLevel === RiskLevel.CRITICAL) {
      return true;
    }

    // Check cooldown for other alerts
    if (lastAlert && (now - lastAlert) < this.ALERT_COOLDOWN_DURATION) {
      return false;
    }

    return riskLevel === RiskLevel.HIGH || riskLevel === RiskLevel.MODERATE;
  }

  /**
   * Update safety zones periodically
   */
  private async updateSafetyZones(location: UserLocation): Promise<void> {
    // Update zones every 5 minutes
    const lastUpdate = store.getState().map.lastZoneUpdate;
    if (lastUpdate && (Date.now() - lastUpdate.getTime()) < 300000) {
      return;
    }

    await this.initializeSafetyZones();
  }

  /**
   * Background safety check when app is not active
   */
  private async performBackgroundSafetyCheck(): Promise<void> {
    if (!this.lastKnownLocation) return;

    try {
      // Get current location
      const currentLocation = await this.getCurrentLocation();
      if (currentLocation) {
        await this.handleLocationUpdate(currentLocation);
      }
    } catch (error) {
      console.error('Background safety check failed:', error);
    }
  }

  /**
   * Get current location promise-based
   */
  private getCurrentLocation(): Promise<UserLocation | null> {
    return new Promise((resolve) => {
      Geolocation.getCurrentPosition(
        (position) => {
          resolve({
            coordinates: {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
            },
            accuracy: position.coords.accuracy,
            timestamp: new Date(position.timestamp),
          });
        },
        () => resolve(null),
        {
          enableHighAccuracy: false,
          timeout: 10000,
          maximumAge: 30000,
        }
      );
    });
  }

  /**
   * Calculate distance between two coordinates
   */
  private calculateDistance(
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number {
    const R = 6371000; // Earth's radius in meters
    const dLat = this.toRadian(lat2 - lat1);
    const dLon = this.toRadian(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.toRadian(lat1)) *
        Math.cos(this.toRadian(lat2)) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  /**
   * Convert degrees to radians
   */
  private toRadian(degree: number): number {
    return (degree * Math.PI) / 180;
  }

  /**
   * Get numeric value for risk level comparison
   */
  private getRiskLevelValue(riskLevel: RiskLevel): number {
    switch (riskLevel) {
      case RiskLevel.SAFE: return 0;
      case RiskLevel.LOW: return 1;
      case RiskLevel.MODERATE: return 2;
      case RiskLevel.HIGH: return 3;
      case RiskLevel.CRITICAL: return 4;
      default: return 0;
    }
  }

  /**
   * Get zone alert message
   */
  private getZoneAlertMessage(geofence: GeofenceRegion, type: 'entry' | 'exit'): string {
    if (type === 'entry') {
      switch (geofence.riskLevel) {
        case RiskLevel.HIGH:
          return 'You have entered a high-risk area. Stay alert and consider leaving if possible.';
        case RiskLevel.CRITICAL:
          return 'CRITICAL: You are in a dangerous area. Leave immediately and contact authorities if needed.';
        case RiskLevel.MODERATE:
          return 'You have entered a moderate-risk area. Stay aware of your surroundings.';
        default:
          return 'You have entered a monitored safety zone.';
      }
    } else {
      return 'You have left the risk zone. Stay safe.';
    }
  }
}

export default GeofencingService;