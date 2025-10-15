/**
 * WebSocket service for real-time safety alerts and live communication
 */

import {store} from '@store/index';
import {
  addSafetyAlert,
  setCurrentRiskLevel,
  updateSafetyZone,
} from '@store/slices/mapSlice';
import {SafetyAlert, SafetyZone, RiskLevel} from '@types/safety';
import PushNotification from 'react-native-push-notification';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
  id?: string;
}

interface ConnectionConfig {
  url: string;
  protocols?: string[];
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
}

class WebSocketService {
  private static instance: WebSocketService;
  private socket: WebSocket | null = null;
  private config: ConnectionConfig;
  private reconnectCount = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimer: NodeJS.Timeout | null = null;
  private isConnecting = false;
  private isManualDisconnect = false;
  private messageQueue: WebSocketMessage[] = [];
  private subscriptions = new Map<string, Set<(data: any) => void>>();

  // Event handlers
  private onConnectedCallbacks = new Set<() => void>();
  private onDisconnectedCallbacks = new Set<() => void>();
  private onErrorCallbacks = new Set<(error: Event) => void>();

  public static getInstance(config?: Partial<ConnectionConfig>): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService(config);
    }
    return WebSocketService.instance;
  }

  private constructor(config?: Partial<ConnectionConfig>) {
    this.config = {
      url: 'wss://api.atlas-ai.com/ws/mobile',
      protocols: [],
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      ...config,
    };

    // Handle app state changes
    this.setupAppStateHandlers();
  }

  /**
   * Connect to WebSocket server
   */
  public async connect(): Promise<void> {
    if (this.socket?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    if (this.isConnecting) {
      console.log('WebSocket connection already in progress');
      return;
    }

    this.isConnecting = true;
    this.isManualDisconnect = false;

    try {
      console.log(`Connecting to WebSocket: ${this.config.url}`);

      this.socket = new WebSocket(this.config.url, this.config.protocols);

      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);

      // Connection timeout
      this.connectionTimer = setTimeout(() => {
        if (this.socket?.readyState !== WebSocket.OPEN) {
          console.log('WebSocket connection timeout');
          this.socket?.close();
        }
      }, 10000);

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.isConnecting = false;
      this.handleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    this.isManualDisconnect = true;
    this.isConnecting = false;
    this.reconnectCount = 0;

    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }

    console.log('WebSocket manually disconnected');
  }

  /**
   * Send message to server
   */
  public send(message: Omit<WebSocketMessage, 'timestamp' | 'id'>): void {
    const fullMessage: WebSocketMessage = {
      ...message,
      timestamp: new Date().toISOString(),
      id: this.generateMessageId(),
    };

    if (this.socket?.readyState === WebSocket.OPEN) {
      try {
        this.socket.send(JSON.stringify(fullMessage));
        console.log('WebSocket message sent:', fullMessage.type);
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        this.queueMessage(fullMessage);
      }
    } else {
      console.log('WebSocket not connected, queuing message');
      this.queueMessage(fullMessage);
    }
  }

  /**
   * Subscribe to specific message types
   */
  public subscribe(messageType: string, callback: (data: any) => void): () => void {
    if (!this.subscriptions.has(messageType)) {
      this.subscriptions.set(messageType, new Set());
    }

    this.subscriptions.get(messageType)!.add(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.subscriptions.get(messageType);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscriptions.delete(messageType);
        }
      }
    };
  }

  /**
   * Add connection event listeners
   */
  public onConnected(callback: () => void): () => void {
    this.onConnectedCallbacks.add(callback);
    return () => this.onConnectedCallbacks.delete(callback);
  }

  public onDisconnected(callback: () => void): () => void {
    this.onDisconnectedCallbacks.add(callback);
    return () => this.onDisconnectedCallbacks.delete(callback);
  }

  public onError(callback: (error: Event) => void): () => void {
    this.onErrorCallbacks.add(callback);
    return () => this.onErrorCallbacks.delete(callback);
  }

  /**
   * Get connection status
   */
  public getConnectionStatus(): string {
    if (!this.socket) return 'disconnected';

    switch (this.socket.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'unknown';
    }
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen(): void {
    console.log('WebSocket connected successfully');
    this.isConnecting = false;
    this.reconnectCount = 0;

    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }

    // Start heartbeat
    this.startHeartbeat();

    // Send authentication and setup messages
    this.authenticateConnection();

    // Send queued messages
    this.flushMessageQueue();

    // Notify listeners
    this.onConnectedCallbacks.forEach(callback => callback());
  }

  /**
   * Handle WebSocket message event
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      console.log('WebSocket message received:', message.type);

      this.processMessage(message);

      // Notify subscribers
      const callbacks = this.subscriptions.get(message.type);
      if (callbacks) {
        callbacks.forEach(callback => callback(message.data));
      }

    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  /**
   * Handle WebSocket close event
   */
  private handleClose(event: CloseEvent): void {
    console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
    this.isConnecting = false;

    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    // Notify listeners
    this.onDisconnectedCallbacks.forEach(callback => callback());

    // Attempt reconnection if not manual disconnect
    if (!this.isManualDisconnect) {
      this.handleReconnect();
    }
  }

  /**
   * Handle WebSocket error event
   */
  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.onErrorCallbacks.forEach(callback => callback(event));
  }

  /**
   * Process incoming WebSocket messages
   */
  private processMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'safety_alert':
        this.handleSafetyAlert(message.data);
        break;

      case 'risk_level_update':
        this.handleRiskLevelUpdate(message.data);
        break;

      case 'zone_update':
        this.handleZoneUpdate(message.data);
        break;

      case 'live_incident':
        this.handleLiveIncident(message.data);
        break;

      case 'watchlist_alert':
        this.handleWatchlistAlert(message.data);
        break;

      case 'emergency_broadcast':
        this.handleEmergencyBroadcast(message.data);
        break;

      case 'heartbeat_response':
        console.log('Heartbeat acknowledged');
        break;

      default:
        console.log(`Unknown message type: ${message.type}`);
    }
  }

  /**
   * Handle safety alert messages
   */
  private handleSafetyAlert(data: any): void {
    const alert: SafetyAlert = {
      id: data.id || this.generateMessageId(),
      type: data.type,
      title: data.title,
      message: data.message,
      severity: data.severity as RiskLevel,
      location: data.location,
      timestamp: new Date(data.timestamp),
    };

    // Add to Redux store
    store.dispatch(addSafetyAlert(alert));

    // Send push notification
    this.sendPushNotification(alert);
  }

  /**
   * Handle risk level updates
   */
  private handleRiskLevelUpdate(data: any): void {
    const newRiskLevel = data.riskLevel as RiskLevel;
    store.dispatch(setCurrentRiskLevel(newRiskLevel));

    // Send notification for significant risk increases
    if (this.isSignificantRiskIncrease(newRiskLevel)) {
      PushNotification.localNotification({
        title: 'Risk Level Changed',
        message: `Your current area risk level is now ${newRiskLevel.toUpperCase()}`,
        channelId: 'safety-alerts',
      });
    }
  }

  /**
   * Handle safety zone updates
   */
  private handleZoneUpdate(data: any): void {
    const zone: SafetyZone = {
      id: data.id,
      coordinates: data.coordinates,
      radius: data.radius,
      riskLevel: data.riskLevel as RiskLevel,
      name: data.name,
      description: data.description,
      lastUpdated: new Date(data.lastUpdated),
      incidentCount: data.incidentCount,
      activeIncidents: data.activeIncidents,
    };

    store.dispatch(updateSafetyZone(zone));
  }

  /**
   * Handle live incident reports
   */
  private handleLiveIncident(data: any): void {
    const alert: SafetyAlert = {
      id: data.id,
      type: 'live_incident',
      title: 'Live Incident Reported',
      message: `${data.incidentType} reported nearby: ${data.description}`,
      severity: data.severity as RiskLevel,
      location: data.location,
      timestamp: new Date(data.timestamp),
    };

    store.dispatch(addSafetyAlert(alert));
    this.sendPushNotification(alert);
  }

  /**
   * Handle watchlist alerts
   */
  private handleWatchlistAlert(data: any): void {
    const alert: SafetyAlert = {
      id: data.id,
      type: 'watchlist_alert',
      title: `${data.locationAlias} Risk Update`,
      message: data.message,
      severity: data.riskLevel as RiskLevel,
      location: data.location,
      timestamp: new Date(data.timestamp),
    };

    store.dispatch(addSafetyAlert(alert));
    this.sendPushNotification(alert);
  }

  /**
   * Handle emergency broadcasts
   */
  private handleEmergencyBroadcast(data: any): void {
    const alert: SafetyAlert = {
      id: data.id,
      type: 'emergency_broadcast',
      title: 'EMERGENCY ALERT',
      message: data.message,
      severity: RiskLevel.CRITICAL,
      location: data.location,
      timestamp: new Date(data.timestamp),
    };

    store.dispatch(addSafetyAlert(alert));

    // High priority notification for emergencies
    PushNotification.localNotification({
      title: alert.title,
      message: alert.message,
      channelId: 'safety-alerts',
      priority: 'high',
      vibrate: true,
      playSound: true,
      ongoing: true,
    });
  }

  /**
   * Send push notification for alert
   */
  private sendPushNotification(alert: SafetyAlert): void {
    const priority = alert.severity === RiskLevel.CRITICAL ? 'high' : 'default';

    PushNotification.localNotification({
      title: alert.title,
      message: alert.message,
      channelId: 'safety-alerts',
      priority: priority,
      vibrate: alert.severity === RiskLevel.CRITICAL || alert.severity === RiskLevel.HIGH,
      playSound: true,
    });
  }

  /**
   * Authenticate WebSocket connection
   */
  private authenticateConnection(): void {
    // Get user location for context
    const state = store.getState();
    const userLocation = state.map.userLocation;

    this.send({
      type: 'authenticate',
      data: {
        deviceId: this.getDeviceId(),
        location: userLocation?.coordinates,
        capabilities: ['safety_alerts', 'live_incidents', 'watchlist_alerts'],
        timestamp: new Date().toISOString(),
      },
    });
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.send({
          type: 'heartbeat',
          data: {timestamp: new Date().toISOString()},
        });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Handle reconnection logic
   */
  private handleReconnect(): void {
    if (this.isManualDisconnect || this.reconnectCount >= this.config.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached or manual disconnect');
      return;
    }

    this.reconnectCount++;
    const delay = Math.min(this.config.reconnectInterval * this.reconnectCount, 30000);

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectCount})`);

    setTimeout(() => {
      if (!this.isManualDisconnect) {
        this.connect();
      }
    }, delay);
  }

  /**
   * Queue message for later sending
   */
  private queueMessage(message: WebSocketMessage): void {
    this.messageQueue.push(message);

    // Limit queue size
    if (this.messageQueue.length > 100) {
      this.messageQueue.shift();
    }
  }

  /**
   * Send all queued messages
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.socket?.readyState === WebSocket.OPEN) {
      const message = this.messageQueue.shift()!;
      try {
        this.socket.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send queued message:', error);
        break;
      }
    }
  }

  /**
   * Setup app state change handlers
   */
  private setupAppStateHandlers(): void {
    // This would integrate with React Native AppState
    // For now, we'll just log the setup
    console.log('WebSocket app state handlers configured');
  }

  /**
   * Check if risk level increase is significant
   */
  private isSignificantRiskIncrease(newRiskLevel: RiskLevel): boolean {
    const state = store.getState();
    const currentRiskLevel = state.map.currentRiskLevel;

    const riskValues = {
      [RiskLevel.SAFE]: 0,
      [RiskLevel.LOW]: 1,
      [RiskLevel.MODERATE]: 2,
      [RiskLevel.HIGH]: 3,
      [RiskLevel.CRITICAL]: 4,
    };

    return riskValues[newRiskLevel] > riskValues[currentRiskLevel];
  }

  /**
   * Generate unique message ID
   */
  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get device identifier
   */
  private getDeviceId(): string {
    // This would use a real device ID in production
    return `device_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Send location update to server
   */
  public updateLocation(latitude: number, longitude: number): void {
    this.send({
      type: 'location_update',
      data: {
        coordinates: {latitude, longitude},
        timestamp: new Date().toISOString(),
      },
    });
  }

  /**
   * Subscribe to location-specific alerts
   */
  public subscribeToLocationAlerts(latitude: number, longitude: number, radius: number = 1000): void {
    this.send({
      type: 'subscribe_location',
      data: {
        coordinates: {latitude, longitude},
        radius: radius,
        alertTypes: ['safety_alerts', 'live_incidents', 'risk_updates'],
      },
    });
  }

  /**
   * Request immediate risk assessment
   */
  public requestRiskAssessment(latitude: number, longitude: number): void {
    this.send({
      type: 'request_risk_assessment',
      data: {
        coordinates: {latitude, longitude},
        requestId: this.generateMessageId(),
      },
    });
  }
}

export default WebSocketService;