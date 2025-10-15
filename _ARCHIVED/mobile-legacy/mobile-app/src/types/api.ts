/**
 * API-related type definitions for Atlas AI Mobile App
 */

import {
  RiskLevel,
  AlertType,
  IncidentType,
  LocationInfo,
  SafetyAssessment,
  IncidentReport,
  SafetyAlert,
  ThreatDetectionResult,
  CommunityReport,
  Coordinates,
} from './safety';

// API Response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: Date;
}

// API Error types
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

// Location-based API requests
export interface LocationSafetyRequest {
  latitude: number;
  longitude: number;
  radius?: number; // meters
}

export interface LocationSearchRequest {
  query: string;
  type?: 'city' | 'landmark' | 'address' | 'all';
  country?: string;
  limit?: number;
}

export interface LocationSearchResult {
  id: string;
  name: string;
  displayName: string;
  location: LocationInfo;
  type: 'city' | 'landmark' | 'address' | 'poi';
  relevanceScore: number;
}

// Safety assessment API
export interface SafetyAssessmentRequest {
  location: LocationInfo;
  includeHistorical?: boolean;
  includeTimeBasedRisk?: boolean;
  includeCommunityReports?: boolean;
}

export interface SafetyAssessmentResponse {
  assessment: SafetyAssessment;
  nearbyIncidents: IncidentReport[];
  communityReports: CommunityReport[];
  policePresence: {
    stationDistance: number;
    patrolFrequency: 'low' | 'moderate' | 'high';
    responseTime: number; // minutes
  };
}

// Incident reporting API
export interface CreateIncidentRequest {
  type: IncidentType;
  location: LocationInfo;
  description: string;
  isAnonymous: boolean;
  severity: RiskLevel;
  mediaFiles?: MediaUpload[];
  audioFile?: MediaUpload;
}

export interface MediaUpload {
  fileName: string;
  mimeType: string;
  base64Data: string;
  size: number;
  duration?: number; // for audio/video
}

export interface IncidentReportResponse {
  reportId: string;
  status: 'submitted' | 'processing' | 'verified' | 'rejected';
  estimatedProcessingTime: number; // minutes
  threatAnalysis?: ThreatDetectionResult;
}

// Alert management API
export interface AlertSubscriptionRequest {
  location: LocationInfo;
  radius: number; // meters
  alertTypes: AlertType[];
  riskThreshold: RiskLevel;
  isEnabled: boolean;
}

export interface AlertSubscription {
  id: string;
  location: LocationInfo;
  radius: number;
  alertTypes: AlertType[];
  riskThreshold: RiskLevel;
  isEnabled: boolean;
  createdAt: Date;
  lastTriggered?: Date;
}

// Watchlist API
export interface WatchlistAddRequest {
  location: LocationInfo;
  alias: string;
  alertsEnabled: boolean;
  alertThreshold: RiskLevel;
}

export interface WatchlistUpdateRequest {
  id: string;
  alias?: string;
  alertsEnabled?: boolean;
  alertThreshold?: RiskLevel;
}

export interface WatchlistResponse {
  id: string;
  location: LocationInfo;
  alias: string;
  alertsEnabled: boolean;
  alertThreshold: RiskLevel;
  currentRiskLevel: RiskLevel;
  lastChecked: Date;
  recentAlerts: SafetyAlert[];
}

// Real-time updates
export interface WebSocketMessage {
  type: 'alert' | 'incident_update' | 'risk_change' | 'system_update';
  data: any;
  timestamp: Date;
  messageId: string;
}

export interface LiveAlertMessage extends WebSocketMessage {
  type: 'alert';
  data: {
    alert: SafetyAlert;
    affectedLocations: LocationInfo[];
    userIsAffected: boolean;
  };
}

export interface RiskChangeMessage extends WebSocketMessage {
  type: 'risk_change';
  data: {
    location: LocationInfo;
    oldRiskLevel: RiskLevel;
    newRiskLevel: RiskLevel;
    reason: string;
    affectedRadius: number;
  };
}

export interface IncidentUpdateMessage extends WebSocketMessage {
  type: 'incident_update';
  data: {
    incidentId: string;
    status: 'new' | 'updated' | 'resolved';
    incident: IncidentReport;
    location: LocationInfo;
  };
}

// User preferences and settings
export interface UserPreferences {
  notifications: {
    alertsEnabled: boolean;
    soundEnabled: boolean;
    vibrationEnabled: boolean;
    emergencyAlertsOnly: boolean;
    alertTypes: AlertType[];
  };
  privacy: {
    shareLocation: boolean;
    anonymousReporting: boolean;
    dataRetention: number; // days
  };
  safety: {
    defaultAlertRadius: number; // meters
    riskThreshold: RiskLevel;
    autoReportThreats: boolean;
  };
  display: {
    mapStyle: 'standard' | 'satellite' | 'hybrid';
    showHistoricalData: boolean;
    colorBlindFriendly: boolean;
  };
}

// Analytics and reporting
export interface SafetyStatsRequest {
  location: LocationInfo;
  radius: number;
  timeframe: 'day' | 'week' | 'month' | 'year';
}

export interface SafetyStats {
  location: LocationInfo;
  timeframe: string;
  totalIncidents: number;
  incidentsByType: Record<IncidentType, number>;
  riskTrend: Array<{
    date: Date;
    riskLevel: RiskLevel;
    incidentCount: number;
  }>;
  safetyRanking: {
    percentile: number; // 0-100, higher is safer
    comparedTo: 'city' | 'region' | 'country';
  };
  recommendations: string[];
}

// Emergency services integration
export interface EmergencyContactRequest {
  location: LocationInfo;
  emergencyType: 'police' | 'medical' | 'fire';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  callerId?: string;
}

export interface EmergencyResponse {
  contactId: string;
  status: 'dispatched' | 'en_route' | 'arrived' | 'resolved';
  estimatedArrival: number; // minutes
  dispatchedUnits: Array<{
    type: string;
    callSign: string;
    location: Coordinates;
  }>;
  instructions: string[];
}

// System health and status
export interface SystemStatus {
  services: {
    api: 'operational' | 'degraded' | 'outage';
    maps: 'operational' | 'degraded' | 'outage';
    alerts: 'operational' | 'degraded' | 'outage';
    ai: 'operational' | 'degraded' | 'outage';
  };
  dataFreshness: {
    crimeData: Date;
    communityReports: Date;
    riskAssessments: Date;
  };
  coverage: {
    supportedRegions: string[];
    dataQuality: Record<string, number>; // region -> quality score 0-1
  };
}