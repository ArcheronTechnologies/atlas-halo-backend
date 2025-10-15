/**
 * Safety-related type definitions for Atlas AI Mobile App
 */

export enum RiskLevel {
  SAFE = 'safe',
  LOW = 'low',
  MODERATE = 'moderate',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export enum AlertType {
  ZONE_ENTRY = 'zone_entry',
  PREDICTIVE = 'predictive',
  LIVE_INCIDENT = 'live_incident',
  COMMUNITY = 'community',
  EMERGENCY = 'emergency',
}

export enum IncidentType {
  THEFT = 'theft',
  ASSAULT = 'assault',
  ROBBERY = 'robbery',
  VANDALISM = 'vandalism',
  DRUG_ACTIVITY = 'drug_activity',
  SUSPICIOUS_ACTIVITY = 'suspicious_activity',
  VIOLENCE = 'violence',
  WEAPON_SIGHTING = 'weapon_sighting',
  EMERGENCY = 'emergency',
  OTHER = 'other',
}

export interface Coordinates {
  latitude: number;
  longitude: number;
}

export interface LocationInfo {
  name: string;
  coordinates: Coordinates;
  address?: string;
  city?: string;
  region?: string;
  country?: string;
}

export interface SafetyZone {
  id: string;
  coordinates: Coordinates;
  radius: number;
  riskLevel: RiskLevel;
  name?: string;
  description?: string;
  lastUpdated: Date;
  incidentCount: number;
  activeIncidents: number;
}

export interface SafetyAlert {
  id: string;
  type: string;
  title: string;
  message: string;
  severity: RiskLevel;
  location?: Coordinates;
  timestamp: Date;
  isRead?: boolean;
  isActive?: boolean;
  metadata?: Record<string, any>;
}

export interface IncidentReport {
  id: string;
  type: IncidentType;
  location: LocationInfo;
  timestamp: Date;
  description: string;
  mediaUrls: string[];
  audioUrl?: string;
  reporterId?: string;
  isAnonymous: boolean;
  isVerified: boolean;
  severity: RiskLevel;
  status: 'pending' | 'verified' | 'false_alarm' | 'resolved';
}

export interface SafetyAssessment {
  location: LocationInfo;
  riskLevel: RiskLevel;
  safetyScore: number; // 0-10 scale
  lastUpdated: Date;
  factors: SafetyFactor[];
  recommendations: string[];
  historicalData: HistoricalSafetyData;
  timeBasedRisk: TimeBasedRisk[];
}

export interface SafetyFactor {
  type: 'crime_rate' | 'lighting' | 'crowd_density' | 'police_presence' | 'time_of_day' | 'weather';
  impact: 'positive' | 'negative' | 'neutral';
  description: string;
  weight: number; // 0-1
}

export interface HistoricalSafetyData {
  totalIncidents: number;
  incidentsByType: Record<IncidentType, number>;
  trendDirection: 'improving' | 'worsening' | 'stable';
  comparisonToAverage: number; // percentage difference from city average
  timeframe: string; // e.g., "last 30 days"
}

export interface TimeBasedRisk {
  hour: number; // 0-23
  riskLevel: RiskLevel;
  incidentProbability: number; // 0-1
}

export interface WatchedLocation {
  id: string;
  location: LocationInfo;
  alias: string; // User-friendly name like "Home", "Work", "School"
  alertsEnabled: boolean;
  lastChecked: Date;
  currentRiskLevel: RiskLevel;
  alertThreshold: RiskLevel; // Alert when risk exceeds this level
}

export interface UserLocation {
  coordinates: Coordinates;
  accuracy: number;
  timestamp: Date;
  isMoving: boolean;
  speed?: number;
  heading?: number;
}

export interface ThreatDetectionResult {
  id: string;
  mediaUrl: string;
  mediaType: 'image' | 'audio' | 'video';
  analysisResult: {
    threatDetected: boolean;
    threatType?: IncidentType;
    confidence: number; // 0-1
    description: string;
    boundingBoxes?: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      label: string;
      confidence: number;
    }>;
  };
  location: LocationInfo;
  timestamp: Date;
  processingTime: number; // milliseconds
}

export interface CommunityReport {
  id: string;
  type: IncidentType;
  location: LocationInfo;
  description: string;
  timestamp: Date;
  reporterHash: string; // Anonymized identifier
  upvotes: number;
  downvotes: number;
  isVerified: boolean;
  verificationSource?: string;
  expiresAt?: Date;
}