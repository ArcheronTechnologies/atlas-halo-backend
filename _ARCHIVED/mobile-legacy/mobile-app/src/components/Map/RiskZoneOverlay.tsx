/**
 * Risk zone overlay component for visualizing crime hotspots on the map
 */

import React from 'react';
import {Polygon, Circle} from 'react-native-maps';
import {RiskLevel, SafetyZone} from '@types/safety';

interface RiskZoneOverlayProps {
  zone: SafetyZone;
  opacity?: number;
  strokeWidth?: number;
}

const RiskZoneOverlay: React.FC<RiskZoneOverlayProps> = ({
  zone,
  opacity = 0.3,
  strokeWidth = 2,
}) => {
  // Get colors based on risk level
  const {fillColor, strokeColor} = getRiskColors(zone.riskLevel);

  // If zone has only one coordinate, render as circle
  if (zone.coordinates.length === 1) {
    return (
      <Circle
        center={zone.coordinates[0]}
        radius={500} // Default 500m radius
        fillColor={`${fillColor}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`}
        strokeColor={strokeColor}
        strokeWidth={strokeWidth}
      />
    );
  }

  // Render as polygon for multiple coordinates
  return (
    <Polygon
      coordinates={zone.coordinates}
      fillColor={`${fillColor}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`}
      strokeColor={strokeColor}
      strokeWidth={strokeWidth}
      lineDashPattern={zone.riskLevel === RiskLevel.CRITICAL ? [10, 5] : undefined}
    />
  );
};

// Helper function to get colors based on risk level
function getRiskColors(riskLevel: RiskLevel): {fillColor: string; strokeColor: string} {
  switch (riskLevel) {
    case RiskLevel.SAFE:
      return {
        fillColor: '#4CAF50', // Green
        strokeColor: '#2E7D32',
      };
    case RiskLevel.LOW:
      return {
        fillColor: '#8BC34A', // Light green
        strokeColor: '#558B2F',
      };
    case RiskLevel.MODERATE:
      return {
        fillColor: '#FFC107', // Yellow
        strokeColor: '#F57C00',
      };
    case RiskLevel.HIGH:
      return {
        fillColor: '#FF9800', // Orange
        strokeColor: '#E65100',
      };
    case RiskLevel.CRITICAL:
      return {
        fillColor: '#F44336', // Red
        strokeColor: '#C62828',
      };
    default:
      return {
        fillColor: '#9E9E9E', // Gray
        strokeColor: '#424242',
      };
  }
}

export default RiskZoneOverlay;