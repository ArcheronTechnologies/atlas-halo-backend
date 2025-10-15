/**
 * Custom user location marker with risk level indication
 */

import React from 'react';
import {View, StyleSheet} from 'react-native';
import {Marker} from 'react-native-maps';
import Icon from 'react-native-vector-icons/MaterialIcons';

import {RiskLevel, UserLocation} from '@types/safety';

interface UserLocationMarkerProps {
  location: UserLocation;
  riskLevel: RiskLevel;
  size?: number;
}

const UserLocationMarker: React.FC<UserLocationMarkerProps> = ({
  location,
  riskLevel,
  size = 20,
}) => {
  const markerColor = getRiskColor(riskLevel);
  const isMoving = location.isMoving;

  return (
    <Marker
      coordinate={location.coordinates}
      anchor={{x: 0.5, y: 0.5}}
      flat={true}
      rotation={location.heading || 0}
    >
      <View style={[styles.container, {width: size * 2, height: size * 2}]}>
        {/* Outer ring for accuracy/movement indication */}
        <View
          style={[
            styles.outerRing,
            {
              width: size * 2,
              height: size * 2,
              borderRadius: size,
              borderColor: markerColor,
              borderWidth: isMoving ? 3 : 2,
              opacity: isMoving ? 0.8 : 0.6,
            },
          ]}
        />

        {/* Inner location dot */}
        <View
          style={[
            styles.innerDot,
            {
              width: size,
              height: size,
              borderRadius: size / 2,
              backgroundColor: markerColor,
            },
          ]}
        >
          {/* Direction arrow for moving user */}
          {isMoving && location.heading !== undefined && (
            <Icon
              name="navigation"
              size={size * 0.6}
              color="white"
              style={[
                styles.directionArrow,
                {transform: [{rotate: `${location.heading}deg`}]},
              ]}
            />
          )}

          {/* Stationary user icon */}
          {!isMoving && (
            <Icon
              name="person-pin"
              size={size * 0.7}
              color="white"
            />
          )}
        </View>

        {/* Pulse animation for high-risk areas */}
        {(riskLevel === RiskLevel.HIGH || riskLevel === RiskLevel.CRITICAL) && (
          <View
            style={[
              styles.pulseRing,
              {
                width: size * 3,
                height: size * 3,
                borderRadius: size * 1.5,
                borderColor: markerColor,
              },
            ]}
          />
        )}
      </View>
    </Marker>
  );
};

// Helper function to get color based on risk level
function getRiskColor(riskLevel: RiskLevel): string {
  switch (riskLevel) {
    case RiskLevel.SAFE:
      return '#4CAF50'; // Green
    case RiskLevel.LOW:
      return '#8BC34A'; // Light green
    case RiskLevel.MODERATE:
      return '#FFC107'; // Yellow
    case RiskLevel.HIGH:
      return '#FF9800'; // Orange
    case RiskLevel.CRITICAL:
      return '#F44336'; // Red
    default:
      return '#2196F3'; // Blue (default)
  }
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  outerRing: {
    position: 'absolute',
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  innerDot: {
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  directionArrow: {
    textAlign: 'center',
  },
  pulseRing: {
    position: 'absolute',
    borderWidth: 2,
    backgroundColor: 'transparent',
    opacity: 0.4,
    // Animation would be added here in a real implementation
  },
});

export default UserLocationMarker;