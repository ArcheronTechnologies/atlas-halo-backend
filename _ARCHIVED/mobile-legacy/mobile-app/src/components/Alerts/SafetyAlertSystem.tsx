/**
 * Safety alert system component for displaying real-time safety notifications
 */

import React, {useEffect, useRef} from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
  FlatList,
  Alert,
  Vibration,
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {useDispatch, useSelector} from 'react-redux';

import {
  selectSafetyAlerts,
  selectCurrentRiskLevel,
  dismissSafetyAlert,
  clearAllSafetyAlerts,
} from '@store/slices/mapSlice';
import {SafetyAlert, RiskLevel} from '@types/safety';
import {AppDispatch} from '@store/index';

const {width} = Dimensions.get('window');

interface SafetyAlertSystemProps {
  style?: any;
}

const SafetyAlertSystem: React.FC<SafetyAlertSystemProps> = ({style}) => {
  const dispatch = useDispatch<AppDispatch>();
  const alerts = useSelector(selectSafetyAlerts);
  const currentRiskLevel = useSelector(selectCurrentRiskLevel);

  // Animation references
  const slideAnimation = useRef(new Animated.Value(-100)).current;
  const pulseAnimation = useRef(new Animated.Value(1)).current;
  const riskIndicatorAnimation = useRef(new Animated.Value(0)).current;

  // Show/hide alert animations
  useEffect(() => {
    if (alerts.length > 0) {
      Animated.spring(slideAnimation, {
        toValue: 0,
        useNativeDriver: true,
        tension: 100,
        friction: 8,
      }).start();
    } else {
      Animated.timing(slideAnimation, {
        toValue: -100,
        duration: 300,
        useNativeDriver: true,
      }).start();
    }
  }, [alerts.length]);

  // Pulse animation for critical alerts
  useEffect(() => {
    const hasCriticalAlert = alerts.some(
      alert => alert.severity === RiskLevel.CRITICAL
    );

    if (hasCriticalAlert) {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnimation, {
            toValue: 1.1,
            duration: 600,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnimation, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();

      // Vibrate for critical alerts
      Vibration.vibrate([0, 500, 200, 500]);

      return () => pulse.stop();
    }
  }, [alerts]);

  // Risk level indicator animation
  useEffect(() => {
    Animated.timing(riskIndicatorAnimation, {
      toValue: getRiskLevelValue(currentRiskLevel),
      duration: 500,
      useNativeDriver: false,
    }).start();
  }, [currentRiskLevel]);

  // Handle alert dismissal
  const handleDismissAlert = (alertId: string) => {
    dispatch(dismissSafetyAlert(alertId));
  };

  // Handle clear all alerts
  const handleClearAllAlerts = () => {
    Alert.alert(
      'Clear All Alerts',
      'Are you sure you want to dismiss all safety alerts?',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: () => dispatch(clearAllSafetyAlerts()),
        },
      ]
    );
  };

  // Get priority alert (most severe)
  const getPriorityAlert = (): SafetyAlert | null => {
    if (alerts.length === 0) return null;

    return alerts.reduce((priority, current) => {
      const priorityValue = getRiskLevelValue(priority.severity);
      const currentValue = getRiskLevelValue(current.severity);
      return currentValue > priorityValue ? current : priority;
    });
  };

  // Render individual alert
  const renderAlert = ({item}: {item: SafetyAlert}) => (
    <Animated.View
      style={[
        styles.alertCard,
        {borderLeftColor: getRiskColor(item.severity)},
        item.severity === RiskLevel.CRITICAL && {
          transform: [{scale: pulseAnimation}],
        },
      ]}
    >
      <View style={styles.alertHeader}>
        <Icon
          name={getAlertIcon(item.type, item.severity)}
          size={24}
          color={getRiskColor(item.severity)}
        />
        <View style={styles.alertHeaderText}>
          <Text style={[styles.alertTitle, {color: getRiskColor(item.severity)}]}>
            {item.title}
          </Text>
          <Text style={styles.alertTimestamp}>
            {formatTimestamp(item.timestamp)}
          </Text>
        </View>
        <TouchableOpacity
          style={styles.dismissButton}
          onPress={() => handleDismissAlert(item.id)}
        >
          <Icon name="close" size={20} color="#666" />
        </TouchableOpacity>
      </View>

      <Text style={styles.alertMessage}>{item.message}</Text>

      {item.location && (
        <View style={styles.alertLocation}>
          <Icon name="place" size={16} color="#666" />
          <Text style={styles.locationText}>
            {item.location.latitude.toFixed(4)}, {item.location.longitude.toFixed(4)}
          </Text>
        </View>
      )}

      {item.severity === RiskLevel.CRITICAL && (
        <View style={styles.emergencyActions}>
          <TouchableOpacity
            style={styles.emergencyButton}
            onPress={() => handleEmergencyAction(item)}
          >
            <Icon name="emergency" size={18} color="white" />
            <Text style={styles.emergencyButtonText}>Emergency Options</Text>
          </TouchableOpacity>
        </View>
      )}
    </Animated.View>
  );

  // Handle emergency actions
  const handleEmergencyAction = (alert: SafetyAlert) => {
    Alert.alert(
      'Emergency Options',
      'Choose an emergency action:',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Call Emergency Services',
          style: 'destructive',
          onPress: () => {
            // Implement emergency call functionality
            console.log('Calling emergency services...');
          },
        },
        {
          text: 'Share Location',
          onPress: () => {
            // Implement location sharing
            console.log('Sharing location...');
          },
        },
        {
          text: 'Report Incident',
          onPress: () => {
            // Open incident reporting
            console.log('Opening incident report...');
          },
        },
      ]
    );
  };

  const priorityAlert = getPriorityAlert();

  return (
    <View style={[styles.container, style]}>
      {/* Current risk level indicator */}
      <Animated.View
        style={[
          styles.riskIndicator,
          {
            backgroundColor: riskIndicatorAnimation.interpolate({
              inputRange: [0, 1, 2, 3, 4],
              outputRange: ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'],
            }),
          },
        ]}
      >
        <Icon
          name={getRiskIcon(currentRiskLevel)}
          size={16}
          color="white"
        />
        <Text style={styles.riskText}>
          {currentRiskLevel.toUpperCase()} RISK
        </Text>
      </Animated.View>

      {/* Alert system */}
      {alerts.length > 0 && (
        <Animated.View
          style={[
            styles.alertSystem,
            {
              transform: [{translateY: slideAnimation}],
            },
          ]}
        >
          {/* Priority alert display */}
          {priorityAlert && (
            <View style={styles.priorityAlert}>
              {renderAlert({item: priorityAlert})}
            </View>
          )}

          {/* All alerts list */}
          {alerts.length > 1 && (
            <View style={styles.alertsList}>
              <View style={styles.alertsHeader}>
                <Text style={styles.alertsCount}>
                  {alerts.length - 1} more alert{alerts.length > 2 ? 's' : ''}
                </Text>
                <TouchableOpacity
                  style={styles.clearAllButton}
                  onPress={handleClearAllAlerts}
                >
                  <Text style={styles.clearAllText}>Clear All</Text>
                </TouchableOpacity>
              </View>

              <FlatList
                data={alerts.slice(1)} // Exclude priority alert
                renderItem={renderAlert}
                keyExtractor={(item) => item.id}
                style={styles.alertsScrollView}
                showsVerticalScrollIndicator={false}
                maxHeight={200}
              />
            </View>
          )}
        </Animated.View>
      )}
    </View>
  );
};

// Helper functions
function getRiskColor(riskLevel: RiskLevel): string {
  switch (riskLevel) {
    case RiskLevel.SAFE:
      return '#4CAF50';
    case RiskLevel.LOW:
      return '#8BC34A';
    case RiskLevel.MODERATE:
      return '#FFC107';
    case RiskLevel.HIGH:
      return '#FF9800';
    case RiskLevel.CRITICAL:
      return '#F44336';
    default:
      return '#757575';
  }
}

function getRiskIcon(riskLevel: RiskLevel): string {
  switch (riskLevel) {
    case RiskLevel.SAFE:
      return 'verified';
    case RiskLevel.LOW:
      return 'info';
    case RiskLevel.MODERATE:
      return 'warning';
    case RiskLevel.HIGH:
      return 'error';
    case RiskLevel.CRITICAL:
      return 'dangerous';
    default:
      return 'help';
  }
}

function getAlertIcon(alertType: string, severity: RiskLevel): string {
  switch (alertType) {
    case 'zone_entry':
      return severity === RiskLevel.CRITICAL ? 'dangerous' : 'place';
    case 'zone_exit':
      return 'exit-to-app';
    case 'risk_prediction':
      return 'trending-up';
    case 'emergency':
      return 'emergency';
    default:
      return 'notification-important';
  }
}

function getRiskLevelValue(riskLevel: RiskLevel): number {
  switch (riskLevel) {
    case RiskLevel.SAFE: return 0;
    case RiskLevel.LOW: return 1;
    case RiskLevel.MODERATE: return 2;
    case RiskLevel.HIGH: return 3;
    case RiskLevel.CRITICAL: return 4;
    default: return 0;
  }
}

function formatTimestamp(timestamp: Date): string {
  const now = new Date();
  const diff = now.getTime() - timestamp.getTime();
  const minutes = Math.floor(diff / 60000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  return timestamp.toLocaleDateString();
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    zIndex: 1000,
  },
  riskIndicator: {
    position: 'absolute',
    top: 10,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  riskText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
    marginLeft: 6,
  },
  alertSystem: {
    paddingHorizontal: 16,
    paddingTop: 60,
  },
  priorityAlert: {
    marginBottom: 8,
  },
  alertCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
    borderLeftWidth: 4,
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 3},
    shadowOpacity: 0.25,
    shadowRadius: 6,
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  alertHeaderText: {
    flex: 1,
    marginLeft: 12,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  alertTimestamp: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  dismissButton: {
    padding: 4,
  },
  alertMessage: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
    marginBottom: 8,
  },
  alertLocation: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  locationText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 4,
  },
  emergencyActions: {
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.1)',
    paddingTop: 12,
    marginTop: 8,
  },
  emergencyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F44336',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  emergencyButtonText: {
    color: 'white',
    fontWeight: '600',
    marginLeft: 8,
  },
  alertsList: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    padding: 12,
  },
  alertsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  alertsCount: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  clearAllButton: {
    paddingVertical: 4,
    paddingHorizontal: 8,
  },
  clearAllText: {
    fontSize: 12,
    color: '#F44336',
    fontWeight: '500',
  },
  alertsScrollView: {
    maxHeight: 200,
  },
});

export default SafetyAlertSystem;