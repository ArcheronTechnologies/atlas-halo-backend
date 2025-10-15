/**
 * Card component for displaying watched location information and status
 */

import React, {useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Animated,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {useDispatch} from 'react-redux';

import {WatchedLocation, RiskLevel} from '@types/safety';
import {AppDispatch} from '@store/index';

interface WatchedLocationCardProps {
  watchedLocation: WatchedLocation;
  onEdit: (location: WatchedLocation) => void;
  onRemove: (locationId: string) => void;
  onViewDetails: (location: WatchedLocation) => void;
  style?: any;
}

const WatchedLocationCard: React.FC<WatchedLocationCardProps> = ({
  watchedLocation,
  onEdit,
  onRemove,
  onViewDetails,
  style,
}) => {
  const dispatch = useDispatch<AppDispatch>();
  const [isExpanded, setIsExpanded] = useState(false);

  // Get risk level styling
  const getRiskColor = (riskLevel: RiskLevel): string => {
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
  };

  const getRiskIcon = (riskLevel: RiskLevel): string => {
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
  };

  // Handle remove with confirmation
  const handleRemove = () => {
    Alert.alert(
      'Remove Watched Location',
      `Are you sure you want to stop monitoring "${watchedLocation.alias}"?`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Remove',
          style: 'destructive',
          onPress: () => onRemove(watchedLocation.id),
        },
      ]
    );
  };

  // Toggle alerts
  const handleToggleAlerts = async () => {
    try {
      // This would update the watched location in the backend
      const updatedLocation = {
        ...watchedLocation,
        alertsEnabled: !watchedLocation.alertsEnabled,
      };

      // Update local state (would dispatch to Redux)
      console.log('Toggling alerts for:', watchedLocation.alias);
    } catch (error) {
      console.error('Failed to toggle alerts:', error);
      Alert.alert('Error', 'Failed to update alert settings');
    }
  };

  // Get time since last check
  const getTimeSinceLastCheck = (): string => {
    const now = new Date();
    const lastCheck = watchedLocation.lastChecked;
    const diffMinutes = Math.floor((now.getTime() - lastCheck.getTime()) / 60000);

    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;

    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;

    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <View style={[styles.container, style]}>
      <TouchableOpacity
        style={[
          styles.card,
          {borderLeftColor: getRiskColor(watchedLocation.currentRiskLevel)},
        ]}
        onPress={() => setIsExpanded(!isExpanded)}
        activeOpacity={0.9}
      >
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.locationInfo}>
            <Text style={styles.alias}>{watchedLocation.alias}</Text>
            <Text style={styles.locationName}>
              {watchedLocation.location.name}
            </Text>
          </View>

          <View style={styles.riskIndicator}>
            <Icon
              name={getRiskIcon(watchedLocation.currentRiskLevel)}
              size={20}
              color={getRiskColor(watchedLocation.currentRiskLevel)}
            />
            <Text
              style={[
                styles.riskText,
                {color: getRiskColor(watchedLocation.currentRiskLevel)},
              ]}
            >
              {watchedLocation.currentRiskLevel.toUpperCase()}
            </Text>
          </View>
        </View>

        {/* Status row */}
        <View style={styles.statusRow}>
          <View style={styles.alertStatus}>
            <Icon
              name={watchedLocation.alertsEnabled ? 'notifications' : 'notifications-off'}
              size={16}
              color={watchedLocation.alertsEnabled ? '#4CAF50' : '#757575'}
            />
            <Text style={styles.alertStatusText}>
              {watchedLocation.alertsEnabled ? 'Alerts On' : 'Alerts Off'}
            </Text>
          </View>

          <Text style={styles.lastChecked}>
            Updated {getTimeSinceLastCheck()}
          </Text>

          <Icon
            name={isExpanded ? 'expand-less' : 'expand-more'}
            size={20}
            color="#666"
          />
        </View>

        {/* Expanded content */}
        {isExpanded && (
          <View style={styles.expandedContent}>
            {/* Location details */}
            <View style={styles.detailSection}>
              <Text style={styles.sectionTitle}>Location Details</Text>
              <Text style={styles.address}>
                {watchedLocation.location.address}
              </Text>
              <Text style={styles.coordinates}>
                {watchedLocation.location.coordinates.latitude.toFixed(4)},
                {watchedLocation.location.coordinates.longitude.toFixed(4)}
              </Text>
            </View>

            {/* Alert settings */}
            <View style={styles.detailSection}>
              <Text style={styles.sectionTitle}>Alert Settings</Text>
              <View style={styles.alertThreshold}>
                <Text style={styles.thresholdLabel}>Alert when risk exceeds:</Text>
                <Text
                  style={[
                    styles.thresholdValue,
                    {color: getRiskColor(watchedLocation.alertThreshold)},
                  ]}
                >
                  {watchedLocation.alertThreshold.toUpperCase()}
                </Text>
              </View>
            </View>

            {/* Action buttons */}
            <View style={styles.actionButtons}>
              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => onViewDetails(watchedLocation)}
              >
                <Icon name="visibility" size={18} color="#2196F3" />
                <Text style={styles.actionButtonText}>View Details</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.actionButton}
                onPress={handleToggleAlerts}
              >
                <Icon
                  name={watchedLocation.alertsEnabled ? 'notifications-off' : 'notifications'}
                  size={18}
                  color="#FF9800"
                />
                <Text style={styles.actionButtonText}>
                  {watchedLocation.alertsEnabled ? 'Disable Alerts' : 'Enable Alerts'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.actionButton}
                onPress={() => onEdit(watchedLocation)}
              >
                <Icon name="edit" size={18} color="#4CAF50" />
                <Text style={styles.actionButtonText}>Edit</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.actionButton}
                onPress={handleRemove}
              >
                <Icon name="delete" size={18} color="#F44336" />
                <Text style={styles.actionButtonText}>Remove</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 12,
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  locationInfo: {
    flex: 1,
  },
  alias: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  locationName: {
    fontSize: 14,
    color: '#666',
  },
  riskIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 16,
  },
  riskText: {
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  alertStatus: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  alertStatusText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 4,
  },
  lastChecked: {
    fontSize: 12,
    color: '#999',
  },
  expandedContent: {
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.1)',
    paddingTop: 16,
    marginTop: 16,
  },
  detailSection: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  address: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  coordinates: {
    fontSize: 12,
    color: '#999',
    fontFamily: 'monospace',
  },
  alertThreshold: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  thresholdLabel: {
    fontSize: 14,
    color: '#666',
  },
  thresholdValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  actionButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    marginRight: 8,
    marginBottom: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    borderRadius: 20,
  },
  actionButtonText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 4,
  },
});

export default WatchedLocationCard;