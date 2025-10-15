/**
 * Modal for adding new watched locations to monitor
 */

import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TextInput,
  TouchableOpacity,
  Alert,
  ScrollView,
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {useDispatch} from 'react-redux';

import {LocationInfo, RiskLevel, WatchedLocation} from '@types/safety';
import LocationSearchBar from '@components/Search/LocationSearchBar';
import {LocationSearchResult} from '@types/api';
import {AppDispatch} from '@store/index';

const {width, height} = Dimensions.get('window');

interface AddWatchedLocationModalProps {
  visible: boolean;
  onClose: () => void;
  onAdd: (watchedLocation: Omit<WatchedLocation, 'id' | 'lastChecked' | 'currentRiskLevel'>) => void;
  editLocation?: WatchedLocation | null;
}

const AddWatchedLocationModal: React.FC<AddWatchedLocationModalProps> = ({
  visible,
  onClose,
  onAdd,
  editLocation,
}) => {
  const dispatch = useDispatch<AppDispatch>();

  // Form state
  const [alias, setAlias] = useState('');
  const [selectedLocation, setSelectedLocation] = useState<LocationInfo | null>(null);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [alertThreshold, setAlertThreshold] = useState<RiskLevel>(RiskLevel.MODERATE);
  const [isLoading, setIsLoading] = useState(false);

  // Preset aliases
  const presetAliases = [
    {label: 'Home', icon: 'home'},
    {label: 'Work', icon: 'work'},
    {label: 'School', icon: 'school'},
    {label: 'Gym', icon: 'fitness-center'},
    {label: 'Shopping', icon: 'shopping-cart'},
    {label: 'Restaurant', icon: 'restaurant'},
  ];

  // Risk threshold options
  const riskThresholds = [
    {level: RiskLevel.LOW, label: 'Low Risk', color: '#8BC34A'},
    {level: RiskLevel.MODERATE, label: 'Moderate Risk', color: '#FFC107'},
    {level: RiskLevel.HIGH, label: 'High Risk', color: '#FF9800'},
    {level: RiskLevel.CRITICAL, label: 'Critical Risk', color: '#F44336'},
  ];

  // Initialize form when editing
  useEffect(() => {
    if (editLocation) {
      setAlias(editLocation.alias);
      setSelectedLocation(editLocation.location);
      setAlertsEnabled(editLocation.alertsEnabled);
      setAlertThreshold(editLocation.alertThreshold);
    } else {
      resetForm();
    }
  }, [editLocation, visible]);

  // Reset form
  const resetForm = () => {
    setAlias('');
    setSelectedLocation(null);
    setAlertsEnabled(true);
    setAlertThreshold(RiskLevel.MODERATE);
  };

  // Handle location selection from search
  const handleLocationSelect = (searchResult: LocationSearchResult) => {
    const locationInfo: LocationInfo = {
      name: searchResult.name,
      coordinates: {
        latitude: searchResult.location.coordinates.latitude,
        longitude: searchResult.location.coordinates.longitude,
      },
      address: searchResult.location.address,
      city: searchResult.location.city,
      region: searchResult.location.region,
      country: searchResult.location.country,
    };

    setSelectedLocation(locationInfo);
  };

  // Handle preset alias selection
  const handlePresetSelect = (presetAlias: string) => {
    setAlias(presetAlias);
  };

  // Handle form submission
  const handleSubmit = async () => {
    // Validation
    if (!alias.trim()) {
      Alert.alert('Error', 'Please enter a name for this location');
      return;
    }

    if (!selectedLocation) {
      Alert.alert('Error', 'Please select a location to monitor');
      return;
    }

    setIsLoading(true);

    try {
      const watchedLocationData = {
        alias: alias.trim(),
        location: selectedLocation,
        alertsEnabled,
        alertThreshold,
      };

      await onAdd(watchedLocationData);

      Alert.alert(
        'Success',
        `${editLocation ? 'Updated' : 'Added'} "${alias}" to your watched locations`,
        [{text: 'OK', onPress: handleClose}]
      );

    } catch (error) {
      console.error('Failed to add watched location:', error);
      Alert.alert(
        'Error',
        `Failed to ${editLocation ? 'update' : 'add'} watched location. Please try again.`
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle modal close
  const handleClose = () => {
    resetForm();
    onClose();
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={handleClose}
    >
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            style={styles.headerButton}
            onPress={handleClose}
          >
            <Icon name="close" size={24} color="#333" />
          </TouchableOpacity>

          <Text style={styles.headerTitle}>
            {editLocation ? 'Edit Watched Location' : 'Add Watched Location'}
          </Text>

          <TouchableOpacity
            style={[
              styles.headerButton,
              (!alias.trim() || !selectedLocation || isLoading) && styles.headerButtonDisabled,
            ]}
            onPress={handleSubmit}
            disabled={!alias.trim() || !selectedLocation || isLoading}
          >
            <Text style={[
              styles.headerButtonText,
              (!alias.trim() || !selectedLocation || isLoading) && styles.headerButtonTextDisabled,
            ]}>
              {isLoading ? 'Saving...' : (editLocation ? 'Update' : 'Add')}
            </Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          {/* Location search */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Search Location</Text>
            <LocationSearchBar
              placeholder="Search for address, city, or landmark..."
              onLocationSelect={handleLocationSelect}
              autoFocus={false}
            />
          </View>

          {/* Selected location display */}
          {selectedLocation && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Selected Location</Text>
              <View style={styles.selectedLocationCard}>
                <Icon name="place" size={20} color="#2196F3" />
                <View style={styles.selectedLocationInfo}>
                  <Text style={styles.selectedLocationName}>
                    {selectedLocation.name}
                  </Text>
                  <Text style={styles.selectedLocationAddress}>
                    {selectedLocation.address}
                  </Text>
                </View>
              </View>
            </View>
          )}

          {/* Alias input */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Give it a Name</Text>
            <TextInput
              style={styles.aliasInput}
              placeholder="e.g., Home, Work, School..."
              value={alias}
              onChangeText={setAlias}
              maxLength={30}
              autoCapitalize="words"
            />

            {/* Preset aliases */}
            <View style={styles.presetContainer}>
              {presetAliases.map((preset) => (
                <TouchableOpacity
                  key={preset.label}
                  style={[
                    styles.presetButton,
                    alias === preset.label && styles.presetButtonSelected,
                  ]}
                  onPress={() => handlePresetSelect(preset.label)}
                >
                  <Icon
                    name={preset.icon}
                    size={16}
                    color={alias === preset.label ? 'white' : '#666'}
                  />
                  <Text
                    style={[
                      styles.presetButtonText,
                      alias === preset.label && styles.presetButtonTextSelected,
                    ]}
                  >
                    {preset.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Alert settings */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Alert Settings</Text>

            {/* Enable alerts toggle */}
            <TouchableOpacity
              style={styles.toggleRow}
              onPress={() => setAlertsEnabled(!alertsEnabled)}
            >
              <View style={styles.toggleInfo}>
                <Text style={styles.toggleTitle}>Enable Alerts</Text>
                <Text style={styles.toggleDescription}>
                  Get notified when risk level changes
                </Text>
              </View>
              <View style={[
                styles.toggle,
                alertsEnabled && styles.toggleActive,
              ]}>
                <View style={[
                  styles.toggleThumb,
                  alertsEnabled && styles.toggleThumbActive,
                ]} />
              </View>
            </TouchableOpacity>

            {/* Alert threshold */}
            {alertsEnabled && (
              <View style={styles.thresholdSection}>
                <Text style={styles.thresholdTitle}>Alert when risk exceeds:</Text>
                <View style={styles.thresholdOptions}>
                  {riskThresholds.map((threshold) => (
                    <TouchableOpacity
                      key={threshold.level}
                      style={[
                        styles.thresholdOption,
                        alertThreshold === threshold.level && {
                          backgroundColor: threshold.color,
                        },
                      ]}
                      onPress={() => setAlertThreshold(threshold.level)}
                    >
                      <Text
                        style={[
                          styles.thresholdOptionText,
                          alertThreshold === threshold.level && styles.thresholdOptionTextSelected,
                        ]}
                      >
                        {threshold.label}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}
          </View>

          {/* Information note */}
          <View style={styles.infoNote}>
            <Icon name="info" size={20} color="#2196F3" />
            <Text style={styles.infoText}>
              We'll monitor this location's safety status and send you alerts
              when risk levels change. You can modify these settings anytime.
            </Text>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 16,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 0, 0, 0.1)',
  },
  headerButton: {
    padding: 8,
  },
  headerButtonDisabled: {
    opacity: 0.5,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  headerButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2196F3',
  },
  headerButtonTextDisabled: {
    color: '#999',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  selectedLocationCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#2196F3',
  },
  selectedLocationInfo: {
    flex: 1,
    marginLeft: 12,
  },
  selectedLocationName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  selectedLocationAddress: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  aliasInput: {
    backgroundColor: 'white',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.1)',
    marginBottom: 16,
  },
  presetContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  presetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.1)',
  },
  presetButtonSelected: {
    backgroundColor: '#2196F3',
    borderColor: '#2196F3',
  },
  presetButtonText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 6,
  },
  presetButtonTextSelected: {
    color: 'white',
  },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  toggleInfo: {
    flex: 1,
  },
  toggleTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  toggleDescription: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  toggle: {
    width: 48,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#ccc',
    justifyContent: 'center',
    paddingHorizontal: 2,
  },
  toggleActive: {
    backgroundColor: '#4CAF50',
  },
  toggleThumb: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'white',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.25,
    shadowRadius: 2,
    elevation: 2,
  },
  toggleThumbActive: {
    transform: [{translateX: 20}],
  },
  thresholdSection: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
  },
  thresholdTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  thresholdOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  thresholdOption: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
  },
  thresholdOptionText: {
    fontSize: 12,
    color: '#666',
  },
  thresholdOptionTextSelected: {
    color: 'white',
    fontWeight: '600',
  },
  infoNote: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(33, 150, 243, 0.1)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
  },
  infoText: {
    flex: 1,
    fontSize: 12,
    color: '#1976D2',
    marginLeft: 8,
    lineHeight: 18,
  },
});

export default AddWatchedLocationModal;