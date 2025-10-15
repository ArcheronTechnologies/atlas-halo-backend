/**
 * Watchlist screen for managing monitored locations
 */

import React, {useState, useEffect, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  Alert,
  Platform,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {useDispatch, useSelector} from 'react-redux';

import {WatchedLocation, RiskLevel} from '@types/safety';
import {AppDispatch} from '@store/index';
import WatchedLocationCard from '@components/Watchlist/WatchedLocationCard';
import AddWatchedLocationModal from '@components/Watchlist/AddWatchedLocationModal';

interface WatchlistScreenProps {
  navigation: any;
}

const WatchlistScreen: React.FC<WatchlistScreenProps> = ({navigation}) => {
  const dispatch = useDispatch<AppDispatch>();

  // Local state
  const [watchedLocations, setWatchedLocations] = useState<WatchedLocation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingLocation, setEditingLocation] = useState<WatchedLocation | null>(null);

  // Load watched locations on mount
  useEffect(() => {
    loadWatchedLocations();
  }, []);

  // Load watched locations from API
  const loadWatchedLocations = useCallback(async () => {
    try {
      setIsLoading(true);

      // This would call the actual API
      const response = await fetch('/api/mobile/watched-locations', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();

      if (result.success) {
        setWatchedLocations(result.data.watchedLocations || []);
      } else {
        throw new Error(result.error || 'Failed to load watched locations');
      }

    } catch (error) {
      console.error('Failed to load watched locations:', error);
      // Show mock data for demo
      setWatchedLocations(getMockWatchedLocations());
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Refresh watched locations
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await loadWatchedLocations();
    setIsRefreshing(false);
  }, [loadWatchedLocations]);

  // Add new watched location
  const handleAddLocation = async (
    locationData: Omit<WatchedLocation, 'id' | 'lastChecked' | 'currentRiskLevel'>
  ) => {
    try {
      const newLocation: WatchedLocation = {
        id: generateLocationId(),
        ...locationData,
        lastChecked: new Date(),
        currentRiskLevel: RiskLevel.SAFE, // Would be fetched from API
      };

      // This would call the actual API
      const response = await fetch('/api/mobile/watched-locations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newLocation),
      });

      const result = await response.json();

      if (result.success) {
        setWatchedLocations(prev => [...prev, newLocation]);
        setShowAddModal(false);
      } else {
        throw new Error(result.error || 'Failed to add watched location');
      }

    } catch (error) {
      console.error('Failed to add watched location:', error);
      // For demo, add locally
      const newLocation: WatchedLocation = {
        id: generateLocationId(),
        ...locationData,
        lastChecked: new Date(),
        currentRiskLevel: RiskLevel.SAFE,
      };
      setWatchedLocations(prev => [...prev, newLocation]);
      setShowAddModal(false);
    }
  };

  // Edit watched location
  const handleEditLocation = (location: WatchedLocation) => {
    setEditingLocation(location);
    setShowAddModal(true);
  };

  // Update watched location
  const handleUpdateLocation = async (
    locationData: Omit<WatchedLocation, 'id' | 'lastChecked' | 'currentRiskLevel'>
  ) => {
    if (!editingLocation) return;

    try {
      const updatedLocation: WatchedLocation = {
        ...editingLocation,
        ...locationData,
        lastChecked: new Date(),
      };

      // This would call the actual API
      const response = await fetch(`/api/mobile/watched-locations/${editingLocation.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedLocation),
      });

      const result = await response.json();

      if (result.success) {
        setWatchedLocations(prev =>
          prev.map(loc => loc.id === editingLocation.id ? updatedLocation : loc)
        );
        setShowAddModal(false);
        setEditingLocation(null);
      } else {
        throw new Error(result.error || 'Failed to update watched location');
      }

    } catch (error) {
      console.error('Failed to update watched location:', error);
      // For demo, update locally
      const updatedLocation: WatchedLocation = {
        ...editingLocation,
        ...locationData,
        lastChecked: new Date(),
      };
      setWatchedLocations(prev =>
        prev.map(loc => loc.id === editingLocation.id ? updatedLocation : loc)
      );
      setShowAddModal(false);
      setEditingLocation(null);
    }
  };

  // Remove watched location
  const handleRemoveLocation = async (locationId: string) => {
    try {
      // This would call the actual API
      const response = await fetch(`/api/mobile/watched-locations/${locationId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();

      if (result.success) {
        setWatchedLocations(prev => prev.filter(loc => loc.id !== locationId));
      } else {
        throw new Error(result.error || 'Failed to remove watched location');
      }

    } catch (error) {
      console.error('Failed to remove watched location:', error);
      // For demo, remove locally
      setWatchedLocations(prev => prev.filter(loc => loc.id !== locationId));
    }
  };

  // View location details
  const handleViewDetails = (location: WatchedLocation) => {
    // Navigate to location details screen
    navigation.navigate('LocationDetails', {location});
  };

  // Close modal
  const handleCloseModal = () => {
    setShowAddModal(false);
    setEditingLocation(null);
  };

  // Render watched location item
  const renderWatchedLocation = ({item}: {item: WatchedLocation}) => (
    <WatchedLocationCard
      watchedLocation={item}
      onEdit={handleEditLocation}
      onRemove={handleRemoveLocation}
      onViewDetails={handleViewDetails}
    />
  );

  // Empty state
  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <Icon name="watch-later" size={64} color="#ccc" />
      <Text style={styles.emptyStateTitle}>No Watched Locations</Text>
      <Text style={styles.emptyStateText}>
        Add locations you want to monitor for safety changes
      </Text>
      <TouchableOpacity
        style={styles.emptyStateButton}
        onPress={() => setShowAddModal(true)}
      >
        <Icon name="add" size={20} color="white" />
        <Text style={styles.emptyStateButtonText}>Add Your First Location</Text>
      </TouchableOpacity>
    </View>
  );

  // Get summary stats
  const getSummaryStats = () => {
    const total = watchedLocations.length;
    const alertsEnabled = watchedLocations.filter(loc => loc.alertsEnabled).length;
    const highRisk = watchedLocations.filter(
      loc => loc.currentRiskLevel === RiskLevel.HIGH || loc.currentRiskLevel === RiskLevel.CRITICAL
    ).length;

    return {total, alertsEnabled, highRisk};
  };

  const stats = getSummaryStats();

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Watched Locations</Text>
        <TouchableOpacity
          style={styles.addButton}
          onPress={() => setShowAddModal(true)}
        >
          <Icon name="add" size={24} color="white" />
        </TouchableOpacity>
      </View>

      {/* Summary stats */}
      {watchedLocations.length > 0 && (
        <View style={styles.summaryContainer}>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{stats.total}</Text>
            <Text style={styles.summaryLabel}>Locations</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryValue}>{stats.alertsEnabled}</Text>
            <Text style={styles.summaryLabel}>With Alerts</Text>
          </View>
          <View style={styles.summaryCard}>
            <Text style={[styles.summaryValue, stats.highRisk > 0 && {color: '#F44336'}]}>
              {stats.highRisk}
            </Text>
            <Text style={styles.summaryLabel}>High Risk</Text>
          </View>
        </View>
      )}

      {/* Watched locations list */}
      <FlatList
        data={watchedLocations}
        renderItem={renderWatchedLocation}
        keyExtractor={(item) => item.id}
        style={styles.list}
        contentContainerStyle={[
          styles.listContent,
          watchedLocations.length === 0 && styles.listContentEmpty,
        ]}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={handleRefresh}
            colors={['#2196F3']}
            tintColor="#2196F3"
          />
        }
        ListEmptyComponent={renderEmptyState}
      />

      {/* Add/Edit location modal */}
      <AddWatchedLocationModal
        visible={showAddModal}
        onClose={handleCloseModal}
        onAdd={editingLocation ? handleUpdateLocation : handleAddLocation}
        editLocation={editingLocation}
      />
    </View>
  );
};

// Helper functions
function generateLocationId(): string {
  return `loc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function getMockWatchedLocations(): WatchedLocation[] {
  return [
    {
      id: 'loc_1',
      alias: 'Home',
      location: {
        name: 'Södermalm',
        coordinates: {latitude: 59.3170, longitude: 18.0740},
        address: 'Götgatan 12, 116 46 Stockholm',
        city: 'Stockholm',
        region: 'Stockholm County',
        country: 'Sweden',
      },
      alertsEnabled: true,
      lastChecked: new Date(Date.now() - 300000), // 5 minutes ago
      currentRiskLevel: RiskLevel.SAFE,
      alertThreshold: RiskLevel.MODERATE,
    },
    {
      id: 'loc_2',
      alias: 'Work',
      location: {
        name: 'Östermalm',
        coordinates: {latitude: 59.3364, longitude: 18.0758},
        address: 'Storgatan 25, 114 55 Stockholm',
        city: 'Stockholm',
        region: 'Stockholm County',
        country: 'Sweden',
      },
      alertsEnabled: true,
      lastChecked: new Date(Date.now() - 600000), // 10 minutes ago
      currentRiskLevel: RiskLevel.LOW,
      alertThreshold: RiskLevel.HIGH,
    },
  ];
}

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
    paddingTop: Platform.OS === 'ios' ? 60 : 16,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
  },
  addButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#2196F3',
    justifyContent: 'center',
    alignItems: 'center',
  },
  summaryContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 16,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 0, 0, 0.1)',
  },
  summaryCard: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  summaryValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  summaryLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  list: {
    flex: 1,
  },
  listContent: {
    padding: 16,
  },
  listContentEmpty: {
    flex: 1,
    justifyContent: 'center',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  emptyStateTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#666',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyStateText: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 32,
  },
  emptyStateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2196F3',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
  },
  emptyStateButtonText: {
    color: 'white',
    fontWeight: '600',
    marginLeft: 8,
  },
});

export default WatchlistScreen;