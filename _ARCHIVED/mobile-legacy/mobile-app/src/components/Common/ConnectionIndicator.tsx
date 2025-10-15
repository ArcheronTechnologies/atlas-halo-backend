/**
 * Connection status indicator for WebSocket connectivity
 */

import React, {useEffect, useRef} from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  TouchableOpacity,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

import {useWebSocket} from '@hooks/useWebSocket';

interface ConnectionIndicatorProps {
  style?: any;
  showText?: boolean;
  compact?: boolean;
  onPress?: () => void;
}

const ConnectionIndicator: React.FC<ConnectionIndicatorProps> = ({
  style,
  showText = true,
  compact = false,
  onPress,
}) => {
  const {connectionStatus, isConnected, connect} = useWebSocket();

  // Animation for pulsing effect
  const pulseAnimation = useRef(new Animated.Value(1)).current;
  const fadeAnimation = useRef(new Animated.Value(0)).current;

  // Start pulse animation for connecting/reconnecting states
  useEffect(() => {
    if (connectionStatus === 'connecting') {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnimation, {
            toValue: 1.2,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnimation, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();

      return () => pulse.stop();
    } else {
      pulseAnimation.setValue(1);
    }
  }, [connectionStatus, pulseAnimation]);

  // Fade in/out animation
  useEffect(() => {
    Animated.timing(fadeAnimation, {
      toValue: connectionStatus === 'connected' ? 0 : 1,
      duration: 300,
      useNativeDriver: true,
    }).start();
  }, [connectionStatus, fadeAnimation]);

  // Get status styling
  const getStatusColor = (): string => {
    switch (connectionStatus) {
      case 'connected':
        return '#4CAF50';
      case 'connecting':
        return '#FF9800';
      case 'disconnected':
        return '#F44336';
      case 'closing':
        return '#FF9800';
      default:
        return '#757575';
    }
  };

  const getStatusIcon = (): string => {
    switch (connectionStatus) {
      case 'connected':
        return 'wifi';
      case 'connecting':
        return 'wifi-find';
      case 'disconnected':
        return 'wifi-off';
      case 'closing':
        return 'wifi-find';
      default:
        return 'help';
    }
  };

  const getStatusText = (): string => {
    switch (connectionStatus) {
      case 'connected':
        return 'Live Updates Active';
      case 'connecting':
        return 'Connecting...';
      case 'disconnected':
        return 'Offline - Tap to Reconnect';
      case 'closing':
        return 'Disconnecting...';
      default:
        return 'Unknown Status';
    }
  };

  // Handle press to reconnect
  const handlePress = () => {
    if (onPress) {
      onPress();
    } else if (!isConnected) {
      connect().catch(error => {
        console.error('Failed to reconnect:', error);
      });
    }
  };

  // Don't show indicator when connected (unless forced)
  if (connectionStatus === 'connected' && !compact) {
    return null;
  }

  const Component = onPress || !isConnected ? TouchableOpacity : View;

  return (
    <Animated.View
      style={[
        styles.container,
        compact && styles.compactContainer,
        {opacity: fadeAnimation},
        style,
      ]}
    >
      <Component
        style={[
          styles.indicator,
          compact && styles.compactIndicator,
          {backgroundColor: getStatusColor()},
        ]}
        onPress={handlePress}
        activeOpacity={0.8}
      >
        <Animated.View
          style={[
            styles.iconContainer,
            {transform: [{scale: pulseAnimation}]},
          ]}
        >
          <Icon
            name={getStatusIcon()}
            size={compact ? 16 : 20}
            color="white"
          />
        </Animated.View>

        {showText && !compact && (
          <Text style={styles.statusText}>
            {getStatusText()}
          </Text>
        )}
      </Component>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignSelf: 'flex-start',
  },
  compactContainer: {
    position: 'absolute',
    top: 10,
    right: 10,
    zIndex: 999,
  },
  indicator: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 4,
  },
  compactIndicator: {
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderRadius: 16,
  },
  iconContainer: {
    marginRight: 6,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
    color: 'white',
  },
});

export default ConnectionIndicator;