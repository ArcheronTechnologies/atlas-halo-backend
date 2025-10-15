/**
 * Safety legend component showing risk level colors and meanings
 */

import React, {useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {RiskLevel} from '@types/safety';

interface SafetyLegendProps {
  style?: any;
  compact?: boolean;
}

const SafetyLegend: React.FC<SafetyLegendProps> = ({
  style,
  compact = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(!compact);
  const [animation] = useState(new Animated.Value(compact ? 0 : 1));

  const toggleExpanded = () => {
    const toValue = isExpanded ? 0 : 1;
    setIsExpanded(!isExpanded);

    Animated.timing(animation, {
      toValue,
      duration: 300,
      useNativeDriver: false,
    }).start();
  };

  const legendItems = [
    {
      level: RiskLevel.SAFE,
      color: '#4CAF50',
      label: 'Safe',
      description: 'Low crime area',
    },
    {
      level: RiskLevel.LOW,
      color: '#8BC34A',
      label: 'Low Risk',
      description: 'Generally safe',
    },
    {
      level: RiskLevel.MODERATE,
      color: '#FFC107',
      label: 'Moderate',
      description: 'Exercise caution',
    },
    {
      level: RiskLevel.HIGH,
      color: '#FF9800',
      label: 'High Risk',
      description: 'Stay alert',
    },
    {
      level: RiskLevel.CRITICAL,
      color: '#F44336',
      label: 'Critical',
      description: 'Avoid if possible',
    },
  ];

  const headerHeight = 40;
  const itemHeight = 32;
  const maxHeight = headerHeight + (legendItems.length * itemHeight) + 16;

  return (
    <View style={[styles.container, style]}>
      {/* Header with toggle button */}
      <TouchableOpacity
        style={styles.header}
        onPress={compact ? toggleExpanded : undefined}
        disabled={!compact}
      >
        <Icon
          name="security"
          size={20}
          color="#666"
          style={styles.headerIcon}
        />
        <Text style={styles.headerText}>Safety Levels</Text>
        {compact && (
          <Icon
            name={isExpanded ? "expand-less" : "expand-more"}
            size={20}
            color="#666"
          />
        )}
      </TouchableOpacity>

      {/* Legend items */}
      <Animated.View
        style={[
          styles.content,
          {
            height: animation.interpolate({
              inputRange: [0, 1],
              outputRange: [0, maxHeight - headerHeight],
            }),
            opacity: animation,
          },
        ]}
      >
        {legendItems.map((item, index) => (
          <View key={item.level} style={styles.legendItem}>
            <View
              style={[
                styles.colorIndicator,
                {backgroundColor: item.color},
              ]}
            />
            <View style={styles.labelContainer}>
              <Text style={styles.labelText}>{item.label}</Text>
              {!compact && (
                <Text style={styles.descriptionText}>
                  {item.description}
                </Text>
              )}
            </View>
          </View>
        ))}

        {/* Additional info */}
        {!compact && (
          <View style={styles.footer}>
            <Text style={styles.footerText}>
              Risk levels update in real-time based on current conditions
            </Text>
          </View>
        )}
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.1)',
    overflow: 'hidden',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.02)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 0, 0, 0.05)',
  },
  headerIcon: {
    marginRight: 8,
  },
  headerText: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  content: {
    overflow: 'hidden',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  colorIndicator: {
    width: 16,
    height: 16,
    borderRadius: 8,
    marginRight: 10,
    borderWidth: 1,
    borderColor: 'rgba(0, 0, 0, 0.1)',
  },
  labelContainer: {
    flex: 1,
  },
  labelText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#333',
  },
  descriptionText: {
    fontSize: 10,
    color: '#666',
    marginTop: 1,
  },
  footer: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.05)',
  },
  footerText: {
    fontSize: 10,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
});

export default SafetyLegend;