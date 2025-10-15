/**
 * Overlay component showing AI threat analysis results
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';

import {RiskLevel} from '@types/safety';

interface ThreatAnalysisProps {
  analysis: {
    threatDetected: boolean;
    threatType?: string;
    confidence: number;
    description: string;
    riskLevel: RiskLevel;
    boundingBoxes?: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      label: string;
      confidence: number;
    }>;
  };
  style?: any;
}

const ThreatAnalysisOverlay: React.FC<ThreatAnalysisProps> = ({
  analysis,
  style,
}) => {
  const confidencePercent = Math.round(analysis.confidence * 100);
  const riskColor = getRiskColor(analysis.riskLevel);
  const riskIcon = getRiskIcon(analysis.riskLevel);

  return (
    <Animated.View style={[styles.container, style]}>
      <View style={[styles.card, {borderLeftColor: riskColor}]}>
        {/* Header */}
        <View style={styles.header}>
          <Icon
            name={riskIcon}
            size={24}
            color={riskColor}
            style={styles.headerIcon}
          />
          <View style={styles.headerText}>
            <Text style={styles.title}>
              {analysis.threatDetected ? 'Threat Detected' : 'No Threat Detected'}
            </Text>
            <Text style={[styles.riskLevel, {color: riskColor}]}>
              {analysis.riskLevel.toUpperCase()} RISK
            </Text>
          </View>
          <Text style={styles.confidence}>
            {confidencePercent}%
          </Text>
        </View>

        {/* Analysis details */}
        <View style={styles.content}>
          {analysis.threatType && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Threat Type:</Text>
              <Text style={styles.detailValue}>{analysis.threatType}</Text>
            </View>
          )}

          <Text style={styles.description}>
            {analysis.description}
          </Text>

          {/* Detected objects */}
          {analysis.boundingBoxes && analysis.boundingBoxes.length > 0 && (
            <View style={styles.objectsSection}>
              <Text style={styles.objectsTitle}>Detected Objects:</Text>
              {analysis.boundingBoxes.map((box, index) => (
                <View key={index} style={styles.objectItem}>
                  <Text style={styles.objectLabel}>{box.label}</Text>
                  <Text style={styles.objectConfidence}>
                    {Math.round(box.confidence * 100)}%
                  </Text>
                </View>
              ))}
            </View>
          )}
        </View>

        {/* Actions */}
        {analysis.threatDetected && analysis.confidence > 0.7 && (
          <View style={styles.actions}>
            <View style={styles.warningBanner}>
              <Icon name="warning" size={16} color="#FF9800" />
              <Text style={styles.warningText}>
                High confidence threat detected. Consider reporting to authorities.
              </Text>
            </View>
          </View>
        )}
      </View>
    </Animated.View>
  );
};

// Helper functions
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
      return '#757575'; // Gray
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

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderRadius: 12,
    overflow: 'hidden',
  },
  card: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderLeftWidth: 4,
    margin: 2,
    borderRadius: 8,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0, 0, 0, 0.1)',
  },
  headerIcon: {
    marginRight: 12,
  },
  headerText: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  riskLevel: {
    fontSize: 12,
    fontWeight: '500',
    marginTop: 2,
  },
  confidence: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  content: {
    padding: 16,
  },
  detailRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  detailLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#666',
    marginRight: 8,
  },
  detailValue: {
    fontSize: 14,
    color: '#333',
    textTransform: 'capitalize',
  },
  description: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
    marginBottom: 12,
  },
  objectsSection: {
    marginTop: 8,
  },
  objectsTitle: {
    fontSize: 14,
    fontWeight: '500',
    color: '#666',
    marginBottom: 8,
  },
  objectItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
    paddingHorizontal: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    borderRadius: 4,
    marginBottom: 4,
  },
  objectLabel: {
    fontSize: 12,
    color: '#333',
    textTransform: 'capitalize',
  },
  objectConfidence: {
    fontSize: 12,
    fontWeight: '500',
    color: '#666',
  },
  actions: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.1)',
  },
  warningBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 152, 0, 0.1)',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#FF9800',
  },
  warningText: {
    fontSize: 12,
    color: '#E65100',
    marginLeft: 8,
    flex: 1,
  },
});

export default ThreatAnalysisOverlay;