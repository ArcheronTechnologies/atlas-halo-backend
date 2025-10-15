/**
 * Camera modal for capturing photos with AI threat detection
 */

import React, {useState, useRef, useEffect} from 'react';
import {
  View,
  StyleSheet,
  Modal,
  TouchableOpacity,
  Text,
  Alert,
  Dimensions,
  Platform,
} from 'react-native';
import {RNCamera} from 'react-native-camera';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {useDispatch} from 'react-redux';

import {UserLocation} from '@types/safety';
import {AppDispatch} from '@store/index';
import ThreatAnalysisOverlay from './ThreatAnalysisOverlay';
import ProcessingOverlay from '@components/Common/ProcessingOverlay';

const {width, height} = Dimensions.get('window');

interface CameraModalProps {
  visible: boolean;
  onClose: () => void;
  emergencyMode?: boolean;
  userLocation: UserLocation | null;
}

const CameraModal: React.FC<CameraModalProps> = ({
  visible,
  onClose,
  emergencyMode = false,
  userLocation,
}) => {
  const dispatch = useDispatch<AppDispatch>();
  const cameraRef = useRef<RNCamera>(null);

  // Local state
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraType, setCameraType] = useState(RNCamera.Constants.Type.back);
  const [flashMode, setFlashMode] = useState(RNCamera.Constants.FlashMode.auto);
  const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null);
  const [threatAnalysis, setThreatAnalysis] = useState<any>(null);
  const [autoRecordAudio, setAutoRecordAudio] = useState(emergencyMode);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (visible) {
      setCapturedPhoto(null);
      setThreatAnalysis(null);
      setIsProcessing(false);
    }
  }, [visible]);

  // Auto-capture in emergency mode
  useEffect(() => {
    if (visible && emergencyMode && cameraRef.current) {
      // Auto-capture after 2 seconds in emergency mode
      const timer = setTimeout(() => {
        handleCapture();
      }, 2000);

      return () => clearTimeout(timer);
    }
  }, [visible, emergencyMode]);

  // Handle photo capture
  const handleCapture = async () => {
    if (!cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);

      const options = {
        quality: 0.8,
        base64: true,
        skipProcessing: false,
        forceUpOrientation: true,
        fixOrientation: true,
      };

      const photo = await cameraRef.current.takePictureAsync(options);
      setCapturedPhoto(photo.uri);

      // Process photo with AI threat detection
      await processPhotoWithAI(photo);

    } catch (error) {
      console.error('Camera capture error:', error);
      Alert.alert('Capture Error', 'Failed to capture photo. Please try again.');
      setIsProcessing(false);
    }
  };

  // Process photo with AI threat detection
  const processPhotoWithAI = async (photo: any) => {
    if (!userLocation) {
      setIsProcessing(false);
      return;
    }

    try {
      // Prepare photo data for API
      const photoData = {
        image: photo.base64,
        location: {
          latitude: userLocation.coordinates.latitude,
          longitude: userLocation.coordinates.longitude,
          timestamp: new Date().toISOString(),
        },
        emergencyMode,
      };

      // Call Atlas AI backend for threat detection
      const response = await fetch('/api/mobile/analyze-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(photoData),
      });

      const result = await response.json();

      if (result.success) {
        setThreatAnalysis(result.data);

        // Handle high-threat detections
        if (result.data.threatDetected && result.data.confidence > 0.7) {
          handleHighThreatDetected(result.data);
        }
      } else {
        throw new Error(result.error || 'Analysis failed');
      }

    } catch (error) {
      console.error('AI analysis error:', error);
      Alert.alert(
        'Analysis Error',
        'Failed to analyze photo for threats. The image has been saved locally.',
      );
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle high threat detection
  const handleHighThreatDetected = (analysis: any) => {
    const threatType = analysis.threatType;
    const confidence = Math.round(analysis.confidence * 100);

    Alert.alert(
      '⚠️ Threat Detected',
      `${threatType} detected with ${confidence}% confidence. Do you want to report this to authorities?`,
      [
        {text: 'No', style: 'cancel'},
        {
          text: 'Report',
          style: 'destructive',
          onPress: () => reportToAuthorities(analysis),
        },
      ]
    );
  };

  // Report to authorities
  const reportToAuthorities = async (analysis: any) => {
    try {
      const reportData = {
        type: 'threat_detection',
        location: userLocation,
        threatAnalysis: analysis,
        photoUri: capturedPhoto,
        timestamp: new Date().toISOString(),
        emergencyMode,
      };

      // Send to authorities API
      const response = await fetch('/api/mobile/report-incident', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData),
      });

      if (response.ok) {
        Alert.alert(
          'Report Sent',
          'Your report has been sent to authorities. Please stay safe.',
        );
      } else {
        throw new Error('Report failed');
      }

    } catch (error) {
      console.error('Report error:', error);
      Alert.alert(
        'Report Error',
        'Failed to send report to authorities. Please call emergency services directly if needed.',
      );
    }
  };

  // Toggle camera type
  const toggleCameraType = () => {
    setCameraType(
      cameraType === RNCamera.Constants.Type.back
        ? RNCamera.Constants.Type.front
        : RNCamera.Constants.Type.back
    );
  };

  // Toggle flash mode
  const toggleFlashMode = () => {
    const modes = [
      RNCamera.Constants.FlashMode.auto,
      RNCamera.Constants.FlashMode.on,
      RNCamera.Constants.FlashMode.off,
    ];
    const currentIndex = modes.indexOf(flashMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    setFlashMode(modes[nextIndex]);
  };

  // Get flash icon
  const getFlashIcon = () => {
    switch (flashMode) {
      case RNCamera.Constants.FlashMode.on:
        return 'flash-on';
      case RNCamera.Constants.FlashMode.off:
        return 'flash-off';
      default:
        return 'flash-auto';
    }
  };

  // Handle retake
  const handleRetake = () => {
    setCapturedPhoto(null);
    setThreatAnalysis(null);
  };

  // Handle save and close
  const handleSaveAndClose = () => {
    // Save photo to gallery or app storage
    // Implementation would depend on requirements
    onClose();
  };

  if (!visible) return null;

  return (
    <Modal
      visible={visible}
      animationType="slide"
      statusBarTranslucent={true}
    >
      <View style={styles.container}>
        {/* Camera view */}
        {!capturedPhoto && (
          <RNCamera
            ref={cameraRef}
            style={styles.camera}
            type={cameraType}
            flashMode={flashMode}
            androidCameraPermissionOptions={{
              title: 'Permission to use camera',
              message: 'Atlas AI needs access to camera for threat detection',
              buttonPositive: 'Ok',
              buttonNegative: 'Cancel',
            }}
            captureAudio={autoRecordAudio}
          >
            {/* Emergency mode indicator */}
            {emergencyMode && (
              <View style={styles.emergencyIndicator}>
                <Icon name="emergency" size={24} color="#F44336" />
                <Text style={styles.emergencyText}>EMERGENCY MODE</Text>
              </View>
            )}

            {/* Camera controls */}
            <View style={styles.cameraControls}>
              {/* Top controls */}
              <View style={styles.topControls}>
                <TouchableOpacity
                  style={styles.controlButton}
                  onPress={onClose}
                >
                  <Icon name="close" size={28} color="white" />
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.controlButton}
                  onPress={toggleFlashMode}
                >
                  <Icon name={getFlashIcon()} size={28} color="white" />
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.controlButton}
                  onPress={toggleCameraType}
                >
                  <Icon name="flip-camera-ios" size={28} color="white" />
                </TouchableOpacity>
              </View>

              {/* Bottom controls */}
              <View style={styles.bottomControls}>
                <View style={styles.captureButtonContainer}>
                  <TouchableOpacity
                    style={[
                      styles.captureButton,
                      emergencyMode && styles.emergencyCaptureButton,
                    ]}
                    onPress={handleCapture}
                    disabled={isProcessing}
                  >
                    <View style={styles.captureButtonInner} />
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          </RNCamera>
        )}

        {/* Captured photo view */}
        {capturedPhoto && (
          <View style={styles.previewContainer}>
            {/* Photo preview would go here */}
            <View style={styles.photoPlaceholder}>
              <Icon name="photo" size={100} color="#666" />
              <Text style={styles.photoText}>Photo captured</Text>
            </View>

            {/* Threat analysis overlay */}
            {threatAnalysis && (
              <ThreatAnalysisOverlay
                analysis={threatAnalysis}
                style={styles.analysisOverlay}
              />
            )}

            {/* Preview controls */}
            <View style={styles.previewControls}>
              <TouchableOpacity
                style={styles.previewButton}
                onPress={handleRetake}
              >
                <Icon name="refresh" size={24} color="white" />
                <Text style={styles.previewButtonText}>Retake</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.previewButton, styles.saveButton]}
                onPress={handleSaveAndClose}
              >
                <Icon name="check" size={24} color="white" />
                <Text style={styles.previewButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Processing overlay */}
        <ProcessingOverlay
          visible={isProcessing}
          message="Analyzing image for threats..."
        />
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
  },
  emergencyIndicator: {
    position: 'absolute',
    top: 60,
    left: 20,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(244, 67, 54, 0.9)',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
  },
  emergencyText: {
    color: 'white',
    fontWeight: 'bold',
    marginLeft: 8,
    fontSize: 16,
  },
  cameraControls: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  bottomControls: {
    alignItems: 'center',
    paddingBottom: 50,
  },
  captureButtonContainer: {
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: 'rgba(255, 255, 255, 0.5)',
  },
  emergencyCaptureButton: {
    borderColor: '#F44336',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#2196F3',
  },
  previewContainer: {
    flex: 1,
    backgroundColor: 'black',
  },
  photoPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  photoText: {
    color: 'white',
    fontSize: 18,
    marginTop: 16,
  },
  analysisOverlay: {
    position: 'absolute',
    bottom: 120,
    left: 20,
    right: 20,
  },
  previewControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 40,
    paddingBottom: 50,
  },
  previewButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
  },
  saveButton: {
    backgroundColor: '#4CAF50',
  },
  previewButtonText: {
    color: 'white',
    fontWeight: '500',
    marginLeft: 8,
  },
});

export default CameraModal;