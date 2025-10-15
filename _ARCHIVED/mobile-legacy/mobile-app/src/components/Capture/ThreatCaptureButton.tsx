/**
 * Floating action button for camera/microphone capture with AI threat detection
 */

import React, {useState, useRef} from 'react';
import {
  View,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Alert,
  Vibration,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import {useDispatch, useSelector} from 'react-redux';

import {selectUserLocation} from '@store/slices/mapSlice';
import CameraModal from './CameraModal';
import AudioRecorderModal from './AudioRecorderModal';
import {AppDispatch} from '@store/index';

interface ThreatCaptureButtonProps {
  style?: any;
  size?: number;
  emergencyMode?: boolean;
}

const ThreatCaptureButton: React.FC<ThreatCaptureButtonProps> = ({
  style,
  size = 60,
  emergencyMode = false,
}) => {
  const dispatch = useDispatch<AppDispatch>();
  const userLocation = useSelector(selectUserLocation);

  // Local state
  const [isExpanded, setIsExpanded] = useState(false);
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [showAudioModal, setShowAudioModal] = useState(false);

  // Animation values
  const expandAnimation = useRef(new Animated.Value(0)).current;
  const rotateAnimation = useRef(new Animated.Value(0)).current;
  const pulseAnimation = useRef(new Animated.Value(1)).current;

  // Start pulse animation for emergency mode
  React.useEffect(() => {
    if (emergencyMode) {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnimation, {
            toValue: 1.2,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnimation, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();

      return () => pulse.stop();
    }
  }, [emergencyMode, pulseAnimation]);

  // Toggle expanded state
  const toggleExpanded = () => {
    const toValue = isExpanded ? 0 : 1;
    setIsExpanded(!isExpanded);

    Animated.parallel([
      Animated.spring(expandAnimation, {
        toValue,
        useNativeDriver: true,
        tension: 100,
        friction: 8,
      }),
      Animated.timing(rotateAnimation, {
        toValue,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();

    // Haptic feedback
    Vibration.vibrate(50);
  };

  // Handle camera capture
  const handleCameraCapture = () => {
    if (!userLocation) {
      Alert.alert(
        'Location Required',
        'Location access is required to capture and analyze incidents.',
        [
          {text: 'Cancel', style: 'cancel'},
          {text: 'Enable Location', onPress: () => {/* Request location */}},
        ]
      );
      return;
    }

    toggleExpanded(); // Close expanded menu
    setTimeout(() => {
      setShowCameraModal(true);
    }, 300);
  };

  // Handle audio recording
  const handleAudioCapture = () => {
    if (!userLocation) {
      Alert.alert(
        'Location Required',
        'Location access is required to capture and analyze incidents.',
        [
          {text: 'Cancel', style: 'cancel'},
          {text: 'Enable Location', onPress: () => {/* Request location */}},
        ]
      );
      return;
    }

    toggleExpanded(); // Close expanded menu
    setTimeout(() => {
      setShowAudioModal(true);
    }, 300);
  };

  // Handle emergency mode (both camera and audio)
  const handleEmergencyCapture = () => {
    if (!userLocation) {
      Alert.alert(
        'Location Required',
        'Location access is required for emergency reporting.',
      );
      return;
    }

    Alert.alert(
      'Emergency Mode',
      'This will simultaneously capture photo and audio evidence. Continue?',
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Start Emergency Capture',
          style: 'destructive',
          onPress: () => {
            setShowCameraModal(true);
            // Audio recording will start automatically in emergency mode
          },
        },
      ]
    );
  };

  // Main button press handler
  const handleMainButtonPress = () => {
    if (emergencyMode) {
      handleEmergencyCapture();
    } else {
      toggleExpanded();
    }
  };

  // Animation interpolations
  const expandScale = expandAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
  });

  const expandOpacity = expandAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
  });

  const rotateValue = rotateAnimation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '45deg'],
  });

  const optionButtonTransforms = [
    {
      translateY: expandAnimation.interpolate({
        inputRange: [0, 1],
        outputRange: [0, -(size + 20)],
      }),
    },
    {
      translateX: expandAnimation.interpolate({
        inputRange: [0, 1],
        outputRange: [0, -(size + 20)],
      }),
    },
  ];

  return (
    <View style={[styles.container, style]}>
      {/* Camera option button */}
      {!emergencyMode && (
        <Animated.View
          style={[
            styles.optionButton,
            {
              width: size * 0.7,
              height: size * 0.7,
              borderRadius: (size * 0.7) / 2,
              opacity: expandOpacity,
              transform: [
                {scale: expandScale},
                {translateY: optionButtonTransforms[0].translateY},
              ],
            },
          ]}
        >
          <TouchableOpacity
            style={styles.optionButtonInner}
            onPress={handleCameraCapture}
            disabled={!isExpanded}
          >
            <Icon
              name="camera-alt"
              size={size * 0.35}
              color="white"
            />
          </TouchableOpacity>
        </Animated.View>
      )}

      {/* Audio option button */}
      {!emergencyMode && (
        <Animated.View
          style={[
            styles.optionButton,
            {
              width: size * 0.7,
              height: size * 0.7,
              borderRadius: (size * 0.7) / 2,
              opacity: expandOpacity,
              transform: [
                {scale: expandScale},
                {translateX: optionButtonTransforms[1].translateX},
              ],
            },
          ]}
        >
          <TouchableOpacity
            style={styles.optionButtonInner}
            onPress={handleAudioCapture}
            disabled={!isExpanded}
          >
            <Icon
              name="mic"
              size={size * 0.35}
              color="white"
            />
          </TouchableOpacity>
        </Animated.View>
      )}

      {/* Main button */}
      <Animated.View
        style={[
          styles.mainButton,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: emergencyMode ? '#F44336' : '#2196F3',
            transform: [
              {scale: pulseAnimation},
              {rotate: rotateValue},
            ],
          },
        ]}
      >
        <TouchableOpacity
          style={styles.mainButtonInner}
          onPress={handleMainButtonPress}
          activeOpacity={0.8}
        >
          <Icon
            name={emergencyMode ? "emergency" : (isExpanded ? "close" : "add")}
            size={size * 0.4}
            color="white"
          />
        </TouchableOpacity>
      </Animated.View>

      {/* Modals */}
      <CameraModal
        visible={showCameraModal}
        onClose={() => setShowCameraModal(false)}
        emergencyMode={emergencyMode}
        userLocation={userLocation}
      />

      <AudioRecorderModal
        visible={showAudioModal}
        onClose={() => setShowAudioModal(false)}
        userLocation={userLocation}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    alignItems: 'center',
    justifyContent: 'center',
  },
  mainButton: {
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.3,
    shadowRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  mainButtonInner: {
    flex: 1,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  optionButton: {
    position: 'absolute',
    backgroundColor: '#666',
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 3},
    shadowOpacity: 0.25,
    shadowRadius: 6,
    justifyContent: 'center',
    alignItems: 'center',
  },
  optionButtonInner: {
    flex: 1,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default ThreatCaptureButton;