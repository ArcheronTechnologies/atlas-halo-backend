#!/usr/bin/env python3
"""
🛡️ Advanced Spectrum Analysis for SAIT_01
==========================================
Sophisticated audio analysis using FFT, PSD, harmonic analysis, and spectral features
"""

import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq
import librosa
import tensorflow as tf
from typing import Dict, List, Tuple, Optional

class AdvancedSpectrumAnalyzer:
    """Advanced spectrum analysis for enhanced audio threat detection"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate // 2
        
        # Threat signature frequency ranges (Hz)
        self.drone_freq_ranges = [
            (1500, 2500),  # Primary rotor harmonics
            (3000, 4000),  # Secondary harmonics
            (4500, 6000)   # Higher harmonics
        ]
        
        self.helicopter_freq_ranges = [
            (8, 25),       # Main rotor fundamental
            (25, 75),      # Main rotor harmonics
            (100, 300),    # Tail rotor and engine
            (300, 800)     # Higher order harmonics
        ]
        
        # Background noise rejection ranges
        self.natural_noise_ranges = [
            (20, 200),     # Wind, low frequency rumble
            (200, 1000),   # Traffic, urban noise
            (1000, 4000),  # Voices, birds
            (4000, 8000)   # High frequency environmental
        ]
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive spectral features from audio"""
        features = {}
        
        # 1. FFT-based analysis
        fft_features = self._compute_fft_features(audio_data)
        features.update(fft_features)
        
        # 2. Power Spectral Density
        psd_features = self._compute_psd_features(audio_data)
        features.update(psd_features)
        
        # 3. Harmonic analysis
        harmonic_features = self._compute_harmonic_features(audio_data)
        features.update(harmonic_features)
        
        # 4. Spectral shape features
        shape_features = self._compute_spectral_shape_features(audio_data)
        features.update(shape_features)
        
        # 5. Threat-specific signatures
        threat_features = self._compute_threat_signatures(audio_data)
        features.update(threat_features)
        
        return features
    
    def _compute_fft_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute FFT-based spectral features"""
        # Windowed FFT for better frequency resolution
        window = signal.windows.hann(len(audio_data))
        windowed_audio = audio_data * window
        
        # Compute FFT
        fft_vals = fft(windowed_audio)
        fft_freqs = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Magnitude spectrum
        magnitude_spectrum = np.abs(fft_vals[:len(fft_vals)//2])
        freq_bins = fft_freqs[:len(fft_freqs)//2]
        
        # Phase spectrum
        phase_spectrum = np.angle(fft_vals[:len(fft_vals)//2])
        
        # Frequency domain statistics
        peak_freq = freq_bins[np.argmax(magnitude_spectrum)]
        spectral_energy = np.sum(magnitude_spectrum**2)
        
        return {
            'magnitude_spectrum': magnitude_spectrum,
            'phase_spectrum': phase_spectrum,
            'freq_bins': freq_bins,
            'peak_frequency': peak_freq,
            'spectral_energy': spectral_energy
        }
    
    def _compute_psd_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Power Spectral Density features"""
        # Welch's method for PSD estimation
        freqs, psd = signal.welch(
            audio_data, 
            fs=self.sample_rate,
            window='hann',
            nperseg=min(1024, len(audio_data)//4),
            noverlap=None,
            scaling='density'
        )
        
        # Frequency band power analysis
        band_powers = {}
        
        # Define frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, self.nyquist)
        }
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            band_powers[f'{band_name}_power'] = band_power
        
        # Total power
        total_power = np.trapz(psd, freqs)
        
        # Relative band powers (normalized)
        for band_name in bands.keys():
            if total_power > 0:
                band_powers[f'{band_name}_power_norm'] = band_powers[f'{band_name}_power'] / total_power
            else:
                band_powers[f'{band_name}_power_norm'] = 0.0
        
        return {
            'psd_freqs': freqs,
            'psd_values': psd,
            'total_power': total_power,
            **band_powers
        }
    
    def _compute_harmonic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Compute harmonic analysis features for drone/helicopter detection"""
        # Get fundamental frequency using autocorrelation
        fundamental_freq = self._estimate_fundamental_frequency(audio_data)
        
        if fundamental_freq is None or fundamental_freq < 10:
            return {'fundamental_freq': 0.0, 'harmonic_ratio': 0.0, 'harmonic_strength': 0.0, 'num_harmonics': 0}
        
        # Compute FFT for harmonic analysis
        fft_vals = fft(audio_data)
        fft_freqs = fftfreq(len(audio_data), 1/self.sample_rate)
        magnitude_spectrum = np.abs(fft_vals[:len(fft_vals)//2])
        freq_bins = fft_freqs[:len(fft_freqs)//2]
        
        # Find harmonic peaks
        harmonics = []
        harmonic_magnitudes = []
        
        for n in range(1, 11):  # Check first 10 harmonics
            harmonic_freq = fundamental_freq * n
            if harmonic_freq >= self.nyquist:
                break
            
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(freq_bins - harmonic_freq))
            
            # Check in small window around expected harmonic
            window_size = max(1, int(0.1 * harmonic_freq / (self.sample_rate / len(audio_data))))
            start_idx = max(0, freq_idx - window_size)
            end_idx = min(len(magnitude_spectrum), freq_idx + window_size + 1)
            
            # Find peak in window
            window_magnitudes = magnitude_spectrum[start_idx:end_idx]
            if len(window_magnitudes) > 0:
                local_peak_idx = np.argmax(window_magnitudes)
                peak_magnitude = window_magnitudes[local_peak_idx]
                
                harmonics.append(harmonic_freq)
                harmonic_magnitudes.append(peak_magnitude)
        
        # Compute harmonic features
        if len(harmonic_magnitudes) > 1:
            # Harmonic-to-noise ratio
            total_energy = np.sum(magnitude_spectrum**2)
            harmonic_energy = np.sum(np.array(harmonic_magnitudes)**2)
            harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0
            
            # Harmonic strength (regularity of harmonic series)
            harmonic_strength = np.std(harmonic_magnitudes) / (np.mean(harmonic_magnitudes) + 1e-6)
        else:
            harmonic_ratio = 0.0
            harmonic_strength = 0.0
        
        return {
            'fundamental_freq': fundamental_freq,
            'harmonic_ratio': harmonic_ratio,
            'harmonic_strength': harmonic_strength,
            'num_harmonics': len(harmonics)
        }
    
    def _estimate_fundamental_frequency(self, audio_data: np.ndarray) -> Optional[float]:
        """Estimate fundamental frequency using autocorrelation"""
        # Autocorrelation
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        min_period = int(self.sample_rate / 800)  # Max 800 Hz
        max_period = int(self.sample_rate / 50)   # Min 50 Hz
        
        if max_period >= len(autocorr):
            return None
        
        # Find first significant peak
        autocorr_segment = autocorr[min_period:max_period]
        if len(autocorr_segment) == 0:
            return None
        
        # Normalize
        autocorr_segment = autocorr_segment / np.max(autocorr_segment)
        
        # Find peak above threshold
        peak_threshold = 0.3
        peaks, _ = signal.find_peaks(autocorr_segment, height=peak_threshold)
        
        if len(peaks) > 0:
            # First significant peak corresponds to fundamental period
            period = peaks[0] + min_period
            fundamental_freq = self.sample_rate / period
            return fundamental_freq
        
        return None
    
    def _compute_spectral_shape_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Compute spectral shape descriptors"""
        # Use librosa for some features
        stft = librosa.stft(audio_data, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)[0]
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)[0]
        rolloff_mean = np.mean(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)[0]
        bandwidth_mean = np.mean(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        zcr_mean = np.mean(zcr)
        
        # Spectral flatness (measure of noisiness)
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
        flatness_mean = np.mean(spectral_flatness)
        
        return {
            'spectral_centroid_mean': centroid_mean,
            'spectral_centroid_std': centroid_std,
            'spectral_rolloff_mean': rolloff_mean,
            'spectral_bandwidth_mean': bandwidth_mean,
            'zero_crossing_rate_mean': zcr_mean,
            'spectral_flatness_mean': flatness_mean
        }
    
    def _compute_threat_signatures(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Compute threat-specific spectral signatures"""
        # Get PSD for analysis
        freqs, psd = signal.welch(audio_data, fs=self.sample_rate, nperseg=min(1024, len(audio_data)//4))
        
        # Drone signature analysis
        drone_signature = 0.0
        for low_freq, high_freq in self.drone_freq_ranges:
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                drone_signature += band_power
        
        # Helicopter signature analysis
        helicopter_signature = 0.0
        for low_freq, high_freq in self.helicopter_freq_ranges:
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                helicopter_signature += band_power
        
        # Natural noise signature (for rejection)
        natural_noise_signature = 0.0
        for low_freq, high_freq in self.natural_noise_ranges:
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                natural_noise_signature += band_power
        
        # Total power for normalization
        total_power = np.trapz(psd, freqs)
        
        # Normalize signatures
        if total_power > 0:
            drone_signature_norm = drone_signature / total_power
            helicopter_signature_norm = helicopter_signature / total_power
            natural_noise_signature_norm = natural_noise_signature / total_power
        else:
            drone_signature_norm = 0.0
            helicopter_signature_norm = 0.0
            natural_noise_signature_norm = 0.0
        
        # Threat-to-noise ratio
        threat_signature = drone_signature + helicopter_signature
        if natural_noise_signature > 0:
            threat_to_noise_ratio = threat_signature / natural_noise_signature
        else:
            threat_to_noise_ratio = threat_signature
        
        return {
            'drone_signature': drone_signature_norm,
            'helicopter_signature': helicopter_signature_norm,
            'natural_noise_signature': natural_noise_signature_norm,
            'threat_to_noise_ratio': threat_to_noise_ratio,
            'total_threat_signature': drone_signature_norm + helicopter_signature_norm
        }
    
    def create_enhanced_feature_vector(self, audio_data: np.ndarray) -> np.ndarray:
        """Create comprehensive feature vector for ML model input"""
        features = self.extract_spectral_features(audio_data)
        
        # Extract key numerical features for ML
        feature_vector = []
        
        # Spectral shape features
        feature_vector.extend([
            features['spectral_centroid_mean'],
            features['spectral_centroid_std'],
            features['spectral_rolloff_mean'],
            features['spectral_bandwidth_mean'],
            features['zero_crossing_rate_mean'],
            features['spectral_flatness_mean']
        ])
        
        # Harmonic features
        feature_vector.extend([
            features['fundamental_freq'],
            features['harmonic_ratio'],
            features['harmonic_strength'],
            features['num_harmonics']
        ])
        
        # Power band features (normalized)
        band_features = [
            'sub_bass_power_norm', 'bass_power_norm', 'low_mid_power_norm',
            'mid_power_norm', 'high_mid_power_norm', 'presence_power_norm',
            'brilliance_power_norm'
        ]
        for band_feature in band_features:
            feature_vector.append(features[band_feature])
        
        # Threat signatures
        feature_vector.extend([
            features['drone_signature'],
            features['helicopter_signature'],
            features['natural_noise_signature'],
            features['threat_to_noise_ratio'],
            features['total_threat_signature']
        ])
        
        return np.array(feature_vector, dtype=np.float32)

class EnhancedSpectralClassifier:
    """Enhanced classifier using advanced spectral features"""
    
    def __init__(self, sample_rate: int = 16000):
        self.analyzer = AdvancedSpectrumAnalyzer(sample_rate)
        self.feature_dim = 23  # Total number of features
        
    def create_spectral_model(self) -> tf.keras.Model:
        """Create model architecture optimized for spectral features"""
        inputs = tf.keras.layers.Input(shape=(self.feature_dim,), name='spectral_features')
        
        # Feature normalization
        x = tf.keras.layers.BatchNormalization()(inputs)
        
        # Multi-branch architecture
        # Branch 1: Spectral shape features (first 6 features)
        shape_branch = tf.keras.layers.Lambda(lambda x: x[:, :6])(x)
        shape_branch = tf.keras.layers.Dense(16, activation='relu')(shape_branch)
        shape_branch = tf.keras.layers.Dropout(0.3)(shape_branch)
        shape_branch = tf.keras.layers.Dense(8, activation='relu')(shape_branch)
        
        # Branch 2: Harmonic features (next 4 features)
        harmonic_branch = tf.keras.layers.Lambda(lambda x: x[:, 6:10])(x)
        harmonic_branch = tf.keras.layers.Dense(12, activation='relu')(harmonic_branch)
        harmonic_branch = tf.keras.layers.Dropout(0.3)(harmonic_branch)
        harmonic_branch = tf.keras.layers.Dense(6, activation='relu')(harmonic_branch)
        
        # Branch 3: Power band features (next 7 features)
        power_branch = tf.keras.layers.Lambda(lambda x: x[:, 10:17])(x)
        power_branch = tf.keras.layers.Dense(14, activation='relu')(power_branch)
        power_branch = tf.keras.layers.Dropout(0.3)(power_branch)
        power_branch = tf.keras.layers.Dense(8, activation='relu')(power_branch)
        
        # Branch 4: Threat signatures (last 6 features)
        threat_branch = tf.keras.layers.Lambda(lambda x: x[:, 17:])(x)
        threat_branch = tf.keras.layers.Dense(12, activation='relu')(threat_branch)
        threat_branch = tf.keras.layers.Dropout(0.3)(threat_branch)
        threat_branch = tf.keras.layers.Dense(6, activation='relu')(threat_branch)
        
        # Combine branches
        combined = tf.keras.layers.concatenate([shape_branch, harmonic_branch, power_branch, threat_branch])
        
        # Final classification layers
        x = tf.keras.layers.Dense(32, activation='relu')(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='spectral_classifier')
        
        return model
    
    def predict(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Make prediction using spectral features"""
        # Extract features
        feature_vector = self.analyzer.create_enhanced_feature_vector(audio_data)
        
        # Add batch dimension
        features_batch = feature_vector[np.newaxis, :]
        
        # Simple rule-based classification for testing
        # This would be replaced with trained model inference
        
        # Threat detection logic based on spectral signatures
        threat_score = feature_vector[21]  # total_threat_signature
        noise_score = feature_vector[20]   # natural_noise_signature
        harmonic_ratio = feature_vector[7] # harmonic_ratio
        
        # Classification logic
        if threat_score > 0.1 and harmonic_ratio > 0.2:
            # High threat signature with harmonic content
            if feature_vector[18] > feature_vector[19]:  # drone > helicopter
                predicted_class = 1  # Drone
                confidence = min(0.95, 0.5 + threat_score * 2)
            else:
                predicted_class = 2  # Helicopter
                confidence = min(0.95, 0.5 + threat_score * 2)
        elif noise_score > 0.3:
            # High natural noise signature
            predicted_class = 0  # Background
            confidence = min(0.95, 0.5 + noise_score * 1.5)
        else:
            # Ambiguous - default to background
            predicted_class = 0
            confidence = 0.6
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'features': feature_vector.tolist(),
            'source': 'spectral_analysis'
        }

def test_advanced_spectrum_analysis():
    """Test advanced spectrum analysis capabilities"""
    print("🔬 Testing Advanced Spectrum Analysis")
    print("=" * 45)
    
    analyzer = AdvancedSpectrumAnalyzer()
    classifier = EnhancedSpectralClassifier()
    
    # Generate test signals
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    test_signals = {
        "Pure drone (2000 Hz)": 0.5 * np.sin(2 * np.pi * 2000 * t),
        "Harmonic drone": 0.3 * np.sin(2 * np.pi * 1800 * t) + 0.2 * np.sin(2 * np.pi * 3600 * t),
        "Helicopter rotor": 0.4 * np.sin(2 * np.pi * 15 * t) + 0.3 * np.sin(2 * np.pi * 30 * t),
        "White noise": np.random.randn(len(t)) * 0.3,
        "Wind noise": 0.2 * np.random.randn(len(t)) * np.exp(-t/0.5)
    }
    
    for signal_name, signal_data in test_signals.items():
        print(f"\n📊 Analyzing: {signal_name}")
        
        # Extract features
        features = analyzer.extract_spectral_features(signal_data)
        feature_vector = analyzer.create_enhanced_feature_vector(signal_data)
        
        # Make prediction
        prediction = classifier.predict(signal_data)
        
        # Display key results
        class_names = ["Background", "Drone", "Helicopter"]
        print(f"   Classification: {class_names[prediction['predicted_class']]} ({prediction['confidence']:.3f})")
        print(f"   Fundamental: {features['fundamental_freq']:.1f} Hz")
        print(f"   Harmonic ratio: {features['harmonic_ratio']:.3f}")
        print(f"   Threat signature: {features['total_threat_signature']:.3f}")
        print(f"   Feature vector shape: {feature_vector.shape}")

if __name__ == "__main__":
    test_advanced_spectrum_analysis()