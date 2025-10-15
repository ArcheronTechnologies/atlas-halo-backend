#!/usr/bin/env python3
"""
🛡️ Hardened Advanced Spectrum Analysis - VULNERABILITY FIXES
============================================================
Fixed version with proper input validation and error handling
"""

import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq
import librosa
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import warnings
import functools
import time

def timeout_decorator(timeout_seconds=5):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except Exception:
                signal.alarm(0)  # Cancel alarm
                raise
        return wrapper
    return decorator

def validate_and_sanitize_audio(audio_data: np.ndarray, min_length: int = 1000, max_length: int = 160000) -> np.ndarray:
    """
    Validate and sanitize audio input to prevent crashes
    
    Args:
        audio_data: Input audio array
        min_length: Minimum required length
        max_length: Maximum allowed length
        
    Returns:
        Sanitized audio array
        
    Raises:
        ValueError: If input is fundamentally invalid
    """
    # Type validation
    if not isinstance(audio_data, np.ndarray):
        raise ValueError(f"Audio must be numpy array, got {type(audio_data)}")
    
    # Flatten if multi-dimensional
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
        warnings.warn("Multi-dimensional audio flattened to 1D")
    
    # Handle empty arrays
    if len(audio_data) == 0:
        warnings.warn("Empty audio array, returning silence")
        return np.zeros(min_length, dtype=np.float32)
    
    # Handle single samples
    if len(audio_data) == 1:
        # Repeat single sample to minimum length
        warnings.warn("Single sample audio, padding to minimum length")
        return np.full(min_length, audio_data[0], dtype=np.float32)
    
    # Length validation
    if len(audio_data) > max_length:
        warnings.warn(f"Audio too long ({len(audio_data)}), truncating to {max_length}")
        audio_data = audio_data[:max_length]
    
    # Pad short audio
    if len(audio_data) < min_length:
        padding = min_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
    
    # Sanitize values
    # Replace inf/nan with zeros
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Clip extreme values
    audio_data = np.clip(audio_data, -10.0, 10.0)
    
    # Ensure float32 type
    audio_data = audio_data.astype(np.float32)
    
    return audio_data

class HardenedAdvancedSpectrumAnalyzer:
    """Hardened spectrum analyzer with comprehensive error handling"""
    
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
    
    @timeout_decorator(timeout_seconds=10)
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive spectral features with error handling"""
        try:
            # Validate and sanitize input
            audio_data = validate_and_sanitize_audio(audio_data)
            
            features = {}
            
            # 1. FFT-based analysis with error handling
            try:
                fft_features = self._compute_fft_features_safe(audio_data)
                features.update(fft_features)
            except Exception as e:
                warnings.warn(f"FFT feature extraction failed: {e}")
                features.update(self._get_default_fft_features())
            
            # 2. Power Spectral Density with error handling
            try:
                psd_features = self._compute_psd_features_safe(audio_data)
                features.update(psd_features)
            except Exception as e:
                warnings.warn(f"PSD feature extraction failed: {e}")
                features.update(self._get_default_psd_features())
            
            # 3. Harmonic analysis with error handling
            try:
                harmonic_features = self._compute_harmonic_features_safe(audio_data)
                features.update(harmonic_features)
            except Exception as e:
                warnings.warn(f"Harmonic feature extraction failed: {e}")
                features.update(self._get_default_harmonic_features())
            
            # 4. Spectral shape features with error handling
            try:
                shape_features = self._compute_spectral_shape_features_safe(audio_data)
                features.update(shape_features)
            except Exception as e:
                warnings.warn(f"Spectral shape feature extraction failed: {e}")
                features.update(self._get_default_shape_features())
            
            # 5. Threat-specific signatures with error handling
            try:
                threat_features = self._compute_threat_signatures_safe(audio_data)
                features.update(threat_features)
            except Exception as e:
                warnings.warn(f"Threat signature extraction failed: {e}")
                features.update(self._get_default_threat_features())
            
            return features
            
        except Exception as e:
            warnings.warn(f"Complete feature extraction failed: {e}")
            return self._get_emergency_features()
    
    def _compute_fft_features_safe(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute FFT-based spectral features with safety checks"""
        # Ensure minimum length for FFT
        if len(audio_data) < 4:
            raise ValueError("Audio too short for FFT analysis")
        
        # Use appropriate window size
        window_size = min(len(audio_data), 1024)
        if window_size > len(audio_data):
            window_size = len(audio_data)
        
        # Create window
        try:
            window = signal.windows.hann(len(audio_data))
            windowed_audio = audio_data * window
        except Exception:
            # Fallback: no windowing
            windowed_audio = audio_data
        
        # Compute FFT with appropriate size
        fft_vals = fft(windowed_audio)
        fft_freqs = fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Magnitude spectrum
        magnitude_spectrum = np.abs(fft_vals[:len(fft_vals)//2])
        freq_bins = fft_freqs[:len(fft_freqs)//2]
        
        # Phase spectrum
        phase_spectrum = np.angle(fft_vals[:len(fft_vals)//2])
        
        # Safe peak frequency finding
        if len(magnitude_spectrum) > 0:
            peak_idx = np.argmax(magnitude_spectrum)
            peak_freq = freq_bins[peak_idx] if peak_idx < len(freq_bins) else 0.0
        else:
            peak_freq = 0.0
        
        # Safe energy calculation
        spectral_energy = np.sum(magnitude_spectrum**2) if len(magnitude_spectrum) > 0 else 0.0
        
        return {
            'magnitude_spectrum': magnitude_spectrum,
            'phase_spectrum': phase_spectrum,
            'freq_bins': freq_bins,
            'peak_frequency': float(peak_freq),
            'spectral_energy': float(spectral_energy)
        }
    
    def _compute_psd_features_safe(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Power Spectral Density features with safety checks"""
        # Use safe nperseg
        nperseg = min(1024, len(audio_data)//4, len(audio_data))
        if nperseg < 4:
            nperseg = len(audio_data)
        
        try:
            freqs, psd = signal.welch(
                audio_data, 
                fs=self.sample_rate,
                window='hann',
                nperseg=nperseg,
                noverlap=None,
                scaling='density'
            )
        except Exception:
            # Fallback PSD calculation
            freqs = np.linspace(0, self.nyquist, len(audio_data)//2)
            psd = np.abs(fft(audio_data)[:len(freqs)])**2
        
        # Safe frequency band analysis
        band_powers = {}
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
            try:
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                else:
                    band_power = 0.0
                band_powers[f'{band_name}_power'] = float(band_power)
            except Exception:
                band_powers[f'{band_name}_power'] = 0.0
        
        # Total power
        try:
            total_power = np.trapz(psd, freqs)
        except Exception:
            total_power = np.sum(psd) if len(psd) > 0 else 0.0
        
        # Relative band powers (normalized)
        for band_name in bands.keys():
            if total_power > 0:
                band_powers[f'{band_name}_power_norm'] = band_powers[f'{band_name}_power'] / total_power
            else:
                band_powers[f'{band_name}_power_norm'] = 0.0
        
        return {
            'psd_freqs': freqs,
            'psd_values': psd,
            'total_power': float(total_power),
            **band_powers
        }
    
    def _compute_harmonic_features_safe(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Compute harmonic analysis features with comprehensive error handling"""
        try:
            # Safe fundamental frequency estimation
            fundamental_freq = self._estimate_fundamental_frequency_safe(audio_data)
            
            if fundamental_freq is None or fundamental_freq < 10 or fundamental_freq > self.nyquist:
                return {
                    'fundamental_freq': 0.0, 
                    'harmonic_ratio': 0.0, 
                    'harmonic_strength': 0.0, 
                    'num_harmonics': 0
                }
            
            # Safe FFT for harmonic analysis
            if len(audio_data) < 4:
                return {
                    'fundamental_freq': float(fundamental_freq),
                    'harmonic_ratio': 0.0,
                    'harmonic_strength': 0.0,
                    'num_harmonics': 0
                }
            
            fft_vals = fft(audio_data)
            fft_freqs = fftfreq(len(audio_data), 1/self.sample_rate)
            magnitude_spectrum = np.abs(fft_vals[:len(fft_vals)//2])
            freq_bins = fft_freqs[:len(fft_freqs)//2]
            
            # Find harmonic peaks safely
            harmonics = []
            harmonic_magnitudes = []
            
            for n in range(1, 11):  # Check first 10 harmonics
                harmonic_freq = fundamental_freq * n
                if harmonic_freq >= self.nyquist:
                    break
                
                try:
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
                except Exception:
                    continue
            
            # Compute harmonic features safely
            if len(harmonic_magnitudes) > 1:
                try:
                    # Harmonic-to-noise ratio
                    total_energy = np.sum(magnitude_spectrum**2)
                    harmonic_energy = np.sum(np.array(harmonic_magnitudes)**2)
                    harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0
                    
                    # Harmonic strength (regularity of harmonic series)
                    mean_mag = np.mean(harmonic_magnitudes)
                    harmonic_strength = np.std(harmonic_magnitudes) / (mean_mag + 1e-6) if mean_mag > 0 else 0.0
                except Exception:
                    harmonic_ratio = 0.0
                    harmonic_strength = 0.0
            else:
                harmonic_ratio = 0.0
                harmonic_strength = 0.0
            
            return {
                'fundamental_freq': float(fundamental_freq),
                'harmonic_ratio': float(harmonic_ratio),
                'harmonic_strength': float(harmonic_strength),
                'num_harmonics': len(harmonics)
            }
            
        except Exception:
            return {
                'fundamental_freq': 0.0,
                'harmonic_ratio': 0.0,
                'harmonic_strength': 0.0,
                'num_harmonics': 0
            }
    
    def _estimate_fundamental_frequency_safe(self, audio_data: np.ndarray) -> Optional[float]:
        """Estimate fundamental frequency with comprehensive error handling"""
        try:
            if len(audio_data) < 100:  # Need minimum samples
                return None
            
            # Autocorrelation
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            min_period = max(1, int(self.sample_rate / 800))  # Max 800 Hz
            max_period = min(len(autocorr), int(self.sample_rate / 50))   # Min 50 Hz
            
            if max_period <= min_period or max_period >= len(autocorr):
                return None
            
            # Find first significant peak
            autocorr_segment = autocorr[min_period:max_period]
            if len(autocorr_segment) == 0:
                return None
            
            # Normalize safely
            max_val = np.max(autocorr_segment)
            if max_val == 0:
                return None
            autocorr_segment = autocorr_segment / max_val
            
            # Find peak above threshold
            peak_threshold = 0.3
            peaks, _ = signal.find_peaks(autocorr_segment, height=peak_threshold)
            
            if len(peaks) > 0:
                # First significant peak corresponds to fundamental period
                period = peaks[0] + min_period
                fundamental_freq = self.sample_rate / period
                return fundamental_freq
            
            return None
            
        except Exception:
            return None
    
    def _compute_spectral_shape_features_safe(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Compute spectral shape descriptors with error handling"""
        try:
            # Use librosa with safe parameters
            n_fft = min(1024, len(audio_data))
            hop_length = min(256, len(audio_data)//4)
            
            if n_fft < 4 or hop_length < 1:
                return self._get_default_shape_features()
            
            stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Spectral centroid
            try:
                spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)[0]
                centroid_mean = float(np.mean(spectral_centroids))
                centroid_std = float(np.std(spectral_centroids))
            except Exception:
                centroid_mean = 0.0
                centroid_std = 0.0
            
            # Spectral rolloff
            try:
                spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)[0]
                rolloff_mean = float(np.mean(spectral_rolloff))
            except Exception:
                rolloff_mean = 0.0
            
            # Spectral bandwidth
            try:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)[0]
                bandwidth_mean = float(np.mean(spectral_bandwidth))
            except Exception:
                bandwidth_mean = 0.0
            
            # Zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
                zcr_mean = float(np.mean(zcr))
            except Exception:
                zcr_mean = 0.0
            
            # Spectral flatness
            try:
                spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
                flatness_mean = float(np.mean(spectral_flatness))
            except Exception:
                flatness_mean = 0.0
            
            return {
                'spectral_centroid_mean': centroid_mean,
                'spectral_centroid_std': centroid_std,
                'spectral_rolloff_mean': rolloff_mean,
                'spectral_bandwidth_mean': bandwidth_mean,
                'zero_crossing_rate_mean': zcr_mean,
                'spectral_flatness_mean': flatness_mean
            }
            
        except Exception:
            return self._get_default_shape_features()
    
    def _compute_threat_signatures_safe(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Compute threat-specific spectral signatures with error handling"""
        try:
            # Get PSD safely
            nperseg = min(1024, len(audio_data)//4, len(audio_data))
            if nperseg < 4:
                return self._get_default_threat_features()
            
            freqs, psd = signal.welch(audio_data, fs=self.sample_rate, nperseg=nperseg)
            
            # Drone signature analysis
            drone_signature = 0.0
            for low_freq, high_freq in self.drone_freq_ranges:
                try:
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        drone_signature += band_power
                except Exception:
                    continue
            
            # Helicopter signature analysis
            helicopter_signature = 0.0
            for low_freq, high_freq in self.helicopter_freq_ranges:
                try:
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        helicopter_signature += band_power
                except Exception:
                    continue
            
            # Natural noise signature
            natural_noise_signature = 0.0
            for low_freq, high_freq in self.natural_noise_ranges:
                try:
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(band_mask):
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        natural_noise_signature += band_power
                except Exception:
                    continue
            
            # Total power for normalization
            try:
                total_power = np.trapz(psd, freqs)
            except Exception:
                total_power = np.sum(psd) if len(psd) > 0 else 1.0
            
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
                'drone_signature': float(drone_signature_norm),
                'helicopter_signature': float(helicopter_signature_norm),
                'natural_noise_signature': float(natural_noise_signature_norm),
                'threat_to_noise_ratio': float(threat_to_noise_ratio),
                'total_threat_signature': float(drone_signature_norm + helicopter_signature_norm)
            }
            
        except Exception:
            return self._get_default_threat_features()
    
    # Default feature sets for error recovery
    def _get_default_fft_features(self) -> Dict:
        return {
            'magnitude_spectrum': np.array([0.0]),
            'phase_spectrum': np.array([0.0]),
            'freq_bins': np.array([0.0]),
            'peak_frequency': 0.0,
            'spectral_energy': 0.0
        }
    
    def _get_default_psd_features(self) -> Dict:
        bands = ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'presence', 'brilliance']
        features = {
            'psd_freqs': np.array([0.0]),
            'psd_values': np.array([0.0]),
            'total_power': 0.0
        }
        for band in bands:
            features[f'{band}_power'] = 0.0
            features[f'{band}_power_norm'] = 0.0
        return features
    
    def _get_default_harmonic_features(self) -> Dict:
        return {
            'fundamental_freq': 0.0,
            'harmonic_ratio': 0.0,
            'harmonic_strength': 0.0,
            'num_harmonics': 0
        }
    
    def _get_default_shape_features(self) -> Dict:
        return {
            'spectral_centroid_mean': 0.0,
            'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0,
            'spectral_bandwidth_mean': 0.0,
            'zero_crossing_rate_mean': 0.0,
            'spectral_flatness_mean': 0.0
        }
    
    def _get_default_threat_features(self) -> Dict:
        return {
            'drone_signature': 0.0,
            'helicopter_signature': 0.0,
            'natural_noise_signature': 1.0,  # Default to natural noise
            'threat_to_noise_ratio': 0.0,
            'total_threat_signature': 0.0
        }
    
    def _get_emergency_features(self) -> Dict:
        """Emergency feature set when everything fails"""
        features = {}
        features.update(self._get_default_fft_features())
        features.update(self._get_default_psd_features())
        features.update(self._get_default_harmonic_features())
        features.update(self._get_default_shape_features())
        features.update(self._get_default_threat_features())
        return features

def test_hardened_analyzer():
    """Test the hardened analyzer with edge cases"""
    print("🛡️ Testing Hardened Spectrum Analyzer")
    print("=" * 45)
    
    analyzer = HardenedAdvancedSpectrumAnalyzer()
    
    # Test cases that previously broke the system
    test_cases = [
        ("Empty array", np.array([])),
        ("Single sample", np.array([0.5])),
        ("Two samples", np.array([1.0, -1.0])),
        ("Infinite values", np.full(1000, np.inf)),
        ("NaN values", np.full(1000, np.nan)),
        ("Normal audio", np.random.randn(16000) * 0.3),
        ("String input", "not_audio"),
        ("None input", None),
    ]
    
    for test_name, audio_data in test_cases:
        print(f"   Testing: {test_name}")
        start_time = time.time()
        
        try:
            if isinstance(audio_data, str) or audio_data is None:
                # These should raise ValueError
                features = analyzer.extract_spectral_features(audio_data)
                print(f"     ⚠️ Unexpectedly accepted invalid input")
            else:
                features = analyzer.extract_spectral_features(audio_data)
                processing_time = (time.time() - start_time) * 1000
                print(f"     ✅ Success: {processing_time:.1f}ms")
        except ValueError as e:
            print(f"     ✅ Properly rejected: {e}")
        except Exception as e:
            print(f"     💀 Unexpected error: {e}")

if __name__ == "__main__":
    test_hardened_analyzer()