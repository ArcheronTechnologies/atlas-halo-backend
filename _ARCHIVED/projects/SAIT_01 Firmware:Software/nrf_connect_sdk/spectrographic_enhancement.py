#!/usr/bin/env python3
"""
Spectrographic Enhancement - Research implementation
Incorporate advanced frequency domain analysis into existing best model
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import librosa
import scipy.signal
import scipy.fftpack
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class SpectrogramicEnhancement:
    """Research-focused spectrographic analysis enhancement"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        self.sample_rate = 16000
        
    def extract_enhanced_features(self, audio):
        """Extract comprehensive spectrographic features for research"""
        features = {}
        
        # 1. Standard mel-spectrogram (baseline)
        mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
        if len(mel_spec.shape) == 2:
            mel_spec = np.expand_dims(mel_spec, axis=-1)
        features['baseline_mel'] = mel_spec
        
        # 2. Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Extract spectrograms for each component
        harmonic_mel = librosa.feature.melspectrogram(y=harmonic, sr=self.sample_rate, n_mels=64)
        percussive_mel = librosa.feature.melspectrogram(y=percussive, sr=self.sample_rate, n_mels=64)
        
        # Resize to match baseline
        harmonic_mel = self._resize_to_target(harmonic_mel, (64, 63))
        percussive_mel = self._resize_to_target(percussive_mel, (64, 63))
        
        features['harmonic_mel'] = np.expand_dims(librosa.amplitude_to_db(harmonic_mel), axis=-1)
        features['percussive_mel'] = np.expand_dims(librosa.amplitude_to_db(percussive_mel), axis=-1)
        
        # 3. Constant-Q transform (logarithmic frequency)
        try:
            cqt = librosa.cqt(y=audio, sr=self.sample_rate, hop_length=256, n_bins=64)
            cqt_magnitude = np.abs(cqt)
            cqt_db = librosa.amplitude_to_db(cqt_magnitude)
            cqt_resized = self._resize_to_target(cqt_db, (64, 63))
            features['cqt'] = np.expand_dims(cqt_resized, axis=-1)
        except:
            features['cqt'] = np.zeros((64, 63, 1))
        
        # 4. Multi-resolution spectrograms
        window_sizes = [512, 1024, 2048]
        for i, n_fft in enumerate(window_sizes):
            try:
                stft = librosa.stft(audio, n_fft=n_fft, hop_length=n_fft//4)
                magnitude = np.abs(stft)
                magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
                magnitude_resized = self._resize_to_target(magnitude_db, (64, 63))
                features[f'stft_res_{i}'] = np.expand_dims(magnitude_resized, axis=-1)
            except:
                features[f'stft_res_{i}'] = np.zeros((64, 63, 1))
        
        # 5. Frequency band energy analysis
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        freq_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250), 
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 8000)
        }
        
        band_energies = []
        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energy = np.sum(magnitude[band_mask]**2)
            band_energies.append(band_energy)
        
        # Normalize band energies
        total_energy = sum(band_energies) + 1e-8
        features['band_energies'] = np.array(band_energies) / total_energy
        
        # 6. Spectral characteristics
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0] 
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Take statistical measures
            features['spectral_stats'] = np.array([
                np.mean(centroid), np.std(centroid),
                np.mean(bandwidth), np.std(bandwidth), 
                np.mean(rolloff), np.std(rolloff),
                np.mean(zcr), np.std(zcr)
            ])
        except:
            features['spectral_stats'] = np.zeros(8)
        
        return features
    
    def _resize_to_target(self, spectrogram, target_shape):
        """Resize spectrogram to target shape"""
        if spectrogram.shape == target_shape:
            return spectrogram
        
        # Use scipy's zoom for resizing
        zoom_factors = (target_shape[0] / spectrogram.shape[0], 
                       target_shape[1] / spectrogram.shape[1])
        
        try:
            import scipy.ndimage
            return scipy.ndimage.zoom(spectrogram, zoom_factors, order=1)
        except:
            # Fallback to simple interpolation
            return np.resize(spectrogram, target_shape)
    
    def test_enhanced_features(self, audio_files, labels, num_samples=200):
        """Test enhanced features on sample data"""
        print("🔍 TESTING ENHANCED SPECTROGRAPHIC FEATURES")
        print("=" * 60)
        
        # Extract features for sample
        enhanced_features_list = []
        baseline_features_list = []
        sample_labels = []
        
        print(f"📊 Processing {num_samples} samples...")
        
        for i, (audio_file, label) in enumerate(zip(audio_files[:num_samples], labels[:num_samples])):
            if i % 50 == 0:
                print(f"   Processing sample {i+1}/{num_samples}")
            
            try:
                audio = self.preprocessor.load_and_resample(audio_file)
                features = self.extract_enhanced_features(audio)
                
                # Baseline features (just mel-spectrogram)
                baseline_features_list.append(features['baseline_mel'])
                
                # Enhanced features (combine multiple representations)
                enhanced_feature = np.concatenate([
                    features['baseline_mel'].flatten(),
                    features['harmonic_mel'].flatten(),
                    features['percussive_mel'].flatten(),
                    features['cqt'].flatten(),
                    features['band_energies'],
                    features['spectral_stats']
                ])
                
                enhanced_features_list.append(enhanced_feature)
                sample_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        baseline_X = np.array(baseline_features_list)
        enhanced_X = np.array(enhanced_features_list)
        y = np.array(sample_labels)
        
        print(f"✅ Processed {len(y)} samples")
        print(f"   Baseline shape: {baseline_X.shape}")
        print(f"   Enhanced shape: {enhanced_X.shape}")
        
        # Load existing best model
        print("\n🔍 Testing with existing best model...")
        try:
            best_model = keras.models.load_model('balanced_multiclass_best.h5', compile=False)
            
            # Test baseline features
            baseline_pred = best_model.predict(baseline_X, verbose=0)
            baseline_accuracy = accuracy_score(y, np.argmax(baseline_pred, axis=1))
            
            print(f"📈 Baseline Model Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
            
            # Analyze per-class performance
            baseline_pred_classes = np.argmax(baseline_pred, axis=1)
            print("\n📊 BASELINE PER-CLASS ACCURACY:")
            for i, class_name in enumerate(self.class_names):
                class_mask = y == i
                if np.sum(class_mask) > 0:
                    class_acc = accuracy_score(y[class_mask], baseline_pred_classes[class_mask])
                    status = '✅' if class_acc >= 0.95 else '❌'
                    print(f"{status} {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
            
        except Exception as e:
            print(f"❌ Could not load model: {e}")
        
        # Analyze spectral characteristics by class
        print(f"\n🎼 SPECTROGRAPHIC ANALYSIS BY CLASS:")
        self._analyze_spectral_differences(enhanced_features_list, sample_labels)
        
        return {
            'baseline_accuracy': baseline_accuracy if 'baseline_accuracy' in locals() else 0,
            'enhanced_features_shape': enhanced_X.shape,
            'baseline_features_shape': baseline_X.shape,
            'samples_processed': len(y)
        }
    
    def _analyze_spectral_differences(self, enhanced_features_list, labels):
        """Analyze spectral differences between classes"""
        
        # Convert to numpy array
        features_array = np.array(enhanced_features_list)
        labels_array = np.array(labels)
        
        # Spectral stats are the last 8 features
        spectral_stats = features_array[:, -8:]
        band_energies = features_array[:, -15:-8]  # 7 band energies before spectral stats
        
        stat_names = ['centroid_mean', 'centroid_std', 'bandwidth_mean', 'bandwidth_std', 
                     'rolloff_mean', 'rolloff_std', 'zcr_mean', 'zcr_std']
        
        band_names = ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'presence', 'brilliance']
        
        print("\n📈 SPECTRAL CHARACTERISTICS BY CLASS:")
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = labels_array == class_idx
            if np.sum(class_mask) == 0:
                continue
                
            print(f"\n🎯 {class_name.upper()}:")
            
            # Spectral statistics
            class_spectral = spectral_stats[class_mask]
            class_bands = band_energies[class_mask]
            
            print("   Spectral Characteristics:")
            for i, stat_name in enumerate(stat_names):
                mean_val = np.mean(class_spectral[:, i])
                print(f"     {stat_name}: {mean_val:.2f}")
            
            print("   Frequency Band Energy Distribution:")
            for i, band_name in enumerate(band_names):
                mean_energy = np.mean(class_bands[:, i])
                print(f"     {band_name}: {mean_energy:.4f}")
        
        # Find most discriminative features
        print(f"\n🔍 MOST DISCRIMINATIVE FEATURES:")
        
        # Calculate variance between classes for each feature
        feature_discriminativeness = []
        
        for feature_idx in range(spectral_stats.shape[1]):
            class_means = []
            for class_idx in range(len(self.class_names)):
                class_mask = labels_array == class_idx
                if np.sum(class_mask) > 0:
                    class_mean = np.mean(spectral_stats[class_mask, feature_idx])
                    class_means.append(class_mean)
            
            if len(class_means) > 1:
                discriminativeness = np.var(class_means)
                feature_discriminativeness.append((stat_names[feature_idx], discriminativeness))
        
        # Sort by discriminativeness
        feature_discriminativeness.sort(key=lambda x: x[1], reverse=True)
        
        print("   Top discriminative spectral features:")
        for feature_name, discriminativeness in feature_discriminativeness[:5]:
            print(f"     {feature_name}: {discriminativeness:.4f}")
        
        # Same for frequency bands
        band_discriminativeness = []
        for band_idx in range(len(band_names)):
            class_means = []
            for class_idx in range(len(self.class_names)):
                class_mask = labels_array == class_idx
                if np.sum(class_mask) > 0:
                    class_mean = np.mean(band_energies[class_mask, band_idx])
                    class_means.append(class_mean)
            
            if len(class_means) > 1:
                discriminativeness = np.var(class_means)
                band_discriminativeness.append((band_names[band_idx], discriminativeness))
        
        band_discriminativeness.sort(key=lambda x: x[1], reverse=True)
        
        print("   Top discriminative frequency bands:")
        for band_name, discriminativeness in band_discriminativeness[:5]:
            print(f"     {band_name}: {discriminativeness:.4f}")

def main():
    print("🎼 SPECTROGRAPHIC ENHANCEMENT RESEARCH")
    print("=" * 70)
    print("🔍 Analyzing advanced frequency domain features")
    
    enhancer = SpectrogramicEnhancement()
    
    # Load sample dataset
    print("\n📊 Loading sample dataset for analysis...")
    dataset_dir = Path("massive_enhanced_dataset")
    
    audio_files = []
    labels = []
    samples_per_class = 100  # Reasonable size for research
    
    for class_idx, class_name in enumerate(enhancer.class_names):
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            np.random.shuffle(files)
            
            for audio_file in files[:samples_per_class]:
                audio_files.append(audio_file)
                labels.append(class_idx)
    
    print(f"Total samples for analysis: {len(audio_files)}")
    
    # Test enhanced features
    results = enhancer.test_enhanced_features(audio_files, labels, num_samples=len(audio_files))
    
    print(f"\n📊 RESEARCH RESULTS:")
    print(f"   Baseline accuracy: {results['baseline_accuracy']*100:.2f}%")
    print(f"   Enhanced features: {results['enhanced_features_shape'][1]:,} dimensions")
    print(f"   Baseline features: {results['baseline_features_shape'][1]:,} dimensions")
    print(f"   Feature expansion: {results['enhanced_features_shape'][1] / results['baseline_features_shape'][1]:.1f}x")
    
    print(f"\n💡 RESEARCH RECOMMENDATIONS:")
    print(f"   1. Incorporate harmonic-percussive separation for better vehicle/aircraft distinction")
    print(f"   2. Use frequency band energy analysis for acoustic signature identification")
    print(f"   3. Apply multi-resolution spectrograms for temporal pattern capture")
    print(f"   4. Implement Constant-Q transform for harmonic analysis")
    print(f"   5. Leverage spectral characteristics for real-time classification")
    
    # Save research results
    with open('spectrographic_research_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Research results saved to spectrographic_research_results.json")

if __name__ == "__main__":
    # Check for scipy dependency
    try:
        import scipy.ndimage
    except ImportError:
        print("Installing scipy for signal processing...")
        os.system("pip install scipy")
    
    main()