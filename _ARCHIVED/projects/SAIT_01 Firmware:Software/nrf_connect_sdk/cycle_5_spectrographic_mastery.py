#!/usr/bin/env python3
"""
Cycle 5: Spectrographic Mastery - Ultimate 95%+ System
Advanced frequency domain analysis, harmonic fingerprinting, and multi-modal fusion
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import json
import librosa
import scipy.signal
import scipy.fftpack
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class SpectrogramicMastery:
    """State-of-the-art spectrographic analysis for ultimate accuracy"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        self.sample_rate = 16000
        
        # Frequency bands for different acoustic signatures
        self.freq_bands = {
            'sub_bass': (20, 60),      # Very low frequency rumbles
            'bass': (60, 250),         # Engine fundamentals
            'low_mid': (250, 500),     # Engine harmonics  
            'mid': (500, 2000),        # Voice/mechanical sounds
            'high_mid': (2000, 4000),  # Tire/road noise
            'presence': (4000, 6000),  # Rotor blade tips
            'brilliance': (6000, 20000) # High frequency harmonics
        }
        
    def extract_spectrographic_features(self, audio):
        """Extract comprehensive spectrographic features"""
        features = {}
        
        # 1. Multi-resolution spectrograms
        features.update(self._multi_resolution_spectrograms(audio))
        
        # 2. Harmonic-percussive separation
        features.update(self._harmonic_percussive_analysis(audio))
        
        # 3. Chromagram for harmonic content
        features.update(self._chromagram_analysis(audio))
        
        # 4. Spectral centroids and rolloff
        features.update(self._spectral_characteristics(audio))
        
        # 5. Constant-Q transform (logarithmic frequency)
        features.update(self._constant_q_analysis(audio))
        
        # 6. Mel-frequency cepstral coefficients (MFCC)
        features.update(self._mfcc_analysis(audio))
        
        # 7. Frequency band energy analysis
        features.update(self._frequency_band_analysis(audio))
        
        # 8. Acoustic fingerprinting
        features.update(self._acoustic_fingerprinting(audio))
        
        return features
    
    def _multi_resolution_spectrograms(self, audio):
        """Generate spectrograms at multiple time-frequency resolutions"""
        spectrograms = {}
        
        # Short-time Fourier transform with different window sizes
        window_sizes = [512, 1024, 2048, 4096]  # Different temporal resolutions
        hop_lengths = [128, 256, 512, 1024]     # Different overlap amounts
        
        for i, (n_fft, hop_length) in enumerate(zip(window_sizes, hop_lengths)):
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Convert to dB scale
            magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # Resize to standard shape for neural network
            if magnitude_db.shape[1] > 128:
                # Downsample time axis
                indices = np.linspace(0, magnitude_db.shape[1]-1, 128, dtype=int)
                magnitude_db = magnitude_db[:, indices]
            elif magnitude_db.shape[1] < 128:
                # Zero-pad time axis  
                pad_width = ((0, 0), (0, 128 - magnitude_db.shape[1]))
                magnitude_db = np.pad(magnitude_db, pad_width, mode='constant')
            
            # Resize frequency axis to 64 bins
            if magnitude_db.shape[0] > 64:
                indices = np.linspace(0, magnitude_db.shape[0]-1, 64, dtype=int)
                magnitude_db = magnitude_db[indices, :]
            elif magnitude_db.shape[0] < 64:
                pad_width = ((0, 64 - magnitude_db.shape[0]), (0, 0))
                magnitude_db = np.pad(magnitude_db, pad_width, mode='constant')
            
            spectrograms[f'spectrogram_res_{i}'] = magnitude_db
        
        return spectrograms
    
    def _harmonic_percussive_analysis(self, audio):
        """Separate harmonic and percussive components"""
        # Harmonic-percussive source separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Generate spectrograms for each component
        harmonic_spec = np.abs(librosa.stft(harmonic))
        percussive_spec = np.abs(librosa.stft(percussive))
        
        # Convert to mel scale
        harmonic_mel = librosa.feature.melspectrogram(S=harmonic_spec**2, sr=self.sample_rate)
        percussive_mel = librosa.feature.melspectrogram(S=percussive_spec**2, sr=self.sample_rate)
        
        # Resize to standard shape
        harmonic_mel = self._resize_spectrogram(harmonic_mel, (64, 128))
        percussive_mel = self._resize_spectrogram(percussive_mel, (64, 128))
        
        return {
            'harmonic_spectrogram': librosa.amplitude_to_db(harmonic_mel),
            'percussive_spectrogram': librosa.amplitude_to_db(percussive_mel)
        }
    
    def _chromagram_analysis(self, audio):
        """Extract chromagram for harmonic content analysis"""
        # Chromagram captures harmonic content across octaves
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        
        # Resize to standard shape
        chroma = self._resize_spectrogram(chroma, (12, 128))  # 12 chromatic pitches
        
        return {'chromagram': chroma}
    
    def _spectral_characteristics(self, audio):
        """Extract spectral centroids, bandwidth, rolloff"""
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # Resize all to standard temporal resolution
        features = {}
        for name, feature in [('spectral_centroid', centroid), 
                             ('spectral_bandwidth', bandwidth),
                             ('spectral_rolloff', rolloff),
                             ('zero_crossing_rate', zcr)]:
            resized = self._resize_temporal_feature(feature, 128)
            features[name] = resized.reshape(1, -1)  # Shape: (1, 128)
        
        return features
    
    def _constant_q_analysis(self, audio):
        """Constant-Q transform for logarithmic frequency analysis"""
        # Constant-Q transform (better for harmonic analysis)
        cqt = librosa.cqt(y=audio, sr=self.sample_rate, hop_length=256, n_bins=64)
        cqt_magnitude = np.abs(cqt)
        cqt_db = librosa.amplitude_to_db(cqt_magnitude)
        
        # Resize to standard shape
        cqt_db = self._resize_spectrogram(cqt_db, (64, 128))
        
        return {'constant_q_transform': cqt_db}
    
    def _mfcc_analysis(self, audio):
        """Mel-frequency cepstral coefficients"""
        # MFCC features (commonly used in speech recognition)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Delta (velocity) and delta-delta (acceleration) features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Resize all to standard shape
        mfcc = self._resize_spectrogram(mfcc, (13, 128))
        mfcc_delta = self._resize_spectrogram(mfcc_delta, (13, 128))
        mfcc_delta2 = self._resize_spectrogram(mfcc_delta2, (13, 128))
        
        return {
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2
        }
    
    def _frequency_band_analysis(self, audio):
        """Analyze energy in specific frequency bands"""
        # Compute FFT
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        band_energies = {}
        
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find frequency indices
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            # Calculate energy in this band
            band_energy = np.sum(magnitude[band_mask]**2)
            band_energies[f'energy_{band_name}'] = np.array([[band_energy]])  # Shape: (1, 1)
        
        return band_energies
    
    def _acoustic_fingerprinting(self, audio):
        """Create acoustic fingerprints for each class type"""
        fingerprints = {}
        
        # Peak finding in frequency domain
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Find prominent peaks
        peaks, _ = scipy.signal.find_peaks(magnitude, height=np.percentile(magnitude, 90))
        
        # Extract peak frequencies and magnitudes
        if len(peaks) > 0:
            peak_freqs = freqs[peaks]
            peak_magnitudes = magnitude[peaks]
            
            # Sort by magnitude and take top 10
            sorted_indices = np.argsort(peak_magnitudes)[::-1][:10]
            top_peak_freqs = peak_freqs[sorted_indices]
            top_peak_mags = peak_magnitudes[sorted_indices]
            
            # Pad to exactly 10 features
            if len(top_peak_freqs) < 10:
                top_peak_freqs = np.pad(top_peak_freqs, (0, 10 - len(top_peak_freqs)))
                top_peak_mags = np.pad(top_peak_mags, (0, 10 - len(top_peak_mags)))
            
            fingerprints['peak_frequencies'] = top_peak_freqs.reshape(1, -1)
            fingerprints['peak_magnitudes'] = top_peak_mags.reshape(1, -1)
        else:
            fingerprints['peak_frequencies'] = np.zeros((1, 10))
            fingerprints['peak_magnitudes'] = np.zeros((1, 10))
        
        return fingerprints
    
    def _resize_spectrogram(self, spectrogram, target_shape):
        """Resize spectrogram to target shape"""
        if spectrogram.shape == target_shape:
            return spectrogram
        
        # Use scipy's zoom for resizing
        zoom_factors = (target_shape[0] / spectrogram.shape[0], 
                       target_shape[1] / spectrogram.shape[1])
        
        return scipy.ndimage.zoom(spectrogram, zoom_factors, order=1)
    
    def _resize_temporal_feature(self, feature, target_length):
        """Resize temporal feature to target length"""
        if feature.shape[1] == target_length:
            return feature[0]  # Remove first dimension
        
        # Linear interpolation for resizing
        x_old = np.linspace(0, 1, feature.shape[1])
        x_new = np.linspace(0, 1, target_length)
        
        return np.interp(x_new, x_old, feature[0])
    
    def create_ultimate_model(self):
        """Create the ultimate multi-modal neural architecture"""
        
        # Input layers for different feature types
        inputs = {}
        
        # Spectrogram inputs (multiple resolutions)
        for i in range(4):
            inputs[f'spec_res_{i}'] = layers.Input(shape=(64, 128, 1), name=f'spectrogram_res_{i}')
        
        # Harmonic/percussive inputs
        inputs['harmonic'] = layers.Input(shape=(64, 128, 1), name='harmonic_spectrogram')
        inputs['percussive'] = layers.Input(shape=(64, 128, 1), name='percussive_spectrogram')
        
        # Chromagram input
        inputs['chroma'] = layers.Input(shape=(12, 128, 1), name='chromagram')
        
        # Constant-Q input
        inputs['cqt'] = layers.Input(shape=(64, 128, 1), name='constant_q_transform')
        
        # MFCC inputs
        inputs['mfcc'] = layers.Input(shape=(13, 128, 1), name='mfcc')
        inputs['mfcc_delta'] = layers.Input(shape=(13, 128, 1), name='mfcc_delta')
        inputs['mfcc_delta2'] = layers.Input(shape=(13, 128, 1), name='mfcc_delta2')
        
        # Spectral characteristic inputs
        inputs['centroid'] = layers.Input(shape=(1, 128), name='spectral_centroid')
        inputs['bandwidth'] = layers.Input(shape=(1, 128), name='spectral_bandwidth')
        inputs['rolloff'] = layers.Input(shape=(1, 128), name='spectral_rolloff')
        inputs['zcr'] = layers.Input(shape=(1, 128), name='zero_crossing_rate')
        
        # Frequency band energy inputs
        for band_name in self.freq_bands.keys():
            inputs[f'energy_{band_name}'] = layers.Input(shape=(1, 1), name=f'energy_{band_name}')
        
        # Acoustic fingerprint inputs
        inputs['peak_freqs'] = layers.Input(shape=(1, 10), name='peak_frequencies')
        inputs['peak_mags'] = layers.Input(shape=(1, 10), name='peak_magnitudes')
        
        # Processing branches for different input types
        
        # 1. Spectrogram processing branch (CNN with attention)
        spectrogram_features = []
        for key in ['spec_res_0', 'spec_res_1', 'spec_res_2', 'spec_res_3', 
                   'harmonic', 'percussive', 'cqt']:
            x = inputs[key]
            
            # Multi-scale CNN with attention
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Attention mechanism
            attention = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
            x = layers.Multiply()([x, attention])
            
            x = layers.GlobalAveragePooling2D()(x)
            spectrogram_features.append(x)
        
        # 2. MFCC processing branch
        mfcc_features = []
        for key in ['mfcc', 'mfcc_delta', 'mfcc_delta2']:
            x = inputs[key]
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling2D()(x)
            mfcc_features.append(x)
        
        # 3. Chromagram processing
        x = inputs['chroma']
        x = layers.Conv2D(24, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        chroma_features = [x]
        
        # 4. Spectral characteristics processing
        spectral_features = []
        for key in ['centroid', 'bandwidth', 'rolloff', 'zcr']:
            x = inputs[key]
            x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
            x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling1D()(x)
            spectral_features.append(x)
        
        # 5. Energy band processing
        energy_features = []
        for band_name in self.freq_bands.keys():
            x = inputs[f'energy_{band_name}']
            x = layers.Flatten()(x)
            x = layers.Dense(8, activation='relu')(x)
            energy_features.append(x)
        
        # 6. Acoustic fingerprint processing  
        peak_freq_branch = layers.Flatten()(inputs['peak_freqs'])
        peak_freq_branch = layers.Dense(32, activation='relu')(peak_freq_branch)
        
        peak_mag_branch = layers.Flatten()(inputs['peak_mags'])
        peak_mag_branch = layers.Dense(32, activation='relu')(peak_mag_branch)
        
        fingerprint_features = [peak_freq_branch, peak_mag_branch]
        
        # Fusion layer - combine all feature types
        all_features = (spectrogram_features + mfcc_features + chroma_features + 
                       spectral_features + energy_features + fingerprint_features)
        
        fused = layers.Concatenate()(all_features)
        
        # Final classification layers with heavy regularization
        x = layers.Dense(1024, activation='relu')(fused)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.6)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Output layer
        outputs = layers.Dense(3, activation='softmax', name='classification')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='spectrographic_mastery')
        
        return model
    
    def prepare_multi_modal_data(self, audio_files, labels):
        """Prepare multi-modal dataset"""
        print("🎼 Extracting comprehensive spectrographic features...")
        
        # Initialize feature containers
        all_features = {key: [] for key in [
            'spectrogram_res_0', 'spectrogram_res_1', 'spectrogram_res_2', 'spectrogram_res_3',
            'harmonic_spectrogram', 'percussive_spectrogram', 'chromagram', 'constant_q_transform',
            'mfcc', 'mfcc_delta', 'mfcc_delta2',
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate',
            'peak_frequencies', 'peak_magnitudes'
        ]}
        
        # Add frequency band keys
        for band_name in self.freq_bands.keys():
            all_features[f'energy_{band_name}'] = []
        
        # Process each audio file
        for i, audio_file in enumerate(audio_files):
            if i % 100 == 0:
                print(f"   Processing {i}/{len(audio_files)} samples...")
            
            try:
                # Load audio
                audio = self.preprocessor.load_and_resample(audio_file)
                
                # Extract all features
                features = self.extract_spectrographic_features(audio)
                
                # Store features
                for key, value in features.items():
                    if len(value.shape) == 2:
                        # Add channel dimension for 2D features
                        value = np.expand_dims(value, axis=-1)
                    all_features[key].append(value)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                # Skip this sample
                continue
        
        # Convert to numpy arrays
        processed_features = {}
        for key, feature_list in all_features.items():
            if feature_list:  # If we have features for this key
                processed_features[key] = np.array(feature_list)
                print(f"   {key}: {processed_features[key].shape}")
        
        return processed_features, np.array(labels[:len(feature_list)])

def main():
    print("🚀 CYCLE 5: SPECTROGRAPHIC MASTERY")
    print("=" * 80)
    print("🎼 Ultimate frequency domain analysis for 95%+ accuracy")
    
    mastery = SpectrogramicMastery()
    
    # Load dataset
    print("\n📊 Loading dataset...")
    dataset_dir = Path("massive_enhanced_dataset")
    audio_files = []
    labels = []
    
    samples_per_class = 800  # Reasonable size for comprehensive feature extraction
    
    for class_idx, class_name in enumerate(mastery.class_names):
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            np.random.shuffle(files)
            
            for audio_file in files[:samples_per_class]:
                audio_files.append(audio_file)
                labels.append(class_idx)
    
    print(f"Total samples: {len(audio_files)}")
    
    # Extract multi-modal features
    X_features, y = mastery.prepare_multi_modal_data(audio_files, labels)
    
    # Create ultimate model
    print("\n🏗️ Building ultimate multi-modal architecture...")
    model = mastery.create_ultimate_model()
    print(f"Model parameters: {model.count_params():,}")
    
    # Compile with advanced optimization
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.0003, weight_decay=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train/validation split
    print("\n🎯 Training spectrographic mastery model...")
    train_indices, val_indices = train_test_split(
        range(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = {key: features[train_indices] for key, features in X_features.items()}
    X_val = {key: features[val_indices] for key, features in X_features.items()}
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    # Advanced callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'cycle_5_spectrographic_best.h5', 
            monitor='val_accuracy', 
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save("sait01_cycle_5_spectrographic_mastery.h5")
    print("💾 Saved: sait01_cycle_5_spectrographic_mastery.h5")
    
    # Final validation
    print("\n✅ Final validation...")
    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(mastery.class_names):
        class_mask = y_val == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_val[class_mask], y_pred[class_mask])
            class_accuracies[class_name] = class_acc
            status = '✅' if class_acc >= 0.95 else '❌'
            print(f"{status} {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Check 95% target
    meets_target = all(acc >= 0.95 for acc in class_accuracies.values()) and accuracy >= 0.95
    print(f"\n🎯 95% TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
    
    if meets_target:
        print("🎉 SPECTROGRAPHIC MASTERY ACHIEVED 95% TARGET!")
    
    # Save results
    results = {
        "cycle": 5,
        "approach": "Spectrographic Mastery",
        "overall_accuracy": float(accuracy),
        "class_accuracies": {k: float(v) for k, v in class_accuracies.items()},
        "meets_95_target": meets_target,
        "model_path": "sait01_cycle_5_spectrographic_mastery.h5",
        "features_used": list(X_features.keys()),
        "model_parameters": int(model.count_params())
    }
    
    with open("cycle_5_spectrographic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Check dependencies
    try:
        import scipy.ndimage
    except ImportError:
        print("Installing scipy for advanced signal processing...")
        os.system("pip install scipy")
        import scipy.ndimage
    
    main()