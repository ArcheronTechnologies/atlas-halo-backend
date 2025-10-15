#!/usr/bin/env python3
"""
Advanced Spectrographic Model - State-of-the-Art 2024
Incorporates cutting-edge spectral analysis techniques:
- Gammatone filterbanks (auditory-motivated)
- Wavelet scaleograms  
- LFCC, MFCC, CQCC fusion
- Multi-scale attention mechanisms
- Learnable spectral features
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

class AdvancedSpectrogramicModel:
    """State-of-the-art spectrographic analysis with 2024 techniques"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        self.sample_rate = 16000
        
        # Advanced frequency analysis parameters
        self.gammatone_channels = 64  # ERB-spaced gammatone filters
        self.wavelet_scales = 32      # Wavelet decomposition scales
        self.cqt_bins = 84           # Constant-Q bins (7 octaves)
        
    def extract_gammatone_features(self, audio):
        """Extract gammatone filterbank features (auditory-motivated)"""
        try:
            # Implement gammatone filterbank using librosa approximation
            # Gammatone filters approximate auditory processing
            
            # Create ERB-spaced center frequencies (20Hz to 8kHz)
            min_freq = 20
            max_freq = 8000
            
            # ERB formula: ERB(f) = 24.7 * (4.37 * f / 1000 + 1)
            erb_scale = np.linspace(
                24.7 * (4.37 * min_freq / 1000 + 1),
                24.7 * (4.37 * max_freq / 1000 + 1),
                self.gammatone_channels
            )
            
            # Convert back to Hz
            center_freqs = 1000 * ((erb_scale / 24.7) - 1) / 4.37
            
            # Apply bandpass filters (gammatone approximation using Butterworth)
            gammatone_responses = []
            for freq in center_freqs:
                # Bandwidth based on ERB
                bandwidth = 24.7 * (4.37 * freq / 1000 + 1)
                
                # Create bandpass filter
                low_freq = max(freq - bandwidth/2, 20)
                high_freq = min(freq + bandwidth/2, self.sample_rate/2 - 100)
                
                if low_freq < high_freq:
                    sos = scipy.signal.butter(4, [low_freq, high_freq], 
                                            btype='band', fs=self.sample_rate, output='sos')
                    filtered = scipy.signal.sosfilt(sos, audio)
                    
                    # Extract envelope (half-wave rectification + smoothing)
                    envelope = np.maximum(filtered, 0)
                    envelope = scipy.signal.savgol_filter(envelope, 51, 3)
                    gammatone_responses.append(envelope)
                else:
                    gammatone_responses.append(np.zeros_like(audio))
            
            # Convert to time-frequency representation
            gammatone_matrix = np.array(gammatone_responses)
            
            # Downsample time dimension to manageable size
            target_time_frames = 128
            if gammatone_matrix.shape[1] > target_time_frames:
                indices = np.linspace(0, gammatone_matrix.shape[1]-1, target_time_frames, dtype=int)
                gammatone_matrix = gammatone_matrix[:, indices]
            elif gammatone_matrix.shape[1] < target_time_frames:
                pad_width = ((0, 0), (0, target_time_frames - gammatone_matrix.shape[1]))
                gammatone_matrix = np.pad(gammatone_matrix, pad_width, mode='constant')
            
            return gammatone_matrix
            
        except Exception as e:
            print(f"Gammatone extraction error: {e}")
            return np.zeros((self.gammatone_channels, 128))
    
    def extract_wavelet_scaleogram(self, audio):
        """Extract wavelet scaleogram (time-scale representation)"""
        try:
            # Use continuous wavelet transform with Morlet wavelet
            # Morlet wavelet is good for audio analysis
            
            # Define scales (corresponding to different frequencies)
            scales = np.logspace(1, 5, self.wavelet_scales, base=2)  # Logarithmic spacing
            
            # Perform CWT
            coefficients, frequencies = scipy.signal.cwt(audio, scipy.signal.morlet2, scales, 
                                                       w=6)  # w parameter controls wavelet width
            
            # Take magnitude
            scaleogram = np.abs(coefficients)
            
            # Convert to dB scale
            scaleogram_db = 20 * np.log10(scaleogram + 1e-10)
            
            # Resize to standard shape
            target_shape = (32, 128)  # (scales, time)
            scaleogram_resized = self._resize_spectrogram(scaleogram_db, target_shape)
            
            return scaleogram_resized
            
        except Exception as e:
            print(f"Wavelet scaleogram error: {e}")
            return np.zeros((32, 128))
    
    def extract_multi_cepstral_features(self, audio):
        """Extract LFCC, MFCC, and CQCC features"""
        features = {}
        
        try:
            # 1. Linear Frequency Cepstral Coefficients (LFCC)
            # Better for spoofing detection
            stft = librosa.stft(audio, n_fft=512, hop_length=256)
            magnitude = np.abs(stft)
            
            # Linear frequency filterbank
            n_filters = 20
            linear_filters = np.linspace(0, self.sample_rate/2, n_filters+2)
            filter_banks = []
            
            for i in range(n_filters):
                fbank = np.zeros(magnitude.shape[0])
                start_idx = int(linear_filters[i] * magnitude.shape[0] / (self.sample_rate/2))
                mid_idx = int(linear_filters[i+1] * magnitude.shape[0] / (self.sample_rate/2))
                end_idx = int(linear_filters[i+2] * magnitude.shape[0] / (self.sample_rate/2))
                
                if start_idx < end_idx and mid_idx < magnitude.shape[0]:
                    # Triangular filter
                    for j in range(start_idx, min(mid_idx, magnitude.shape[0])):
                        if mid_idx > start_idx:
                            fbank[j] = (j - start_idx) / (mid_idx - start_idx)
                    for j in range(mid_idx, min(end_idx, magnitude.shape[0])):
                        if end_idx > mid_idx:
                            fbank[j] = (end_idx - j) / (end_idx - mid_idx)
                
                filter_banks.append(fbank)
            
            # Apply filterbank
            filter_banks = np.array(filter_banks)
            filtered_magnitude = np.dot(filter_banks, magnitude)
            
            # Log and DCT
            log_magnitude = np.log(filtered_magnitude + 1e-10)
            lfcc = scipy.fftpack.dct(log_magnitude, axis=0)[:13]  # Take first 13 coefficients
            
            # Resize to standard shape
            lfcc_resized = self._resize_spectrogram(lfcc, (13, 128))
            features['lfcc'] = lfcc_resized
            
        except Exception as e:
            print(f"LFCC extraction error: {e}")
            features['lfcc'] = np.zeros((13, 128))
        
        try:
            # 2. Mel Frequency Cepstral Coefficients (MFCC) - enhanced
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, 
                                      n_mels=40, fmin=20, fmax=8000)
            mfcc_resized = self._resize_spectrogram(mfcc, (13, 128))
            features['mfcc'] = mfcc_resized
            
            # Delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            features['mfcc_delta'] = self._resize_spectrogram(mfcc_delta, (13, 128))
            features['mfcc_delta2'] = self._resize_spectrogram(mfcc_delta2, (13, 128))
            
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            features['mfcc'] = np.zeros((13, 128))
            features['mfcc_delta'] = np.zeros((13, 128))
            features['mfcc_delta2'] = np.zeros((13, 128))
        
        try:
            # 3. Constant-Q Cepstral Coefficients (CQCC)
            cqt = librosa.cqt(y=audio, sr=self.sample_rate, n_bins=self.cqt_bins, 
                            bins_per_octave=12, fmin=librosa.note_to_hz('C1'))
            cqt_magnitude = np.abs(cqt)
            
            # Apply mel-scale filtering to CQT
            mel_filters = librosa.filters.mel(sr=self.sample_rate, n_fft=512, n_mels=20,
                                            fmin=20, fmax=8000)
            
            # Adapt mel filters for CQT frequency bins
            cqt_freqs = librosa.cqt_frequencies(n_bins=self.cqt_bins, fmin=librosa.note_to_hz('C1'),
                                              bins_per_octave=12)
            
            # Resample mel filters to match CQT frequencies
            mel_cqt = np.interp(cqt_freqs, 
                               np.linspace(20, 8000, mel_filters.shape[1]), 
                               mel_filters.mean(axis=0))
            mel_cqt = mel_cqt.reshape(1, -1)
            
            # Apply filtering
            filtered_cqt = np.dot(mel_cqt, cqt_magnitude)
            
            # Log and DCT
            log_cqt = np.log(filtered_cqt + 1e-10)
            cqcc = scipy.fftpack.dct(log_cqt, axis=0)[:13]
            
            # Resize to standard shape
            cqcc_resized = self._resize_spectrogram(cqcc, (13, 128))
            features['cqcc'] = cqcc_resized
            
        except Exception as e:
            print(f"CQCC extraction error: {e}")
            features['cqcc'] = np.zeros((13, 128))
        
        return features
    
    def extract_learnable_spectral_features(self, audio):
        """Extract features optimized for deep learning"""
        features = {}
        
        # Multi-resolution mel spectrograms
        window_sizes = [512, 1024, 2048]
        for i, n_fft in enumerate(window_sizes):
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_fft=n_fft, 
                hop_length=n_fft//4, n_mels=64, fmin=20, fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec)
            mel_spec_resized = self._resize_spectrogram(mel_spec_db, (64, 128))
            features[f'mel_resolution_{i}'] = mel_spec_resized
        
        # Log power spectrogram
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        log_power = np.log(np.abs(stft)**2 + 1e-10)
        log_power_resized = self._resize_spectrogram(log_power, (513, 128))
        features['log_power_spectrogram'] = log_power_resized
        
        return features
    
    def _resize_spectrogram(self, spectrogram, target_shape):
        """Resize spectrogram to target shape using scipy zoom"""
        if spectrogram.shape == target_shape:
            return spectrogram
        
        zoom_factors = (target_shape[0] / spectrogram.shape[0], 
                       target_shape[1] / spectrogram.shape[1])
        
        try:
            import scipy.ndimage
            return scipy.ndimage.zoom(spectrogram, zoom_factors, order=1)
        except:
            return np.resize(spectrogram, target_shape)
    
    def extract_all_advanced_features(self, audio):
        """Extract comprehensive advanced spectral features"""
        features = {}
        
        # 1. Gammatone filterbank features
        gammatone = self.extract_gammatone_features(audio)
        features['gammatone_filterbank'] = gammatone
        
        # 2. Wavelet scaleogram
        scaleogram = self.extract_wavelet_scaleogram(audio)
        features['wavelet_scaleogram'] = scaleogram
        
        # 3. Multi-cepstral features
        cepstral_features = self.extract_multi_cepstral_features(audio)
        features.update(cepstral_features)
        
        # 4. Learnable spectral features
        learnable_features = self.extract_learnable_spectral_features(audio)
        features.update(learnable_features)
        
        # 5. Enhanced baseline features
        mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.squeeze(-1)  # Remove channel dimension
        features['baseline_mel'] = self._resize_spectrogram(mel_spec, (64, 128))
        
        return features
    
    def create_advanced_neural_architecture(self):
        """Create state-of-the-art neural architecture with attention mechanisms"""
        
        inputs = {}
        
        # Define inputs for each feature type
        inputs['gammatone_filterbank'] = layers.Input(shape=(64, 128, 1), name='gammatone_input')
        inputs['wavelet_scaleogram'] = layers.Input(shape=(32, 128, 1), name='wavelet_input')
        inputs['lfcc'] = layers.Input(shape=(13, 128, 1), name='lfcc_input')
        inputs['mfcc'] = layers.Input(shape=(13, 128, 1), name='mfcc_input')
        inputs['mfcc_delta'] = layers.Input(shape=(13, 128, 1), name='mfcc_delta_input')
        inputs['mfcc_delta2'] = layers.Input(shape=(13, 128, 1), name='mfcc_delta2_input')
        inputs['cqcc'] = layers.Input(shape=(13, 128, 1), name='cqcc_input')
        inputs['baseline_mel'] = layers.Input(shape=(64, 128, 1), name='baseline_mel_input')
        
        # Mel spectrograms at different resolutions
        for i in range(3):
            inputs[f'mel_resolution_{i}'] = layers.Input(shape=(64, 128, 1), 
                                                        name=f'mel_res_{i}_input')
        
        inputs['log_power_spectrogram'] = layers.Input(shape=(513, 128, 1), 
                                                      name='log_power_input')
        
        # Feature extraction branches with attention
        
        def create_attention_cnn_branch(input_tensor, filters, name_prefix):
            """Create CNN branch with self-attention mechanism"""
            x = input_tensor
            
            # Multi-scale convolutions
            conv_3x3 = layers.Conv2D(filters//2, (3, 3), padding='same', activation='relu')(x)
            conv_5x5 = layers.Conv2D(filters//4, (5, 5), padding='same', activation='relu')(x)
            conv_1x1 = layers.Conv2D(filters//4, (1, 1), padding='same', activation='relu')(x)
            
            # Concatenate multi-scale features
            x = layers.Concatenate()([conv_3x3, conv_5x5, conv_1x1])
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Second layer
            x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # Self-attention mechanism
            attention_weights = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
            x = layers.Multiply()([x, attention_weights])
            
            x = layers.GlobalAveragePooling2D()(x)
            return x
        
        # Process each input through attention CNN
        feature_branches = []
        
        # Gammatone branch (auditory-motivated)
        gammatone_features = create_attention_cnn_branch(
            inputs['gammatone_filterbank'], 128, 'gammatone')
        feature_branches.append(gammatone_features)
        
        # Wavelet branch (time-scale analysis)
        wavelet_features = create_attention_cnn_branch(
            inputs['wavelet_scaleogram'], 96, 'wavelet')
        feature_branches.append(wavelet_features)
        
        # Cepstral feature branches
        for feature_name in ['lfcc', 'mfcc', 'mfcc_delta', 'mfcc_delta2', 'cqcc']:
            cep_features = create_attention_cnn_branch(inputs[feature_name], 64, feature_name)
            feature_branches.append(cep_features)
        
        # Mel spectrogram branches
        for feature_name in ['baseline_mel'] + [f'mel_resolution_{i}' for i in range(3)]:
            mel_features = create_attention_cnn_branch(inputs[feature_name], 128, feature_name)
            feature_branches.append(mel_features)
        
        # Log power spectrogram branch (high resolution)
        log_power_features = create_attention_cnn_branch(
            inputs['log_power_spectrogram'], 256, 'log_power')
        feature_branches.append(log_power_features)
        
        # Cross-modal attention fusion
        concatenated_features = layers.Concatenate()(feature_branches)
        
        # Global attention mechanism
        attention_dense = layers.Dense(len(feature_branches), activation='softmax', 
                                     name='global_attention')(concatenated_features)
        attention_reshaped = layers.Reshape((len(feature_branches), 1))(attention_dense)
        
        # Apply attention weights to each branch
        weighted_features = []
        start_idx = 0
        for i, branch in enumerate(feature_branches):
            branch_size = branch.shape[-1]
            branch_weight = layers.Lambda(lambda x: x[:, i:i+1])(attention_dense)
            weighted_branch = layers.Multiply()([branch, 
                                               layers.Reshape((branch_size,))(
                                                   layers.Dense(branch_size)(branch_weight))])
            weighted_features.append(weighted_branch)
        
        # Final fusion
        fused_features = layers.Add()(weighted_features)
        
        # Classification head with residual connections
        x = layers.Dense(1024, activation='relu')(fused_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Residual connection
        residual = layers.Dense(1024, activation='linear')(fused_features)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(3, activation='softmax', name='classification')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, 
                           name='advanced_spectrographic_model_2024')
        
        return model
    
    def prepare_training_data(self, audio_files, labels, max_samples=1000):
        """Prepare advanced training data with all features"""
        print("🔬 Extracting state-of-the-art spectral features...")
        
        # Initialize feature containers
        all_features = {
            'gammatone_filterbank': [],
            'wavelet_scaleogram': [],
            'lfcc': [],
            'mfcc': [],
            'mfcc_delta': [],
            'mfcc_delta2': [],
            'cqcc': [],
            'baseline_mel': [],
            'mel_resolution_0': [],
            'mel_resolution_1': [],
            'mel_resolution_2': [],
            'log_power_spectrogram': []
        }
        
        processed_labels = []
        
        # Process audio files
        for i, (audio_file, label) in enumerate(zip(audio_files[:max_samples], 
                                                   labels[:max_samples])):
            if i % 50 == 0:
                print(f"   Processing {i}/{min(len(audio_files), max_samples)} samples...")
            
            try:
                # Load audio
                audio = self.preprocessor.load_and_resample(audio_file)
                
                # Extract all advanced features
                features = self.extract_all_advanced_features(audio)
                
                # Store features (add channel dimension where needed)
                for key, value in features.items():
                    if key in all_features:
                        if len(value.shape) == 2:
                            value = np.expand_dims(value, axis=-1)
                        all_features[key].append(value)
                
                processed_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        # Convert to numpy arrays
        processed_features = {}
        for key, feature_list in all_features.items():
            if feature_list:
                processed_features[key] = np.array(feature_list)
                print(f"   {key}: {processed_features[key].shape}")
        
        return processed_features, np.array(processed_labels)

def main():
    print("🚀 ADVANCED SPECTROGRAPHIC MODEL 2024")
    print("=" * 80)
    print("🎯 State-of-the-art spectral analysis with:")
    print("   • Gammatone filterbanks (auditory-motivated)")
    print("   • Wavelet scaleograms (time-scale analysis)")  
    print("   • Multi-cepstral fusion (LFCC+MFCC+CQCC)")
    print("   • Learnable spectral features")
    print("   • Cross-modal attention mechanisms")
    
    model_builder = AdvancedSpectrogramicModel()
    
    # Load dataset
    print("\n📊 Loading enhanced dataset...")
    dataset_dir = Path("massive_enhanced_dataset")
    audio_files = []
    labels = []
    
    samples_per_class = 600  # Manageable size for comprehensive feature extraction
    
    for class_idx, class_name in enumerate(model_builder.class_names):
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            np.random.shuffle(files)
            
            for audio_file in files[:samples_per_class]:
                audio_files.append(audio_file)
                labels.append(class_idx)
    
    print(f"Total samples: {len(audio_files)}")
    
    # Extract advanced features
    X_features, y = model_builder.prepare_training_data(audio_files, labels)
    
    # Create advanced model
    print("\n🏗️ Building advanced neural architecture...")
    model = model_builder.create_advanced_neural_architecture()
    print(f"Model parameters: {model.count_params():,}")
    
    # Compile with state-of-the-art optimizer
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=0.0002, 
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model configuration
    with open('advanced_spectrographic_config.json', 'w') as f:
        config = {
            'model_name': 'advanced_spectrographic_model_2024',
            'features': list(X_features.keys()),
            'parameters': int(model.count_params()),
            'techniques': [
                'Gammatone filterbanks',
                'Wavelet scaleograms', 
                'Multi-cepstral fusion (LFCC+MFCC+CQCC)',
                'Cross-modal attention',
                'Residual connections'
            ]
        }
        json.dump(config, f, indent=2)
    
    print("💾 Model configuration saved: advanced_spectrographic_config.json")
    print("🔧 Model architecture ready for training!")
    
    return model, X_features, y

if __name__ == "__main__":
    # Check dependencies
    required_packages = ['scipy', 'librosa', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    model, X_features, y = main()