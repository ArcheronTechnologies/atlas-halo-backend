#!/usr/bin/env python3
"""
Focused Spectrographic Model - Top 4 Most Promising Techniques
Based on 2024 research, focusing on the most effective methods:
1. Gammatone filterbanks (auditory-motivated, proven effective)
2. Multi-resolution mel spectrograms (different temporal resolutions)
3. MFCC with delta features (industry standard, highly effective)
4. Harmonic-percussive separation (great for vehicle/aircraft distinction)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class FocusedSpectrogramicModel:
    """Focused model with 4 most promising techniques"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        self.sample_rate = 16000
        self.gammatone_channels = 32  # Reduced for efficiency
        
    def extract_gammatone_features(self, audio):
        """Extract gammatone filterbank features (auditory-motivated)"""
        try:
            # ERB-spaced frequencies (20Hz to 8kHz)
            min_freq, max_freq = 20, 8000
            erb_scale = np.linspace(
                24.7 * (4.37 * min_freq / 1000 + 1),
                24.7 * (4.37 * max_freq / 1000 + 1),
                self.gammatone_channels
            )
            center_freqs = 1000 * ((erb_scale / 24.7) - 1) / 4.37
            
            # Apply gammatone-approximated filters
            gammatone_responses = []
            for freq in center_freqs:
                bandwidth = 24.7 * (4.37 * freq / 1000 + 1)
                low_freq = max(freq - bandwidth/2, 20)
                high_freq = min(freq + bandwidth/2, self.sample_rate/2 - 100)
                
                if low_freq < high_freq:
                    sos = scipy.signal.butter(4, [low_freq, high_freq], 
                                            btype='band', fs=self.sample_rate, output='sos')
                    filtered = scipy.signal.sosfilt(sos, audio)
                    envelope = np.maximum(filtered, 0)  # Half-wave rectification
                    gammatone_responses.append(envelope)
                else:
                    gammatone_responses.append(np.zeros_like(audio))
            
            # Convert to time-frequency matrix and downsample
            gammatone_matrix = np.array(gammatone_responses)
            target_time = 128
            if gammatone_matrix.shape[1] > target_time:
                indices = np.linspace(0, gammatone_matrix.shape[1]-1, target_time, dtype=int)
                gammatone_matrix = gammatone_matrix[:, indices]
            elif gammatone_matrix.shape[1] < target_time:
                pad_width = ((0, 0), (0, target_time - gammatone_matrix.shape[1]))
                gammatone_matrix = np.pad(gammatone_matrix, pad_width)
            
            return gammatone_matrix
            
        except Exception as e:
            print(f"Gammatone error: {e}")
            return np.zeros((self.gammatone_channels, 128))
    
    def extract_multiresolution_mel(self, audio):
        """Extract mel spectrograms at 3 different resolutions"""
        mel_features = {}
        
        # Three different window sizes for different temporal resolutions
        window_configs = [
            {'n_fft': 512, 'hop_length': 128, 'name': 'fine'},      # Fine temporal resolution
            {'n_fft': 1024, 'hop_length': 256, 'name': 'medium'},  # Medium resolution 
            {'n_fft': 2048, 'hop_length': 512, 'name': 'coarse'}   # Coarse resolution
        ]
        
        for config in window_configs:
            try:
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, sr=self.sample_rate,
                    n_fft=config['n_fft'], 
                    hop_length=config['hop_length'],
                    n_mels=64, fmin=20, fmax=8000
                )
                mel_spec_db = librosa.power_to_db(mel_spec)
                
                # Resize to standard shape
                mel_resized = self._resize_to_shape(mel_spec_db, (64, 128))
                mel_features[f'mel_{config["name"]}'] = mel_resized
                
            except Exception as e:
                print(f"Mel {config['name']} error: {e}")
                mel_features[f'mel_{config["name"]}'] = np.zeros((64, 128))
        
        return mel_features
    
    def extract_enhanced_mfcc(self, audio):
        """Extract MFCC with delta and delta-delta features"""
        try:
            # Standard MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13,
                                      n_mels=40, fmin=20, fmax=8000)
            
            # Delta (velocity) features
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # Delta-delta (acceleration) features  
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Resize all to standard shape
            mfcc_resized = self._resize_to_shape(mfcc, (13, 128))
            mfcc_delta_resized = self._resize_to_shape(mfcc_delta, (13, 128))
            mfcc_delta2_resized = self._resize_to_shape(mfcc_delta2, (13, 128))
            
            return {
                'mfcc': mfcc_resized,
                'mfcc_delta': mfcc_delta_resized,
                'mfcc_delta2': mfcc_delta2_resized
            }
            
        except Exception as e:
            print(f"MFCC error: {e}")
            return {
                'mfcc': np.zeros((13, 128)),
                'mfcc_delta': np.zeros((13, 128)),
                'mfcc_delta2': np.zeros((13, 128))
            }
    
    def extract_harmonic_percussive_features(self, audio):
        """Harmonic-percussive separation for vehicle/aircraft distinction"""
        try:
            # Separate harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Extract mel spectrograms for each component
            harmonic_mel = librosa.feature.melspectrogram(
                y=harmonic, sr=self.sample_rate, n_mels=64, fmin=20, fmax=8000)
            percussive_mel = librosa.feature.melspectrogram(
                y=percussive, sr=self.sample_rate, n_mels=64, fmin=20, fmax=8000)
            
            # Convert to dB and resize
            harmonic_db = librosa.power_to_db(harmonic_mel)
            percussive_db = librosa.power_to_db(percussive_mel)
            
            harmonic_resized = self._resize_to_shape(harmonic_db, (64, 128))
            percussive_resized = self._resize_to_shape(percussive_db, (64, 128))
            
            return {
                'harmonic_mel': harmonic_resized,
                'percussive_mel': percussive_resized
            }
            
        except Exception as e:
            print(f"Harmonic-percussive error: {e}")
            return {
                'harmonic_mel': np.zeros((64, 128)),
                'percussive_mel': np.zeros((64, 128))
            }
    
    def _resize_to_shape(self, spectrogram, target_shape):
        """Resize spectrogram using scipy zoom"""
        if spectrogram.shape == target_shape:
            return spectrogram
        
        zoom_factors = (target_shape[0] / spectrogram.shape[0], 
                       target_shape[1] / spectrogram.shape[1])
        
        try:
            import scipy.ndimage
            return scipy.ndimage.zoom(spectrogram, zoom_factors, order=1)
        except:
            return np.resize(spectrogram, target_shape)
    
    def extract_focused_features(self, audio):
        """Extract all 4 focused feature types"""
        features = {}
        
        # 1. Gammatone filterbank (auditory-motivated)
        gammatone = self.extract_gammatone_features(audio)
        features['gammatone'] = gammatone
        
        # 2. Multi-resolution mel spectrograms
        mel_features = self.extract_multiresolution_mel(audio)
        features.update(mel_features)
        
        # 3. Enhanced MFCC with deltas
        mfcc_features = self.extract_enhanced_mfcc(audio)
        features.update(mfcc_features)
        
        # 4. Harmonic-percussive separation
        hp_features = self.extract_harmonic_percussive_features(audio)
        features.update(hp_features)
        
        return features
    
    def create_focused_model(self):
        """Create streamlined model with attention mechanisms"""
        
        # Input layers
        inputs = {
            'gammatone': layers.Input(shape=(32, 128, 1), name='gammatone_input'),
            'mel_fine': layers.Input(shape=(64, 128, 1), name='mel_fine_input'),
            'mel_medium': layers.Input(shape=(64, 128, 1), name='mel_medium_input'),
            'mel_coarse': layers.Input(shape=(64, 128, 1), name='mel_coarse_input'),
            'mfcc': layers.Input(shape=(13, 128, 1), name='mfcc_input'),
            'mfcc_delta': layers.Input(shape=(13, 128, 1), name='mfcc_delta_input'),
            'mfcc_delta2': layers.Input(shape=(13, 128, 1), name='mfcc_delta2_input'),
            'harmonic_mel': layers.Input(shape=(64, 128, 1), name='harmonic_input'),
            'percussive_mel': layers.Input(shape=(64, 128, 1), name='percussive_input')
        }
        
        def create_cnn_branch(input_tensor, filters, name_prefix):
            """Create CNN branch with attention"""
            x = input_tensor
            
            # First conv block
            x = layers.Conv2D(filters//2, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Second conv block
            x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # Attention mechanism
            attention = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
            x = layers.Multiply()([x, attention])
            
            # Global pooling
            x = layers.GlobalAveragePooling2D()(x)
            return x
        
        # Process each input type
        feature_branches = []
        
        # Gammatone branch (smaller input, fewer filters)
        gammatone_branch = create_cnn_branch(inputs['gammatone'], 64, 'gammatone')
        feature_branches.append(gammatone_branch)
        
        # Mel spectrogram branches (3 resolutions)
        for mel_type in ['mel_fine', 'mel_medium', 'mel_coarse']:
            mel_branch = create_cnn_branch(inputs[mel_type], 96, mel_type)
            feature_branches.append(mel_branch)
        
        # MFCC branches
        for mfcc_type in ['mfcc', 'mfcc_delta', 'mfcc_delta2']:
            mfcc_branch = create_cnn_branch(inputs[mfcc_type], 48, mfcc_type)
            feature_branches.append(mfcc_branch)
        
        # Harmonic-percussive branches
        for hp_type in ['harmonic_mel', 'percussive_mel']:
            hp_branch = create_cnn_branch(inputs[hp_type], 96, hp_type)
            feature_branches.append(hp_branch)
        
        # Simple concatenation fusion (more stable)
        fused = layers.Concatenate()(feature_branches)
        
        # Classification head
        x = layers.Dense(512, activation='relu')(fused)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(3, activation='softmax', name='classification')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='focused_spectrographic_model')
        
        return model
    
    def prepare_focused_data(self, audio_files, labels, max_samples=1200):
        """Prepare training data with focused features"""
        print("🎯 Extracting focused spectral features...")
        print("   Techniques: Gammatone + Multi-res Mel + Enhanced MFCC + Harmonic-Percussive")
        
        all_features = {
            'gammatone': [],
            'mel_fine': [],
            'mel_medium': [],
            'mel_coarse': [],
            'mfcc': [],
            'mfcc_delta': [],
            'mfcc_delta2': [],
            'harmonic_mel': [],
            'percussive_mel': []
        }
        
        processed_labels = []
        
        for i, (audio_file, label) in enumerate(zip(audio_files[:max_samples], 
                                                   labels[:max_samples])):
            if i % 100 == 0:
                print(f"   Processing {i}/{min(len(audio_files), max_samples)} samples...")
            
            try:
                # Load audio
                audio = self.preprocessor.load_and_resample(audio_file)
                
                # Extract focused features
                features = self.extract_focused_features(audio)
                
                # Store features (add channel dimension)
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

def test_focused_approaches():
    """Test different combinations of the 4 techniques"""
    print("🧪 TESTING FOCUSED SPECTROGRAPHIC APPROACHES")
    print("=" * 70)
    
    model_builder = FocusedSpectrogramicModel()
    
    # Test configurations
    test_configs = [
        {
            'name': 'Gammatone + Multi-Mel',
            'features': ['gammatone', 'mel_fine', 'mel_medium', 'mel_coarse']
        },
        {
            'name': 'MFCC Enhanced + Harmonic-Percussive', 
            'features': ['mfcc', 'mfcc_delta', 'mfcc_delta2', 'harmonic_mel', 'percussive_mel']
        },
        {
            'name': 'All 4 Techniques',
            'features': ['gammatone', 'mel_fine', 'mel_medium', 'mel_coarse', 
                        'mfcc', 'mfcc_delta', 'mfcc_delta2', 'harmonic_mel', 'percussive_mel']
        }
    ]
    
    return test_configs

def main():
    print("🎯 FOCUSED SPECTROGRAPHIC MODEL")
    print("=" * 60)
    print("Top 4 most promising techniques:")
    print("1. 🎧 Gammatone filterbanks (auditory-motivated)")
    print("2. 📊 Multi-resolution mel spectrograms")  
    print("3. 🔢 Enhanced MFCC with deltas")
    print("4. 🎵 Harmonic-percussive separation")
    
    model_builder = FocusedSpectrogramicModel()
    
    # Create model architecture
    print("\n🏗️ Building focused model...")
    model = model_builder.create_focused_model()
    print(f"Model parameters: {model.count_params():,}")
    
    # Save model configuration
    config = {
        'model_name': 'focused_spectrographic_model',
        'techniques': [
            'Gammatone filterbanks (32 channels)',
            'Multi-resolution mel spectrograms (3 resolutions)',
            'Enhanced MFCC with delta features', 
            'Harmonic-percussive separation'
        ],
        'parameters': int(model.count_params()),
        'input_features': 9,
        'efficient_design': True
    }
    
    with open('focused_spectrographic_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("💾 Configuration saved: focused_spectrographic_config.json")
    print("🔧 Model ready for training!")
    
    # Show test configurations
    test_configs = test_focused_approaches()
    print(f"\n🧪 Available test configurations:")
    for i, config in enumerate(test_configs, 1):
        print(f"   {i}. {config['name']}: {len(config['features'])} features")
    
    return model

if __name__ == "__main__":
    model = main()