#!/usr/bin/env python3
"""
SAIT_01 TinyML Model Architecture - DS-CNN+GRU for Edge Audio Classification
Designed for nRF5340 deployment with TensorFlow Lite Micro
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import librosa
import os
from pathlib import Path

# Model configuration matching nRF5340 constraints
MODEL_CONFIG = {
    # Audio parameters (matching sait01_tinyml_integration.h)
    'sample_rate': 16000,           # 16kHz sampling (downsampled from 44.1kHz)
    'window_ms': 1000,              # 1 second analysis window
    'hop_length': 256,              # STFT hop length
    'n_fft': 512,                   # FFT window size
    'n_mels': 64,                   # Mel frequency bins
    'n_frames': 63,                 # Time frames per spectrogram
    
    # Model architecture
    'input_shape': (64, 63, 1),     # Mel spectrogram dimensions + channel
    'num_classes': 8,               # SAIT_01 classification classes
    'model_size_kb': 80,            # Target model size (80KB)
    
    # DS-CNN layers
    'depthwise_filters': 64,
    'pointwise_filters': 64,
    'conv_dropout': 0.2,
    
    # GRU layers
    'gru_units': 32,
    'gru_dropout': 0.3,
    
    # Training parameters
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2
}

class SaitAudioPreprocessor:
    """Audio preprocessing pipeline for SAIT_01 dataset conversion"""
    
    def __init__(self, config=MODEL_CONFIG):
        self.config = config
        
    def load_and_resample(self, audio_path):
        """Load audio file and resample to 16kHz"""
        # Load original audio (44.1kHz)
        audio, orig_sr = librosa.load(audio_path, sr=None)
        
        # Resample to 16kHz for edge deployment
        if orig_sr != self.config['sample_rate']:
            audio = librosa.resample(audio, orig_sr=orig_sr, 
                                   target_sr=self.config['sample_rate'])
        
        return audio
        
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram features"""
        # Ensure consistent length (1 second = 16000 samples)
        target_length = self.config['sample_rate'] * self.config['window_ms'] // 1000
        
        if len(audio) > target_length:
            # Crop to target length
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))
            
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels'],
            fmin=20,    # Remove very low frequencies
            fmax=8000   # Nyquist limit for 16kHz
        )
        
        # Ensure correct orientation: mel_spec is (n_mels, time_frames)
        # which should be (64, 63) for our model
        # DEBUG: Check actual shape
        # print(f"DEBUG: mel_spec shape: {mel_spec.shape}, expected: (64, time_frames)")
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1] range
        mel_spec_norm = (mel_spec_db + 80) / 80  # Assuming -80dB to 0dB range
        mel_spec_norm = np.clip(mel_spec_norm, -1, 1)
        
        # Ensure correct shape - mel_spec_norm should be (n_mels=64, time_frames)
        # If it's transposed, fix it
        if mel_spec_norm.shape[0] != self.config['n_mels']:
            if mel_spec_norm.shape[1] == self.config['n_mels']:
                # It's transposed, fix it
                mel_spec_norm = mel_spec_norm.T
            else:
                print(f"⚠️  Unexpected mel spectrogram shape: {mel_spec_norm.shape}")
        
        # Ensure correct time dimension
        if mel_spec_norm.shape[1] != self.config['n_frames']:
            # Resize time dimension
            mel_spec_norm = tf.image.resize(
                mel_spec_norm[..., np.newaxis], 
                (self.config['n_mels'], self.config['n_frames'])
            ).numpy().squeeze()
            
        # Final shape check
        if mel_spec_norm.shape != (self.config['n_mels'], self.config['n_frames']):
            print(f"⚠️  Final mel spectrogram shape mismatch: {mel_spec_norm.shape}, expected: ({self.config['n_mels']}, {self.config['n_frames']})")
            
        return mel_spec_norm[..., np.newaxis]  # Return as (freq, time, channels) = (64, 63, 1) for CNN

class SaitModelArchitecture:
    """DS-CNN+GRU model architecture optimized for nRF5340"""
    
    def __init__(self, config=MODEL_CONFIG):
        self.config = config
        
    def create_ds_cnn_block(self, inputs, filters, name_prefix):
        """Create Depthwise Separable CNN block"""
        # Depthwise convolution
        x = layers.DepthwiseConv2D(
            (3, 3), 
            padding='same',
            name=f'{name_prefix}_depthwise'
        )(inputs)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.ReLU(name=f'{name_prefix}_relu1')(x)
        
        # Pointwise convolution
        x = layers.Conv2D(
            filters,
            (1, 1),
            padding='same',
            name=f'{name_prefix}_pointwise'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = layers.ReLU(name=f'{name_prefix}_relu2')(x)
        
        # Dropout for regularization
        x = layers.Dropout(
            self.config['conv_dropout'],
            name=f'{name_prefix}_dropout'
        )(x)
        
        return x
        
    def build_model(self):
        """Build complete DS-CNN+GRU model"""
        # Input layer
        inputs = layers.Input(
            shape=self.config['input_shape'],
            name='mel_spectrogram_input'
        )
        
        # DS-CNN feature extraction layers
        x = self.create_ds_cnn_block(inputs, 32, 'ds_cnn_1')
        x = layers.MaxPooling2D((2, 2), name='pool_1')(x)
        
        x = self.create_ds_cnn_block(x, 64, 'ds_cnn_2')
        x = layers.MaxPooling2D((2, 2), name='pool_2')(x)
        
        x = self.create_ds_cnn_block(x, 128, 'ds_cnn_3')
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Reshape for RNN processing  
        # Flatten and create sequence dimension
        x = layers.Flatten(name='flatten_features')(x)
        x = layers.RepeatVector(8, name='repeat_for_sequence')(x)  # Create 8-step sequence
        
        # GRU layers for temporal modeling
        x = layers.GRU(
            self.config['gru_units'],
            return_sequences=True,
            dropout=self.config['gru_dropout'],
            name='gru_1'
        )(x)
        
        x = layers.GRU(
            self.config['gru_units'] // 2,
            dropout=self.config['gru_dropout'],
            name='gru_2'
        )(x)
        
        # Classification head
        x = layers.Dense(
            64,
            activation='relu',
            name='dense_1'
        )(x)
        x = layers.Dropout(0.5, name='final_dropout')(x)
        
        # Output layer matching SAIT_01 classes
        outputs = layers.Dense(
            self.config['num_classes'],
            activation='softmax',
            name='classification_output'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='SAIT01_DS_CNN_GRU')
        
        return model
        
    def compile_model(self, model):
        """Compile model with appropriate optimizer and loss"""
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return model
        
    def convert_to_tflite(self, model, output_path):
        """Convert trained model to TensorFlow Lite for nRF5340"""
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations for edge deployment
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantize to INT8 for maximum compression
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        # Print model statistics
        model_size = len(tflite_model) / 1024  # Size in KB
        print(f"TensorFlow Lite model size: {model_size:.1f} KB")
        print(f"Target size: {self.config['model_size_kb']} KB")
        
        if model_size <= self.config['model_size_kb']:
            print("✅ Model fits within nRF5340 memory constraints")
        else:
            print("❌ Model too large for nRF5340, consider further optimization")
            
        return tflite_model

def create_dataset_from_drone_acoustics(data_dir, config=MODEL_CONFIG):
    """Create training dataset from Helsinki drone acoustics data"""
    preprocessor = SaitAudioPreprocessor(config)
    
    # Class mapping for drone acoustics dataset
    class_mapping = {
        'background': 0,  # SAIT01_ML_CLASS_UNKNOWN
        'drone': 4,       # SAIT01_ML_CLASS_AIRCRAFT  
        'helicopter': 4   # SAIT01_ML_CLASS_AIRCRAFT (same as drone for now)
    }
    
    X_data = []
    y_data = []
    
    data_path = Path(data_dir)
    
    for split in ['train', 'val']:
        split_path = data_path / split
        if not split_path.exists():
            continue
            
        for class_name in class_mapping.keys():
            class_path = split_path / class_name
            if not class_path.exists():
                continue
                
            print(f"Processing {split}/{class_name}...")
            
            for audio_file in class_path.glob("*.wav"):
                try:
                    # Load and preprocess audio
                    audio = preprocessor.load_and_resample(str(audio_file))
                    mel_spec = preprocessor.extract_mel_spectrogram(audio)
                    
                    # mel_spec already has channel dimension from preprocessing
                    
                    X_data.append(mel_spec)
                    y_data.append(class_mapping[class_name])
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
                    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"Dataset created: {X_data.shape[0]} samples")
    print(f"Input shape: {X_data.shape[1:]}")
    print(f"Class distribution: {np.bincount(y_data)}")
    
    return X_data, y_data

def train_sait01_model(data_dir, output_dir="models"):
    """Complete training pipeline for SAIT_01 model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Creating dataset from drone acoustics data...")
    X, y = create_dataset_from_drone_acoustics(data_dir)
    
    # Create model architecture
    print("Building DS-CNN+GRU model...")
    model_builder = SaitModelArchitecture()
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    # Print model summary
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        X, y,
        batch_size=MODEL_CONFIG['batch_size'],
        epochs=MODEL_CONFIG['epochs'],
        validation_split=MODEL_CONFIG['validation_split'],
        verbose=1
    )
    
    # Save Keras model
    keras_path = os.path.join(output_dir, "sait01_audio_model.keras")
    model.save(keras_path)
    print(f"Keras model saved to: {keras_path}")
    
    # Convert to TensorFlow Lite
    print("Converting to TensorFlow Lite...")
    tflite_path = os.path.join(output_dir, "sait01_audio_model.tflite")
    tflite_model = model_builder.convert_to_tflite(model, tflite_path)
    print(f"TensorFlow Lite model saved to: {tflite_path}")
    
    # Generate C header file for nRF5340
    generate_c_header(tflite_model, os.path.join(output_dir, "sait01_model_data.h"))
    
    return model, history

def generate_c_header(tflite_model, output_path):
    """Generate C header file with model data for nRF5340"""
    model_data = list(tflite_model)
    
    header_content = f"""/*
 * SAIT_01 TinyML Model Data
 * Generated TensorFlow Lite model for nRF5340 deployment
 * Model size: {len(model_data)} bytes
 */

#ifndef SAIT01_MODEL_DATA_H
#define SAIT01_MODEL_DATA_H

#include <stdint.h>

// Model size in bytes
#define SAIT01_MODEL_SIZE {len(model_data)}

// TensorFlow Lite model data
extern const unsigned char sait01_model_data[SAIT01_MODEL_SIZE];

const unsigned char sait01_model_data[SAIT01_MODEL_SIZE] = {{
"""
    
    # Add model data as hex bytes
    for i, byte in enumerate(model_data):
        if i % 16 == 0:
            header_content += "\n    "
        header_content += f"0x{byte:02x}, "
        
    header_content = header_content.rstrip(", ")
    header_content += """
};

#endif /* SAIT01_MODEL_DATA_H */
"""
    
    with open(output_path, 'w') as f:
        f.write(header_content)
        
    print(f"C header file generated: {output_path}")

if __name__ == "__main__":
    # Train model using Helsinki drone acoustics dataset
    data_directory = "edth-copenhagen-drone-acoustics/data/raw"
    
    if os.path.exists(data_directory):
        print("Starting SAIT_01 model training...")
        model, history = train_sait01_model(data_directory)
        print("Training completed successfully!")
    else:
        print(f"Data directory not found: {data_directory}")
        print("Please ensure the Helsinki drone acoustics dataset is downloaded and extracted.")