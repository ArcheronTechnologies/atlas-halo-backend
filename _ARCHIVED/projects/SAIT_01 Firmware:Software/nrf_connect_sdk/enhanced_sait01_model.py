#!/usr/bin/env python3
"""
Enhanced SAIT_01 TinyML Model with YAMNet Transfer Learning
Optimized for maximum accuracy on vehicle/aircraft detection
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import librosa
import os
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm

# Enhanced model configuration for high accuracy
ENHANCED_CONFIG = {
    # Audio parameters optimized for transfer learning
    'sample_rate': 16000,
    'window_ms': 1000,
    'hop_length': 512,              # Optimized for YAMNet
    'n_fft': 1024,                  # Higher resolution
    'n_mels': 64,
    'n_frames': 63,
    
    # Enhanced architecture
    'input_shape': (64, 63, 1),
    'num_classes': 5,               # Background, Vehicle, Aircraft, Helicopter, Drone
    'yamnet_embedding_size': 1024,  # YAMNet output size
    'model_size_kb': 120,           # Slightly larger for better accuracy
    
    # Training parameters
    'batch_size': 16,               # Smaller batches for stability
    'epochs': 50,
    'learning_rate': 0.0001,        # Lower for transfer learning
    'validation_split': 0.2,
    
    # Augmentation parameters
    'augment_factor': 5,            # 5x data augmentation
    'noise_factor': 0.02,
    'pitch_shift_range': 3,         # ±3 semitones
    'time_stretch_range': 0.2,      # ±20%
}

class EnhancedAudioDataset:
    """Enhanced dataset manager with public audio sources"""
    
    def __init__(self, base_dir="enhanced_audio_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_esc50_dataset(self):
        """Download ESC-50 environmental sound dataset"""
        print("📥 Downloading ESC-50 dataset...")
        
        esc50_dir = self.base_dir / "esc50"
        if esc50_dir.exists():
            print("✅ ESC-50 already downloaded")
            return esc50_dir
            
        esc50_dir.mkdir(exist_ok=True)
        
        # Download ESC-50 from GitHub
        url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
        response = requests.get(url, stream=True)
        
        zip_path = esc50_dir / "esc50.zip"
        with open(zip_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(esc50_dir)
        
        zip_path.unlink()  # Remove zip file
        print("✅ ESC-50 downloaded and extracted")
        return esc50_dir
    
    def download_fsd50k_subset(self):
        """Download FSD50K subset for aircraft/vehicle sounds"""
        print("📥 Downloading FSD50K aircraft/vehicle subset...")
        
        fsd50k_dir = self.base_dir / "fsd50k_subset"
        if fsd50k_dir.exists():
            print("✅ FSD50K subset already downloaded")
            return fsd50k_dir
            
        fsd50k_dir.mkdir(exist_ok=True)
        
        # Note: In production, this would download from Zenodo
        # For now, create structure for manual download
        print("📋 FSD50K requires manual download from https://zenodo.org/records/4060432")
        print("    Extract aircraft, helicopter, and vehicle sounds to:", fsd50k_dir)
        
        return fsd50k_dir
    
    def organize_balanced_dataset(self, existing_data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        """Create balanced dataset from existing and new sources"""
        print("🔄 Creating balanced dataset...")
        
        balanced_dir = self.base_dir / "balanced_dataset"
        balanced_dir.mkdir(exist_ok=True)
        
        # Create class directories
        classes = ['background', 'vehicle', 'aircraft', 'helicopter', 'drone']
        for cls in classes:
            (balanced_dir / cls).mkdir(exist_ok=True)
        
        # Copy existing data
        existing_path = Path(existing_data_dir)
        if existing_path.exists():
            print(f"📁 Processing existing data from {existing_path}")
            
            # Process each class
            for src_class in ['background', 'drone', 'helicopter']:
                src_dir = existing_path / "train" / src_class
                if src_dir.exists():
                    dst_class = 'drone' if src_class == 'drone' else src_class
                    if src_class == 'helicopter':
                        dst_class = 'aircraft'  # Merge helicopters into aircraft
                    
                    dst_dir = balanced_dir / dst_class
                    
                    # Copy files with limit to balance dataset
                    max_files = 500 if src_class == 'background' else 1000
                    files = list(src_dir.glob("*.wav"))[:max_files]
                    
                    for i, file in enumerate(files):
                        dst_file = dst_dir / f"{src_class}_{i:04d}.wav"
                        if not dst_file.exists():
                            import shutil
                            shutil.copy2(file, dst_file)
        
        print("✅ Balanced dataset created")
        return balanced_dir

class YAMNetTransferLearning:
    """YAMNet-based transfer learning for audio classification"""
    
    def __init__(self, config=ENHANCED_CONFIG):
        self.config = config
        self.yamnet_model = None
        self.load_yamnet()
        
    def load_yamnet(self):
        """Load pre-trained YAMNet model"""
        print("🧠 Loading YAMNet pre-trained model...")
        try:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            print("✅ YAMNet loaded successfully")
        except Exception as e:
            print(f"⚠️  YAMNet loading failed: {e}")
            print("📋 Using fallback mel spectrogram features")
            self.yamnet_model = None
    
    def extract_yamnet_embeddings(self, audio_data):
        """Extract YAMNet embeddings from audio"""
        if self.yamnet_model is None:
            return self.extract_mel_features(audio_data)
        
        # Ensure audio is the right length (YAMNet expects 15600 samples minimum)
        if len(audio_data) < 15600:
            audio_data = np.pad(audio_data, (0, 15600 - len(audio_data)))
        
        # YAMNet expects float32 in [-1, 1] range
        audio_data = audio_data.astype(np.float32)
        
        # Extract embeddings
        _, embeddings, _ = self.yamnet_model(audio_data)
        
        # Average embeddings over time
        return tf.reduce_mean(embeddings, axis=0).numpy()
    
    def extract_mel_features(self, audio_data):
        """Fallback: Extract mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels']
        )
        
        # Convert to log scale and normalize
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Flatten and pad/crop to fixed size
        features = mel_spec_db.flatten()
        target_size = 1024  # Match YAMNet embedding size
        
        if len(features) > target_size:
            features = features[:target_size]
        elif len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        
        return features

class AdvancedAugmentation:
    """Advanced audio augmentation techniques for 2024"""
    
    def __init__(self, config=ENHANCED_CONFIG):
        self.config = config
    
    def augment_audio(self, audio_data):
        """Apply random augmentation to audio"""
        augmented = audio_data.copy()
        
        # Time stretching (±20%)
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.8, 1.2)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
        
        # Pitch shifting (±3 semitones)
        if np.random.random() < 0.5:
            n_steps = np.random.uniform(-3, 3)
            augmented = librosa.effects.pitch_shift(
                augmented, sr=self.config['sample_rate'], n_steps=n_steps
            )
        
        # Noise injection
        if np.random.random() < 0.3:
            noise = np.random.randn(len(augmented)) * self.config['noise_factor']
            augmented = augmented + noise
        
        # Normalize length
        target_length = self.config['sample_rate']  # 1 second
        if len(augmented) > target_length:
            start = np.random.randint(0, len(augmented) - target_length)
            augmented = augmented[start:start + target_length]
        elif len(augmented) < target_length:
            augmented = np.pad(augmented, (0, target_length - len(augmented)))
        
        return augmented
    
    def create_augmented_dataset(self, audio_files, labels, augment_factor=5):
        """Create augmented dataset"""
        print(f"🔄 Creating {augment_factor}x augmented dataset...")
        
        augmented_audio = []
        augmented_labels = []
        
        for i, (audio_file, label) in enumerate(tqdm(zip(audio_files, labels))):
            # Load original audio
            audio, _ = librosa.load(audio_file, sr=self.config['sample_rate'])
            
            # Add original
            augmented_audio.append(audio)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augment_factor - 1):
                aug_audio = self.augment_audio(audio)
                augmented_audio.append(aug_audio)
                augmented_labels.append(label)
        
        print(f"✅ Dataset augmented: {len(audio_files)} -> {len(augmented_audio)} samples")
        return augmented_audio, augmented_labels

class EnhancedSAIT01Model:
    """Enhanced SAIT_01 model with transfer learning"""
    
    def __init__(self, config=ENHANCED_CONFIG):
        self.config = config
        self.yamnet_transfer = YAMNetTransferLearning(config)
        self.model = None
        
    def create_transfer_model(self):
        """Create model using YAMNet transfer learning"""
        print("🏗️  Building enhanced transfer learning model...")
        
        # Input for YAMNet embeddings
        inputs = keras.Input(shape=(self.config['yamnet_embedding_size'],), name='yamnet_embeddings')
        
        # Classification head optimized for edge deployment
        x = layers.Dense(256, activation='relu', name='dense_1')(inputs)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        
        x = layers.Dense(64, activation='relu', name='dense_3')(x)
        x = layers.Dropout(0.1, name='dropout_3')(x)
        
        # Output layer
        outputs = layers.Dense(self.config['num_classes'], activation='softmax', name='classification')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SAIT01_YAMNet_Transfer')
        
        # Compile with optimized settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"✅ Model created: {model.count_params()} parameters")
        return model
    
    def create_hybrid_model(self):
        """Create hybrid model combining multiple approaches"""
        print("🔥 Building hybrid multi-path model...")
        
        # YAMNet embedding path
        yamnet_input = keras.Input(shape=(self.config['yamnet_embedding_size'],), name='yamnet_features')
        yamnet_path = layers.Dense(128, activation='relu')(yamnet_input)
        yamnet_path = layers.Dropout(0.2)(yamnet_path)
        
        # Mel spectrogram path
        mel_input = keras.Input(shape=self.config['input_shape'], name='mel_spectrogram')
        
        # Lightweight CNN for mel features
        mel_path = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(mel_input)
        mel_path = layers.MaxPooling2D((2, 2))(mel_path)
        mel_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(mel_path)
        mel_path = layers.GlobalAveragePooling2D()(mel_path)
        mel_path = layers.Dense(128, activation='relu')(mel_path)
        mel_path = layers.Dropout(0.2)(mel_path)
        
        # Combine paths
        combined = layers.concatenate([yamnet_path, mel_path])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        
        outputs = layers.Dense(self.config['num_classes'], activation='softmax')(combined)
        
        model = keras.Model(
            inputs=[yamnet_input, mel_input], 
            outputs=outputs, 
            name='SAIT01_Hybrid'
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"✅ Hybrid model created: {model.count_params()} parameters")
        return model
    
    def train_model(self, audio_files, labels, validation_split=0.2):
        """Train the enhanced model"""
        print("🚀 Starting enhanced training pipeline...")
        
        # Create augmented dataset
        augmenter = AdvancedAugmentation(self.config)
        aug_audio, aug_labels = augmenter.create_augmented_dataset(
            audio_files, labels, self.config['augment_factor']
        )
        
        # Extract features
        print("🔍 Extracting YAMNet features...")
        X_features = []
        for audio in tqdm(aug_audio):
            features = self.yamnet_transfer.extract_yamnet_embeddings(audio)
            X_features.append(features)
        
        X = np.array(X_features)
        y = np.array(aug_labels)
        
        print(f"📊 Training data: {X.shape}, Labels: {y.shape}")
        
        # Create and train model
        if self.model is None:
            self.create_transfer_model()
        
        # Advanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_sait01_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def convert_to_tflite(self, model_path='best_sait01_model.h5'):
        """Convert model to TensorFlow Lite for nRF5340 deployment"""
        print("📱 Converting to TensorFlow Lite...")
        
        if self.model is None:
            self.model = keras.models.load_model(model_path)
        
        # Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # Representative dataset for quantization
        def representative_dataset():
            # Use a small sample of data for calibration
            for _ in range(100):
                yield [np.random.randn(1, self.config['yamnet_embedding_size']).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        
        tflite_model = converter.convert()
        
        # Save model
        tflite_path = 'sait01_enhanced_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved: {tflite_path} ({len(tflite_model)} bytes)")
        return tflite_model

def main():
    """Main execution function"""
    print("🎯 Enhanced SAIT_01 Audio Classification System")
    print("=" * 60)
    
    # Initialize components
    dataset_manager = EnhancedAudioDataset()
    model = EnhancedSAIT01Model()
    
    # Download and prepare datasets
    print("\n📊 Phase 1: Dataset Preparation")
    print("-" * 30)
    
    # Create balanced dataset
    balanced_dir = dataset_manager.organize_balanced_dataset()
    
    # Collect audio files and labels
    audio_files = []
    labels = []
    
    class_names = ['background', 'vehicle', 'aircraft', 'helicopter', 'drone']
    for class_idx, class_name in enumerate(class_names):
        class_dir = balanced_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            audio_files.extend(files)
            labels.extend([class_idx] * len(files))
    
    print(f"📈 Total samples: {len(audio_files)}")
    print(f"📊 Class distribution: {np.bincount(labels)}")
    
    if len(audio_files) > 0:
        print("\n🚀 Phase 2: Model Training")
        print("-" * 30)
        
        # Train enhanced model
        history = model.train_model(audio_files, labels)
        
        print("\n📱 Phase 3: Model Deployment")
        print("-" * 30)
        
        # Convert to TFLite
        tflite_model = model.convert_to_tflite()
        
        print("\n🎉 Enhanced SAIT_01 Model Complete!")
        print("✅ Ready for nRF5340 deployment")
        
    else:
        print("⚠️  No audio files found. Please ensure dataset is properly set up.")

if __name__ == "__main__":
    main()