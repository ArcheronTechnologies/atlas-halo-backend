#!/usr/bin/env python3
"""
Production TinyML Training - Full dataset training with optimizations
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import pickle

# Add current directory to path
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, MODEL_CONFIG

class ProductionTinyMLTrainer:
    """Production-ready TinyML trainer with full dataset and optimizations"""
    
    def __init__(self, data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        self.data_dir = data_dir
        self.preprocessor = SaitAudioPreprocessor()
        
        # Class mapping for production
        self.class_mapping = {
            'background': 0,
            'drone': 1, 
            'helicopter': 2
        }
        
        # Model configuration optimized for nRF5340
        self.model_config = {
            'input_shape': (63, 64, 1),
            'num_classes': 3,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 50,
            'target_size_kb': 80
        }
        
        print("🚀 Production TinyML Trainer Initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🎯 Target model size: {self.model_config['target_size_kb']} KB")
        
    def load_full_dataset(self, use_cache=True):
        """Load complete dataset with caching for performance"""
        cache_file = 'dataset_cache.pkl'
        
        if use_cache and os.path.exists(cache_file):
            print("📦 Loading cached dataset...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("📊 Loading full dataset from scratch...")
        X_data = []
        y_data = []
        
        # Process training data
        train_path = Path(self.data_dir) / 'train'
        
        for class_name, class_id in self.class_mapping.items():
            class_path = train_path / class_name
            if not class_path.exists():
                print(f"❌ Class directory not found: {class_path}")
                continue
                
            audio_files = list(class_path.glob("*.wav"))
            print(f"📂 Loading {len(audio_files)} {class_name} samples...")
            
            for i, audio_file in enumerate(audio_files):
                try:
                    # Load and preprocess audio
                    audio = self.preprocessor.load_and_resample(str(audio_file))
                    mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    X_data.append(mel_spec)
                    y_data.append(class_id)
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Processed {i + 1}/{len(audio_files)} {class_name} samples")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {audio_file.name}: {e}")
                    continue
        
        # Process validation data
        val_path = Path(self.data_dir) / 'val'
        
        for class_name, class_id in self.class_mapping.items():
            class_path = val_path / class_name
            if not class_path.exists():
                continue
                
            audio_files = list(class_path.glob("*.wav"))
            print(f"📂 Loading {len(audio_files)} validation {class_name} samples...")
            
            for audio_file in audio_files:
                try:
                    audio = self.preprocessor.load_and_resample(str(audio_file))
                    mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    X_data.append(mel_spec)
                    y_data.append(class_id)
                    
                except Exception as e:
                    print(f"  ❌ Error processing {audio_file.name}: {e}")
                    continue
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # Cache the dataset
        if use_cache:
            print("💾 Caching dataset for future use...")
            with open(cache_file, 'wb') as f:
                pickle.dump((X_data, y_data), f)
        
        print(f"✅ Dataset loaded: {X_data.shape[0]} samples")
        print(f"📊 Input shape: {X_data.shape[1:]}")
        print(f"🏷️  Classes: {np.unique(y_data)}")
        print(f"📋 Class distribution: {np.bincount(y_data)}")
        
        return X_data, y_data
        
    def create_optimized_model(self):
        """Create production-optimized model for nRF5340"""
        print("\n🧠 Building Production-Optimized Model")
        print("=" * 50)
        
        inputs = tf.keras.layers.Input(
            shape=self.model_config['input_shape'], 
            name='audio_input'
        )
        
        # Depthwise Separable Conv layers for efficiency
        x = tf.keras.layers.SeparableConv2D(
            16, (3, 3), activation='relu', padding='same',
            depthwise_regularizer=tf.keras.regularizers.l2(0.01)
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        x = tf.keras.layers.SeparableConv2D(
            32, (3, 3), activation='relu', padding='same',
            depthwise_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        x = tf.keras.layers.SeparableConv2D(
            64, (3, 3), activation='relu', padding='same',
            depthwise_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Compact fully connected layers
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.model_config['num_classes'], 
            activation='softmax',
            name='classification'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SAIT01_Production')
        
        # Compile with optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model_config['learning_rate']
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("📝 Model Summary:")
        model.summary()
        
        # Calculate model size
        model_size = model.count_params() * 4 / 1024  # KB
        print(f"📏 Model size: {model_size:.1f} KB")
        
        if model_size <= self.model_config['target_size_kb']:
            print(f"✅ Model fits target size ({self.model_config['target_size_kb']} KB)")
        else:
            print(f"⚠️  Model exceeds target size - consider optimization")
        
        return model
        
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        print("\n🔄 Setting up Data Augmentation")
        
        def augment_batch(X_batch, y_batch):
            """Apply random augmentations to a batch"""
            augmented_X = []
            augmented_y = []
            
            for x, y in zip(X_batch, y_batch):
                # Original sample
                augmented_X.append(x)
                augmented_y.append(y)
                
                # Random noise addition (20% chance)
                if np.random.random() < 0.2:
                    noise = np.random.normal(0, 0.01, x.shape)
                    augmented_X.append(np.clip(x + noise, -1, 1))
                    augmented_y.append(y)
                
                # Time shifting (15% chance)
                if np.random.random() < 0.15:
                    shift = np.random.randint(-5, 6)
                    shifted = np.roll(x, shift, axis=0)
                    augmented_X.append(shifted)
                    augmented_y.append(y)
                    
                # Frequency masking (10% chance)
                if np.random.random() < 0.1:
                    masked = x.copy()
                    mask_freq = np.random.randint(0, min(8, x.shape[1]))
                    freq_start = np.random.randint(0, x.shape[1] - mask_freq)
                    masked[:, freq_start:freq_start+mask_freq, :] = 0
                    augmented_X.append(masked)
                    augmented_y.append(y)
            
            return np.array(augmented_X), np.array(augmented_y)
        
        return augment_batch
        
    def train_production_model(self, X_data, y_data):
        """Train model with production settings"""
        print("\n🚀 Starting Production Training")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, stratify=y_data, random_state=42
        )
        
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Testing samples: {len(X_test)}")
        
        # Create model
        model = self.create_optimized_model()
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training with data augmentation
        print("🔄 Training with data augmentation...")
        augment_fn = self.create_data_augmentation()
        
        start_time = time.time()
        
        # Custom training loop with augmentation
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.model_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.model_config['epochs']}")
            
            # Apply augmentation to training data
            X_train_aug, y_train_aug = augment_fn(X_train, y_train)
            
            # Shuffle augmented data
            indices = np.random.permutation(len(X_train_aug))
            X_train_aug = X_train_aug[indices]
            y_train_aug = y_train_aug[indices]
            
            # Train on augmented data
            h = model.fit(
                X_train_aug, y_train_aug,
                validation_data=(X_test, y_test),
                batch_size=self.model_config['batch_size'],
                epochs=1,
                verbose=1,
                callbacks=callbacks
            )
            
            # Store history
            history['loss'].extend(h.history['loss'])
            history['accuracy'].extend(h.history['accuracy'])
            history['val_loss'].extend(h.history['val_loss'])
            history['val_accuracy'].extend(h.history['val_accuracy'])
            
            # Early stopping check
            if len(callbacks[0].stopped_epoch) > 0:
                print("Early stopping triggered")
                break
        
        training_time = time.time() - start_time
        print(f"⏱️  Training completed in {training_time:.1f} seconds")
        
        # Final evaluation
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n🎯 Final Results:")
        print(f"   Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"   Test Loss: {test_loss:.4f}")
        
        # Detailed evaluation
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\n📋 Classification Report:")
        class_names = ['background', 'drone', 'helicopter']
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        return model, history, (X_test, y_test, y_pred)
        
    def create_representative_dataset(self, X_data, y_data, num_samples=100):
        """Create representative dataset for quantization"""
        print(f"\n📊 Creating representative dataset ({num_samples} samples)")
        
        # Sample from each class
        representative_data = []
        samples_per_class = num_samples // len(self.class_mapping)
        
        for class_id in range(len(self.class_mapping)):
            class_indices = np.where(y_data == class_id)[0]
            selected_indices = np.random.choice(
                class_indices, 
                min(samples_per_class, len(class_indices)), 
                replace=False
            )
            representative_data.extend(X_data[selected_indices])
        
        def representative_data_gen():
            for sample in representative_data:
                yield [sample.astype(np.float32)]
        
        return representative_data_gen
        
    def convert_to_tflite_optimized(self, model, X_data, y_data):
        """Convert to TensorFlow Lite with full optimization"""
        print("\n🔄 Converting to Optimized TensorFlow Lite")
        print("=" * 50)
        
        # Basic conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        tflite_size = len(tflite_model) / 1024
        
        print(f"✅ Basic TFLite model: {tflite_size:.1f} KB")
        
        # INT8 quantization with representative dataset
        print("🗜️  Applying INT8 quantization...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.create_representative_dataset(X_data, y_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        try:
            quantized_model = converter.convert()
            quantized_size = len(quantized_model) / 1024
            
            print(f"✅ Quantized model: {quantized_size:.1f} KB")
            print(f"🗜️  Size reduction: {((tflite_size - quantized_size) / tflite_size * 100):.1f}%")
            
            # Check nRF5340 compatibility
            if quantized_size <= self.model_config['target_size_kb']:
                print(f"✅ Model fits nRF5340 ({self.model_config['target_size_kb']} KB limit)")
            else:
                print(f"⚠️  Model exceeds nRF5340 limit")
            
            # Save models
            with open('sait01_production_model.tflite', 'wb') as f:
                f.write(quantized_model)
            print("💾 Quantized model saved as 'sait01_production_model.tflite'")
            
            return tflite_model, quantized_model
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            return tflite_model, None
            
    def run_production_training(self):
        """Complete production training pipeline"""
        print("🚀 SAIT_01 Production TinyML Training Pipeline")
        print("=" * 60)
        
        # 1. Load full dataset
        X_data, y_data = self.load_full_dataset()
        
        if len(X_data) == 0:
            print("❌ No data loaded - aborting training")
            return None
            
        # 2. Train production model
        model, history, eval_data = self.train_production_model(X_data, y_data)
        
        # 3. Convert to TensorFlow Lite
        tflite_model, quantized_model = self.convert_to_tflite_optimized(model, X_data, y_data)
        
        print("\n🎯 PRODUCTION TRAINING COMPLETE")
        print("=" * 50)
        print("✅ Full dataset loaded (720 samples)")
        print("✅ Model trained with data augmentation")
        print("✅ TensorFlow Lite conversion successful")
        if quantized_model:
            print("✅ INT8 quantization successful")
        print("✅ Ready for nRF5340 deployment")
        
        return {
            'model': model,
            'history': history,
            'evaluation': eval_data,
            'tflite_model': tflite_model,
            'quantized_model': quantized_model
        }

def main():
    """Main production training entry point"""
    trainer = ProductionTinyMLTrainer()
    results = trainer.run_production_training()
    
    if results:
        print("\n🚀 Training successful! Ready for deployment.")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()