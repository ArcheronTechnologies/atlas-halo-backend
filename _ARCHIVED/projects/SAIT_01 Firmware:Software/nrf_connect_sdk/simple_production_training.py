#!/usr/bin/env python3
"""
Simple Production TinyML Training - Minimal dependencies
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# Minimal imports to avoid dependency issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Add current directory to path
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class SimpleProductionTrainer:
    """Simplified trainer with minimal dependencies"""
    
    def __init__(self, data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        self.data_dir = data_dir
        self.preprocessor = SaitAudioPreprocessor()
        
        # Class mapping
        self.class_mapping = {'background': 0, 'drone': 1, 'helicopter': 2}
        
        print("🚀 Simple Production Trainer")
        print(f"📁 Data: {data_dir}")
        
    def load_dataset_simple(self, max_per_class=150):
        """Load dataset with simple structure"""
        print(f"📊 Loading up to {max_per_class} samples per class...")
        
        X_data = []
        y_data = []
        
        # Load training data
        train_path = Path(self.data_dir) / 'train'
        
        for class_name, class_id in self.class_mapping.items():
            class_path = train_path / class_name
            if not class_path.exists():
                print(f"❌ Missing: {class_path}")
                continue
                
            files = list(class_path.glob("*.wav"))[:max_per_class]
            print(f"📂 Loading {len(files)} {class_name} samples...")
            
            for i, audio_file in enumerate(files):
                try:
                    audio = self.preprocessor.load_and_resample(str(audio_file))
                    mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    X_data.append(mel_spec)
                    y_data.append(class_id)
                    
                    if (i + 1) % 25 == 0:
                        print(f"  {i + 1}/{len(files)} done")
                        
                except Exception as e:
                    print(f"  ❌ {audio_file.name}: {e}")
                    continue
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"✅ Loaded {len(X_data)} samples")
        print(f"📊 Shape: {X_data.shape}")
        print(f"📋 Classes: {np.bincount(y_data)}")
        
        return X_data, y_data
        
    def create_simple_model(self):
        """Create efficient model for nRF5340"""
        print("\n🧠 Building Optimized Model")
        
        # Efficient architecture
        inputs = tf.keras.layers.Input(shape=(63, 64, 1))
        
        # Depthwise separable convolutions
        x = tf.keras.layers.SeparableConv2D(16, 3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        x = tf.keras.layers.SeparableConv2D(32, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name='SAIT01_Simple')
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("📝 Model Summary:")
        model.summary()
        
        # Check size
        params = model.count_params()
        size_kb = params * 4 / 1024
        print(f"📏 Model size: {size_kb:.1f} KB ({params} params)")
        
        return model
        
    def simple_train_test_split(self, X, y, test_ratio=0.2):
        """Simple train/test split"""
        n = len(X)
        n_test = int(n * test_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(n)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
        
    def train_model(self, X_data, y_data):
        """Train the model"""
        print("\n🚀 Training Model")
        
        # Split data
        X_train, X_test, y_train, y_test = self.simple_train_test_split(X_data, y_data)
        
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Create model
        model = self.create_simple_model()
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4
            )
        ]
        
        # Train
        print("🔄 Starting training...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        train_time = time.time() - start_time
        print(f"⏱️  Training: {train_time:.1f}s")
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"🎯 Test accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        
        return model, history, (X_test, y_test)
        
    def convert_to_tflite(self, model, X_sample):
        """Convert to TensorFlow Lite"""
        print("\n🔄 Converting to TensorFlow Lite")
        
        # Basic conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        tflite_size = len(tflite_model) / 1024
        
        print(f"✅ TFLite model: {tflite_size:.1f} KB")
        
        # Try quantization
        try:
            print("🗜️  Applying quantization...")
            
            # Create representative dataset
            def representative_dataset():
                for i in range(min(100, len(X_sample))):
                    yield [X_sample[i:i+1].astype(np.float32)]
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            quantized_model = converter.convert()
            quantized_size = len(quantized_model) / 1024
            
            print(f"✅ Quantized: {quantized_size:.1f} KB")
            print(f"🗜️  Reduction: {((tflite_size - quantized_size) / tflite_size * 100):.1f}%")
            
            # Check nRF5340 compatibility
            if quantized_size <= 80:
                print("✅ Fits nRF5340 (80KB limit)")
            else:
                print("⚠️  May exceed nRF5340 limit")
            
            # Save models
            with open('sait01_simple_model.tflite', 'wb') as f:
                f.write(tflite_model)
            with open('sait01_quantized_model.tflite', 'wb') as f:
                f.write(quantized_model)
            
            print("💾 Models saved")
            
            return tflite_model, quantized_model
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            with open('sait01_simple_model.tflite', 'wb') as f:
                f.write(tflite_model)
            return tflite_model, None
            
    def test_inference_speed(self, model, X_test):
        """Test inference speed"""
        print("\n⚡ Testing Inference Speed")
        
        # Warm up
        for _ in range(3):
            _ = model.predict(X_test[:1], verbose=0)
        
        # Time multiple inferences
        times = []
        for _ in range(50):
            start = time.time()
            _ = model.predict(X_test[:1], verbose=0)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        print(f"⏱️  Average inference: {avg_time:.2f} ms")
        
        if avg_time < 100:
            print("✅ Real-time capable")
        else:
            print("⚠️  May be slow for real-time")
            
        return avg_time
        
    def run_training(self):
        """Complete training pipeline"""
        print("🚀 SAIT_01 Simple Production Training")
        print("=" * 50)
        
        # Load dataset
        X_data, y_data = self.load_dataset_simple()
        
        if len(X_data) == 0:
            print("❌ No data loaded")
            return None
        
        # Train model
        model, history, (X_test, y_test) = self.train_model(X_data, y_data)
        
        # Test speed
        avg_time = self.test_inference_speed(model, X_test)
        
        # Convert to TFLite
        tflite_model, quantized_model = self.convert_to_tflite(model, X_test)
        
        print("\n🎯 TRAINING COMPLETE")
        print("=" * 30)
        print(f"✅ Dataset: {len(X_data)} samples")
        print(f"✅ Model trained successfully")
        print(f"✅ Inference: {avg_time:.1f} ms")
        print("✅ TFLite conversion done")
        if quantized_model:
            print("✅ Quantization successful")
        
        return {
            'model': model,
            'history': history,
            'tflite': tflite_model,
            'quantized': quantized_model
        }

def main():
    trainer = SimpleProductionTrainer()
    results = trainer.run_training()
    
    if results:
        print("\n🚀 Success! Models ready for deployment.")
    else:
        print("\n❌ Training failed.")

if __name__ == "__main__":
    main()