#!/usr/bin/env python3
"""
Test Fixed SAIT_01 Model with Shape Corrections and Balanced Dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import librosa
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import time

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, SaitModelArchitecture, MODEL_CONFIG

class FixedSAIT01Tester:
    """Test the SAIT_01 model with fixed preprocessing and balanced data"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.model_builder = SaitModelArchitecture()
        
    def load_balanced_dataset(self):
        """Load the balanced dataset"""
        print("📊 Loading balanced dataset...")
        
        # Check for quick balanced data first
        balanced_dir = Path("quick_balanced_data")
        if not balanced_dir.exists():
            print("❌ Balanced dataset not found. Creating minimal test dataset...")
            return self.create_minimal_test_dataset()
        
        X = []
        y = []
        class_names = ['background', 'drone', 'aircraft']
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = balanced_dir / class_name
            if class_dir.exists():
                audio_files = list(class_dir.glob("*.wav"))
                print(f"   {class_name}: {len(audio_files)} files")
                
                for audio_file in tqdm(audio_files[:100], desc=f"Loading {class_name}"):  # Limit for testing
                    try:
                        # Load and preprocess audio
                        audio = self.preprocessor.load_and_resample(audio_file)
                        features = self.preprocessor.extract_mel_spectrogram(audio)
                        
                        X.append(features)
                        y.append(class_idx)
                        
                    except Exception as e:
                        print(f"⚠️  Error processing {audio_file}: {e}")
                        continue
        
        if len(X) == 0:
            return self.create_minimal_test_dataset()
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Dataset loaded: {X.shape}, Labels: {len(y)}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_minimal_test_dataset(self):
        """Create minimal synthetic dataset for testing"""
        print("🔄 Creating minimal synthetic test dataset...")
        
        X = []
        y = []
        
        # Create synthetic spectrograms for each class
        for class_idx in range(3):  # background, drone, aircraft
            for i in range(50):  # 50 samples per class
                # Generate synthetic mel spectrogram
                if class_idx == 0:  # background - lower energy
                    synth_spec = np.random.randn(64, 63) * 0.3
                elif class_idx == 1:  # drone - periodic patterns
                    synth_spec = np.random.randn(64, 63) * 0.5
                    # Add periodic patterns for drone sound
                    for f in range(10, 30):  # Focus energy in certain frequency bands
                        synth_spec[f, :] += np.sin(np.linspace(0, 4*np.pi, 63)) * 0.5
                else:  # aircraft - different pattern
                    synth_spec = np.random.randn(64, 63) * 0.6
                    # Add different frequency characteristics
                    for f in range(35, 55):
                        synth_spec[f, :] += np.cos(np.linspace(0, 2*np.pi, 63)) * 0.4
                
                # Normalize
                synth_spec = (synth_spec - np.mean(synth_spec)) / (np.std(synth_spec) + 1e-8)
                synth_spec = np.clip(synth_spec, -1, 1)
                
                X.append(synth_spec[..., np.newaxis])
                y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Synthetic dataset: {X.shape}, Labels: {len(y)}")
        return X, y
    
    def create_optimized_model(self):
        """Create optimized model for better accuracy"""
        print("🏗️  Creating optimized model...")
        
        inputs = keras.Input(shape=MODEL_CONFIG['input_shape'], name='mel_input')
        
        # Lighter CNN layers for better generalization
        x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.5)(x)
        
        # Classification head
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        outputs = keras.layers.Dense(3, activation='softmax', name='classification')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SAIT01_Fixed')
        
        # Compile with appropriate settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created: {model.count_params()} parameters")
        return model
    
    def train_and_test_model(self, X, y, test_split=0.2):
        """Train and test the model"""
        print("🚀 Training and testing model...")
        
        # Split data
        split_idx = int(len(X) * (1 - test_split))
        indices = np.random.permutation(len(X))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Create model
        model = self.create_optimized_model()
        
        # Train model
        print("🔄 Training model...")
        start_time = time.time()
        
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
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=30,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Test model
        print("🔍 Testing model...")
        test_start = time.time()
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        inference_time = (time.time() - test_start) / len(X_test) * 1000  # ms per sample
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"\n📊 Results:")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Inference time: {inference_time:.2f}ms per sample")
        print(f"   Test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed classification report
        class_names = ['Background', 'Drone', 'Aircraft']
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        # Model size estimation
        model.save('sait01_fixed_model.h5')
        model_size = os.path.getsize('sait01_fixed_model.h5') / 1024  # KB
        print(f"📏 Model size: {model_size:.1f} KB")
        
        # Convert to TFLite for deployment
        print("📱 Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open('sait01_fixed_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        tflite_size = len(tflite_model) / 1024  # KB
        print(f"📱 TFLite model size: {tflite_size:.1f} KB")
        
        # Test TFLite model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test a few samples with TFLite
        tflite_correct = 0
        tflite_times = []
        
        for i in range(min(10, len(X_test))):
            sample = X_test[i:i+1].astype(np.float32)
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            tflite_time = (time.time() - start_time) * 1000
            
            tflite_times.append(tflite_time)
            pred_class = np.argmax(output[0])
            if pred_class == y_test[i]:
                tflite_correct += 1
        
        tflite_accuracy = tflite_correct / min(10, len(X_test))
        avg_tflite_time = np.mean(tflite_times)
        
        print(f"📱 TFLite accuracy: {tflite_accuracy:.3f} ({tflite_accuracy*100:.1f}%)")
        print(f"📱 TFLite inference: {avg_tflite_time:.2f}ms per sample")
        
        return {
            'accuracy': accuracy,
            'model_size_kb': model_size,
            'tflite_size_kb': tflite_size,
            'inference_time_ms': inference_time,
            'tflite_time_ms': avg_tflite_time,
            'tflite_accuracy': tflite_accuracy,
            'training_time': training_time
        }

def main():
    """Main test execution"""
    print("🎯 SAIT_01 Fixed Model Test")
    print("=" * 50)
    
    # Initialize tester
    tester = FixedSAIT01Tester()
    
    # Load dataset
    X, y = tester.load_balanced_dataset()
    
    if len(X) == 0:
        print("❌ No data available for testing")
        return
    
    # Train and test
    results = tester.train_and_test_model(X, y)
    
    # Summary
    print(f"\n🎉 SAIT_01 Model Test Complete!")
    print("=" * 50)
    print(f"✅ Accuracy: {results['accuracy']*100:.1f}%")
    print(f"📏 Model Size: {results['tflite_size_kb']:.1f} KB")
    print(f"⏱️  Inference: {results['tflite_time_ms']:.1f}ms")
    print(f"🎯 Status: {'READY FOR DEPLOYMENT' if results['accuracy'] > 0.6 else 'NEEDS IMPROVEMENT'}")
    
    if results['accuracy'] > 0.8:
        print("🏆 EXCELLENT ACCURACY ACHIEVED!")
    elif results['accuracy'] > 0.6:
        print("✅ Good accuracy for prototype")
    else:
        print("⚠️  Accuracy needs improvement - Consider more data")

if __name__ == "__main__":
    main()