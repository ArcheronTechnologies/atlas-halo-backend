#!/usr/bin/env python3
"""
SAIT_01 TinyML Accuracy Testing - Comprehensive Model Validation
Tests audio recognition capabilities and accuracy on drone acoustics dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import time

# Add current directory to path for imports
sys.path.append('.')
from sait01_model_architecture import (
    SaitAudioPreprocessor, SaitModelArchitecture, 
    create_dataset_from_drone_acoustics, MODEL_CONFIG
)

class TinyMLAccuracyTester:
    """Comprehensive testing suite for SAIT_01 TinyML model"""
    
    def __init__(self, data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        self.data_dir = data_dir
        self.preprocessor = SaitAudioPreprocessor()
        self.model_builder = SaitModelArchitecture()
        
        # Class mapping for drone acoustics
        self.class_names = ['background', 'aircraft', 'aircraft']  # drone/helicopter -> aircraft
        self.sait_class_names = ['Unknown', 'Vehicle', 'Footsteps', 'Voices', 
                                'Aircraft', 'Machinery', 'Gunshot', 'Explosion']
        
        print("🎯 SAIT_01 TinyML Accuracy Tester Initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🎵 Audio config: {MODEL_CONFIG['sample_rate']}Hz, {MODEL_CONFIG['n_mels']} mel bins")
        
    def load_and_prepare_dataset(self):
        """Load drone acoustics dataset and prepare for training/testing"""
        print("\n📊 Loading and preparing dataset...")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ Dataset not found at {self.data_dir}")
            return None, None, None, None
            
        # Load dataset
        X, y = create_dataset_from_drone_acoustics(self.data_dir)
        
        if len(X) == 0:
            print("❌ No data loaded from dataset")
            return None, None, None, None
            
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"✅ Dataset loaded successfully:")
        print(f"   📈 Training samples: {len(X_train)}")
        print(f"   🔍 Testing samples: {len(X_test)}")
        print(f"   📊 Input shape: {X_train.shape[1:]}")
        print(f"   🏷️  Classes: {np.unique(y)}")
        print(f"   📋 Class distribution: {np.bincount(y)}")
        
        return X_train, X_test, y_train, y_test
        
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the DS-CNN+GRU model"""
        print("\n🔧 Building and training DS-CNN+GRU model...")
        
        # Create model
        model = self.model_builder.build_model()
        model = self.model_builder.compile_model(model)
        
        print("📝 Model Architecture:")
        model.summary()
        
        # Calculate model size
        model_size = model.count_params() * 4 / 1024  # Assuming float32, convert to KB
        print(f"📏 Estimated model size: {model_size:.1f} KB")
        
        if model_size > MODEL_CONFIG['model_size_kb']:
            print(f"⚠️  Model exceeds target size of {MODEL_CONFIG['model_size_kb']} KB")
        else:
            print(f"✅ Model fits within {MODEL_CONFIG['model_size_kb']} KB target")
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]
        
        # Train model
        print("\n🚀 Starting training...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=MODEL_CONFIG['batch_size'],
            epochs=min(MODEL_CONFIG['epochs'], 20),  # Limit for testing
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Training completed in {training_time:.1f} seconds")
        
        return model, history
        
    def evaluate_accuracy(self, model, X_test, y_test):
        """Comprehensive accuracy evaluation"""
        print("\n📊 Evaluating model accuracy...")
        
        # Basic evaluation
        test_loss, test_accuracy, test_top_k = model.evaluate(X_test, y_test, verbose=0)
        
        # Detailed predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Model Performance Metrics:")
        print(f"   📈 Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   📉 Test Loss: {test_loss:.4f}")
        print(f"   🏆 Top-K Accuracy: {test_top_k:.3f}")
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[self.sait_class_names[i] for i in np.unique(y_test)]))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'test_loss': test_loss,
            'top_k_accuracy': test_top_k,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
        
    def test_inference_speed(self, model, X_test):
        """Test inference speed and latency"""
        print("\n⚡ Testing inference speed...")
        
        # Single sample inference
        single_sample = X_test[:1]
        
        # Warm up
        for _ in range(5):
            _ = model.predict(single_sample, verbose=0)
            
        # Time multiple inferences
        times = []
        for _ in range(100):
            start = time.time()
            _ = model.predict(single_sample, verbose=0)
            times.append((time.time() - start) * 1000)  # Convert to ms
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"⏱️  Inference Timing (100 runs):")
        print(f"   📊 Average: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   ⚡ Fastest: {min_time:.2f} ms")
        print(f"   🐌 Slowest: {max_time:.2f} ms")
        
        # Check real-time capability
        target_time = 500  # 500ms for real-time audio processing
        if avg_time < target_time:
            print(f"✅ Real-time capable (< {target_time}ms)")
        else:
            print(f"❌ Too slow for real-time (> {target_time}ms)")
            
        return {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time
        }
        
    def test_tflite_conversion(self, model):
        """Test TensorFlow Lite conversion and quantization"""
        print("\n🔄 Testing TensorFlow Lite conversion...")
        
        try:
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Enable optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert
            tflite_model = converter.convert()
            tflite_size = len(tflite_model) / 1024  # KB
            
            print(f"✅ TensorFlow Lite conversion successful")
            print(f"📏 TFLite model size: {tflite_size:.1f} KB")
            
            # Test quantized version
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            try:
                quantized_model = converter.convert()
                quantized_size = len(quantized_model) / 1024
                
                print(f"✅ INT8 quantization successful")
                print(f"📏 Quantized model size: {quantized_size:.1f} KB")
                print(f"🗜️  Size reduction: {((tflite_size - quantized_size) / tflite_size * 100):.1f}%")
                
                # Check if it fits nRF5340 constraints
                if quantized_size <= MODEL_CONFIG['model_size_kb']:
                    print(f"✅ Quantized model fits nRF5340 ({MODEL_CONFIG['model_size_kb']} KB limit)")
                else:
                    print(f"❌ Quantized model too large for nRF5340")
                    
                return {
                    'tflite_size_kb': tflite_size,
                    'quantized_size_kb': quantized_size,
                    'fits_nrf5340': quantized_size <= MODEL_CONFIG['model_size_kb'],
                    'tflite_model': tflite_model,
                    'quantized_model': quantized_model
                }
                
            except Exception as e:
                print(f"❌ INT8 quantization failed: {e}")
                return {
                    'tflite_size_kb': tflite_size,
                    'quantized_size_kb': None,
                    'fits_nrf5340': tflite_size <= MODEL_CONFIG['model_size_kb'],
                    'tflite_model': tflite_model,
                    'quantized_model': None
                }
                
        except Exception as e:
            print(f"❌ TensorFlow Lite conversion failed: {e}")
            return None
            
    def test_audio_preprocessing(self):
        """Test audio preprocessing pipeline"""
        print("\n🎵 Testing audio preprocessing pipeline...")
        
        # Find sample audio files
        sample_files = []
        for class_dir in ['background', 'drone', 'helicopter']:
            class_path = Path(self.data_dir) / 'train' / class_dir
            if class_path.exists():
                files = list(class_path.glob("*.wav"))[:2]  # Take first 2 files
                sample_files.extend([(f, class_dir) for f in files])
                
        if not sample_files:
            print("❌ No sample audio files found")
            return
            
        print(f"🔍 Testing preprocessing on {len(sample_files)} sample files...")
        
        processing_times = []
        
        for audio_file, class_name in sample_files:
            start_time = time.time()
            
            try:
                # Load and preprocess
                audio = self.preprocessor.load_and_resample(str(audio_file))
                mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                
                processing_time = (time.time() - start_time) * 1000  # ms
                processing_times.append(processing_time)
                
                print(f"   ✅ {class_name}: {audio_file.name} -> {mel_spec.shape} ({processing_time:.1f}ms)")
                
            except Exception as e:
                print(f"   ❌ {class_name}: {audio_file.name} -> Error: {e}")
                
        if processing_times:
            avg_processing = np.mean(processing_times)
            print(f"⏱️  Average preprocessing time: {avg_processing:.1f} ms")
            
            if avg_processing < 100:  # 100ms target
                print("✅ Preprocessing is real-time capable")
            else:
                print("⚠️  Preprocessing may be too slow for real-time")
                
    def generate_test_report(self, accuracy_results, speed_results, tflite_results):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("🎯 SAIT_01 TINYML ACCURACY TEST REPORT")
        print("="*60)
        
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   🎯 Overall Accuracy: {accuracy_results['accuracy']:.3f} ({accuracy_results['accuracy']*100:.1f}%)")
        print(f"   📉 Test Loss: {accuracy_results['test_loss']:.4f}")
        print(f"   🏆 Top-K Accuracy: {accuracy_results['top_k_accuracy']:.3f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   🕐 Avg Inference Time: {speed_results['avg_inference_time_ms']:.2f} ms")
        print(f"   ⚡ Min Inference Time: {speed_results['min_inference_time_ms']:.2f} ms")
        print(f"   🐌 Max Inference Time: {speed_results['max_inference_time_ms']:.2f} ms")
        
        if tflite_results:
            print(f"\n🔄 DEPLOYMENT READINESS:")
            print(f"   📏 TFLite Size: {tflite_results['tflite_size_kb']:.1f} KB")
            if tflite_results['quantized_size_kb']:
                print(f"   🗜️  Quantized Size: {tflite_results['quantized_size_kb']:.1f} KB")
            print(f"   📱 nRF5340 Compatible: {'✅ Yes' if tflite_results['fits_nrf5340'] else '❌ No'}")
            
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        if accuracy_results['accuracy'] > 0.8:
            print("   ✅ EXCELLENT accuracy (>80%)")
        elif accuracy_results['accuracy'] > 0.7:
            print("   ✅ GOOD accuracy (>70%)")
        elif accuracy_results['accuracy'] > 0.6:
            print("   ⚠️  FAIR accuracy (>60%)")
        else:
            print("   ❌ POOR accuracy (<60%)")
            
        if speed_results['avg_inference_time_ms'] < 100:
            print("   ✅ FAST inference (<100ms)")
        elif speed_results['avg_inference_time_ms'] < 500:
            print("   ✅ ACCEPTABLE inference (<500ms)")
        else:
            print("   ❌ SLOW inference (>500ms)")
            
        if tflite_results and tflite_results['fits_nrf5340']:
            print("   ✅ DEPLOYMENT READY for nRF5340")
        else:
            print("   ❌ NEEDS OPTIMIZATION for nRF5340")
            
        print("\n" + "="*60)
        
    def run_full_test_suite(self):
        """Run complete accuracy and capability testing"""
        print("🚀 Starting SAIT_01 TinyML Full Test Suite")
        print("="*60)
        
        # 1. Load dataset
        X_train, X_test, y_train, y_test = self.load_and_prepare_dataset()
        if X_train is None:
            print("❌ Failed to load dataset - aborting tests")
            return
            
        # 2. Test preprocessing
        self.test_audio_preprocessing()
        
        # 3. Train model
        model, history = self.train_model(X_train, y_train, X_test, y_test)
        
        # 4. Evaluate accuracy
        accuracy_results = self.evaluate_accuracy(model, X_test, y_test)
        
        # 5. Test inference speed
        speed_results = self.test_inference_speed(model, X_test)
        
        # 6. Test TensorFlow Lite conversion
        tflite_results = self.test_tflite_conversion(model)
        
        # 7. Generate final report
        self.generate_test_report(accuracy_results, speed_results, tflite_results)
        
        return {
            'model': model,
            'history': history,
            'accuracy': accuracy_results,
            'speed': speed_results,
            'tflite': tflite_results
        }

def main():
    """Main testing entry point"""
    print("🎯 SAIT_01 TinyML Audio Recognition Accuracy Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = TinyMLAccuracyTester()
    
    # Run full test suite
    results = tester.run_full_test_suite()
    
    if results:
        print("\n✅ All tests completed successfully!")
        
        # Save results for further analysis
        if results['tflite'] and results['tflite'].get('quantized_model'):
            print("\n💾 Saving optimized model for nRF5340 deployment...")
            with open('sait01_optimized_model.tflite', 'wb') as f:
                f.write(results['tflite']['quantized_model'])
            print("✅ Model saved as 'sait01_optimized_model.tflite'")
    else:
        print("\n❌ Testing failed!")

if __name__ == "__main__":
    main()