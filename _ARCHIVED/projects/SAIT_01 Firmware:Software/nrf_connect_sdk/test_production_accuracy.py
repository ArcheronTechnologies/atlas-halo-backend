#!/usr/bin/env python3
"""
Test Production SAIT_01 Model Accuracy
Comprehensive accuracy testing with proper dataset loading
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import librosa
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import json
import random

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, MODEL_CONFIG

class ProductionAccuracyTester:
    """Test production model accuracy with real audio data"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_production_model(self):
        """Load the trained production model"""
        model_path = "sait01_production_model.h5"
        tflite_path = "sait01_production_model.tflite"
        
        if not os.path.exists(model_path):
            print("❌ Production model not found")
            return None, None
        
        try:
            model = keras.models.load_model(model_path)
            print(f"✅ Production model loaded: {model.count_params()} parameters")
            
            # Load TFLite model
            tflite_model = None
            if os.path.exists(tflite_path):
                with open(tflite_path, 'rb') as f:
                    tflite_model = f.read()
                print(f"✅ TFLite model loaded: {len(tflite_model) / 1024:.1f} KB")
            
            return model, tflite_model
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    
    def load_test_samples_from_original_dataset(self, max_samples_per_class=100):
        """Load test samples from original dataset for accuracy testing"""
        print("📊 Loading test samples from original dataset...")
        
        X = []
        y = []
        
        # Check original drone acoustics dataset
        data_dir = Path("edth-copenhagen-drone-acoustics/data/raw")
        if data_dir.exists():
            splits = ['val', 'train']  # Use validation first, then supplement with train
            
            for split in splits:
                split_path = data_dir / split
                if not split_path.exists():
                    continue
                
                for class_name in ['background', 'drone', 'helicopter']:
                    if class_name == 'drone':
                        target_class = 1  # vehicle
                    elif class_name == 'helicopter':
                        target_class = 2  # aircraft
                    else:
                        target_class = 0  # background
                    
                    class_dir = split_path / class_name
                    if not class_dir.exists():
                        continue
                    
                    # Get current count for this class
                    current_count = sum(1 for label in y if label == target_class)
                    if current_count >= max_samples_per_class:
                        continue
                    
                    audio_files = list(class_dir.glob("*.wav"))
                    random.shuffle(audio_files)  # Randomize selection
                    
                    needed = min(len(audio_files), max_samples_per_class - current_count)
                    
                    print(f"   Loading {class_name} → {self.class_names[target_class]}: {needed} samples")
                    
                    for audio_file in tqdm(audio_files[:needed], desc=f"Processing {class_name}"):
                        try:
                            # Load and preprocess audio
                            audio = self.preprocessor.load_and_resample(audio_file)
                            features = self.preprocessor.extract_mel_spectrogram(audio)
                            
                            X.append(features)
                            y.append(target_class)
                            
                        except Exception as e:
                            continue
        
        # Also load some samples from expanded dataset if needed
        expanded_dir = Path("expanded_sait01_dataset")
        if expanded_dir.exists() and len(X) < 150:  # Supplement if we don't have enough
            print("📊 Supplementing with expanded dataset samples...")
            
            for class_idx, class_name in enumerate(self.class_names):
                current_count = sum(1 for label in y if label == class_idx)
                if current_count >= max_samples_per_class:
                    continue
                
                class_dir = expanded_dir / class_name
                if not class_dir.exists():
                    continue
                
                # Load original files only (not synthetic)
                audio_files = [f for f in class_dir.glob("*.wav") if 'original_' in f.name]
                random.shuffle(audio_files)
                
                needed = min(len(audio_files), max_samples_per_class - current_count)
                if needed > 0:
                    print(f"   Adding {class_name}: {needed} samples from expanded dataset")
                    
                    for audio_file in audio_files[:needed]:
                        try:
                            audio = self.preprocessor.load_and_resample(audio_file)
                            features = self.preprocessor.extract_mel_spectrogram(audio)
                            
                            X.append(features)
                            y.append(class_idx)
                            
                        except Exception as e:
                            continue
        
        if len(X) == 0:
            print("❌ No test samples loaded")
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Test dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def test_model_accuracy(self, model, tflite_model, X_test, y_test):
        """Test model accuracy on real audio data"""
        print(f"\n🔍 Testing Production Model Accuracy...")
        print("=" * 50)
        
        # Test Keras model
        print("🧠 Testing Keras Model...")
        start_time = time.time()
        
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        keras_time = time.time() - start_time
        keras_inference_time = keras_time / len(X_test) * 1000  # ms per sample
        
        # Calculate accuracy
        keras_accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"📊 Keras Model Results:")
        print(f"   Accuracy: {keras_accuracy:.3f} ({keras_accuracy*100:.1f}%)")
        print(f"   Inference time: {keras_inference_time:.2f}ms per sample")
        print(f"   Total test time: {keras_time:.2f}s")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print(f"\n🔍 Confusion Matrix:")
        print(f"           Predicted")
        print(f"         BG  VH  AC")
        for i, true_class in enumerate(['BG', 'VH', 'AC']):
            print(f"True {true_class}: {cm[i]}")
        
        # Test TFLite model if available
        tflite_accuracy = None
        tflite_inference_time = None
        
        if tflite_model is not None:
            print(f"\n📱 Testing TensorFlow Lite Model...")
            
            try:
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Test on all samples
                tflite_correct = 0
                tflite_times = []
                
                for i in tqdm(range(len(X_test)), desc="TFLite Testing"):
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
                
                tflite_accuracy = tflite_correct / len(X_test)
                tflite_inference_time = np.mean(tflite_times)
                
                print(f"📱 TFLite Model Results:")
                print(f"   Accuracy: {tflite_accuracy:.3f} ({tflite_accuracy*100:.1f}%)")
                print(f"   Inference time: {tflite_inference_time:.2f}ms per sample")
                print(f"   Min inference: {np.min(tflite_times):.2f}ms")
                print(f"   Max inference: {np.max(tflite_times):.2f}ms")
                
            except Exception as e:
                print(f"⚠️  TFLite testing failed: {e}")
        
        return {
            'keras_accuracy': keras_accuracy,
            'tflite_accuracy': tflite_accuracy,
            'keras_inference_ms': keras_inference_time,
            'tflite_inference_ms': tflite_inference_time,
            'test_samples': len(X_test),
            'confusion_matrix': cm
        }
    
    def generate_accuracy_report(self, results):
        """Generate comprehensive accuracy report"""
        print(f"\n🎯 PRODUCTION MODEL ACCURACY REPORT")
        print("=" * 60)
        
        print(f"📊 Test Configuration:")
        print(f"   Test samples: {results['test_samples']}")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Model type: Production CNN")
        
        print(f"\n📈 Accuracy Results:")
        keras_acc = results['keras_accuracy']
        tflite_acc = results['tflite_accuracy']
        
        print(f"   Keras Model:  {keras_acc*100:.1f}%")
        if tflite_acc is not None:
            print(f"   TFLite Model: {tflite_acc*100:.1f}%")
            acc_diff = abs(keras_acc - tflite_acc) * 100
            print(f"   Accuracy Drop: {acc_diff:.1f}%")
        
        print(f"\n⚡ Performance:")
        print(f"   Keras Inference:  {results['keras_inference_ms']:.2f}ms")
        if results['tflite_inference_ms'] is not None:
            print(f"   TFLite Inference: {results['tflite_inference_ms']:.2f}ms")
        
        print(f"\n🎯 Assessment:")
        if keras_acc >= 0.85:
            status = "🏆 EXCELLENT"
            verdict = "Production ready"
        elif keras_acc >= 0.70:
            status = "✅ GOOD"
            verdict = "Acceptable for deployment"
        elif keras_acc >= 0.50:
            status = "📈 MODERATE"
            verdict = "Needs optimization"
        else:
            status = "⚠️ POOR"
            verdict = "Requires significant improvement"
        
        print(f"   Status: {status}")
        print(f"   Verdict: {verdict}")
        
        # Deployment readiness
        print(f"\n🚀 Deployment Readiness:")
        model_size = os.path.getsize("sait01_production_model.tflite") / 1024 if os.path.exists("sait01_production_model.tflite") else 0
        
        checks = [
            ("Accuracy", f"{keras_acc*100:.1f}%", "✅" if keras_acc >= 0.70 else "❌"),
            ("Model Size", f"{model_size:.1f}KB", "✅" if model_size < 200 else "❌"),
            ("Inference Speed", f"{results['keras_inference_ms']:.1f}ms", "✅" if results['keras_inference_ms'] < 50 else "❌"),
            ("TFLite Compatible", "Yes" if tflite_acc is not None else "No", "✅" if tflite_acc is not None else "❌")
        ]
        
        for check_name, value, status in checks:
            print(f"   {status} {check_name}: {value}")
        
        # Final recommendation
        ready_count = sum(1 for _, _, status in checks if status == "✅")
        total_checks = len(checks)
        
        print(f"\n📋 Final Assessment: {ready_count}/{total_checks} criteria met")
        
        if ready_count == total_checks:
            print("🚀 READY FOR PRODUCTION DEPLOYMENT!")
        elif ready_count >= 3:
            print("📈 READY FOR TESTING DEPLOYMENT")
        else:
            print("🔧 NEEDS FURTHER OPTIMIZATION")

def main():
    """Main testing execution"""
    print("🎯 SAIT_01 Production Model Accuracy Testing")
    print("=" * 70)
    
    # Initialize tester
    tester = ProductionAccuracyTester()
    
    # Load production model
    model, tflite_model = tester.load_production_model()
    if model is None:
        return
    
    # Load test data
    X_test, y_test = tester.load_test_samples_from_original_dataset(max_samples_per_class=50)
    if X_test is None:
        return
    
    # Test accuracy
    results = tester.test_model_accuracy(model, tflite_model, X_test, y_test)
    
    # Generate report
    tester.generate_accuracy_report(results)
    
    print(f"\n✅ Accuracy testing completed!")

if __name__ == "__main__":
    main()