#!/usr/bin/env python3
"""
Quick Model Accuracy Test
Test available base models without custom losses
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class QuickTester:
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_quick_test_data(self, samples_per_class=100):
        """Load a small test dataset quickly"""
        print("📊 Loading quick test dataset...")
        
        expanded_dir = Path("expanded_sait01_dataset")
        if not expanded_dir.exists():
            print("❌ Dataset not found")
            return None, None
        
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = expanded_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))[:samples_per_class]
            
            print(f"   Testing {class_name}: {len(audio_files)} samples")
            
            for audio_file in audio_files:
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    X.append(features)
                    y.append(class_idx)
                except Exception:
                    continue
        
        return np.array(X), np.array(y)
    
    def test_model(self, model_path, X_test, y_test):
        """Test a single model"""
        print(f"\n🤖 Testing {model_path}...")
        
        try:
            # Load model
            model = keras.models.load_model(model_path)
            
            # Test inference
            start_time = time.time()
            y_pred = model.predict(X_test, verbose=0)
            inference_time = (time.time() - start_time) / len(X_test) * 1000
            
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            # Model info
            params = model.count_params()
            model_size = os.path.getsize(model_path) / 1024
            
            print(f"   📈 Accuracy: {accuracy*100:.1f}%")
            print(f"   ⚡ Inference: {inference_time:.2f}ms per sample")
            print(f"   📏 Model size: {model_size:.1f} KB")
            print(f"   🔧 Parameters: {params:,}")
            
            # Per-class performance
            report = classification_report(y_test, y_pred_classes, 
                                         target_names=self.class_names, 
                                         output_dict=True)
            
            print(f"   📋 Per-class performance:")
            for i, class_name in enumerate(self.class_names):
                if str(i) in report:
                    precision = report[str(i)]['precision']
                    recall = report[str(i)]['recall']
                    print(f"      {class_name}: {precision*100:.1f}% precision, {recall*100:.1f}% recall")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_classes)
            print(f"   🔍 Confusion Matrix:")
            print(f"      BG  VH  AC")
            for i, true_class in enumerate(['BG', 'VH', 'AC']):
                print(f"   {true_class}: {cm[i]}")
            
            return accuracy, inference_time, report
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, None, None
    
    def test_tflite_model(self, tflite_path, X_test):
        """Test TFLite model"""
        print(f"\n📱 Testing TFLite: {tflite_path}...")
        
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            
            # Test single inference
            sample = X_test[:1].astype(np.float32)
            start_time = time.time()
            
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            
            inference_time = (time.time() - start_time) * 1000
            model_size = os.path.getsize(tflite_path) / 1024
            
            print(f"   📏 Size: {model_size:.1f} KB")
            print(f"   ⚡ Inference: {inference_time:.2f}ms")
            print(f"   ✅ Working correctly")
            
            return model_size, inference_time
            
        except Exception as e:
            print(f"   ❌ TFLite Error: {e}")
            return None, None

def main():
    print("🎯 SAIT_01 QUICK ACCURACY TEST")
    print("=" * 50)
    
    tester = QuickTester()
    
    # Load test data
    X_test, y_test = tester.load_quick_test_data(samples_per_class=50)
    
    if X_test is None:
        print("❌ Could not load test data")
        return
    
    print(f"✅ Loaded {len(X_test)} test samples")
    print(f"📊 Distribution: {np.bincount(y_test)}")
    
    # Test base models
    base_models = [
        'sait01_fixed_model.h5',
        'sait01_production_model.h5', 
        'sait01_quickfix_model.h5',
        'best_sait01_model.h5'
    ]
    
    results = {}
    best_accuracy = 0
    best_model = None
    
    for model_path in base_models:
        if os.path.exists(model_path):
            accuracy, inference_time, report = tester.test_model(model_path, X_test, y_test)
            if accuracy is not None:
                results[model_path] = accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_path
    
    # Test TFLite models
    tflite_models = [
        'sait01_fixed_model.tflite',
        'sait01_cnn_specialist_final.tflite',
        'sait01_multiscale_specialist_final.tflite'
    ]
    
    print(f"\n📱 TFLITE MODEL TESTS:")
    for tflite_path in tflite_models:
        if os.path.exists(tflite_path):
            tester.test_tflite_model(tflite_path, X_test)
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("-" * 30)
    
    if results:
        print(f"🏆 Best Model: {best_model}")
        print(f"📈 Best Accuracy: {best_accuracy*100:.1f}%")
        
        print(f"\n📋 All Results:")
        for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model}: {acc*100:.1f}%")
        
        # 95% target analysis
        print(f"\n🎯 95% TARGET ANALYSIS:")
        if best_accuracy >= 0.95:
            print("   ✅ SUCCESS: 95% target achieved!")
        else:
            gap = 0.95 - best_accuracy
            print(f"   📈 Current best: {best_accuracy*100:.1f}%")
            print(f"   🔧 Gap to 95%: {gap*100:.1f} percentage points")
            print(f"   💡 Next: Battlefield audio integration needed")
    else:
        print("❌ No models could be tested")
    
    print(f"\n✅ QUICK TEST COMPLETE")

if __name__ == "__main__":
    main()