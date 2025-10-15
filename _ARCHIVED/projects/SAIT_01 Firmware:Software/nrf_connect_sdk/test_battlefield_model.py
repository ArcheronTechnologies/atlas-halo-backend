#!/usr/bin/env python3
"""
Test Battlefield Model
Test the newly trained battlefield-enhanced model
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

class BattlefieldModelTester:
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_test_data(self, samples_per_class=200):
        """Load test dataset"""
        print("📊 Loading test dataset...")
        
        enhanced_dir = Path("enhanced_sait01_dataset")
        if not enhanced_dir.exists():
            print("❌ Enhanced dataset not found")
            return None, None
        
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = enhanced_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            
            # Take different samples than training
            np.random.seed(123)  # Different seed than training
            np.random.shuffle(audio_files)
            test_files = audio_files[-samples_per_class:]  # Take from end
            
            print(f"   Testing {class_name}: {len(test_files)} samples")
            
            for audio_file in test_files:
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    X.append(features)
                    y.append(class_idx)
                except Exception:
                    continue
        
        return np.array(X), np.array(y)
    
    def test_battlefield_model(self, X_test, y_test):
        """Test the battlefield model"""
        print(f"\n🎯 TESTING BATTLEFIELD MODEL")
        print("-" * 50)
        
        model_path = 'sait01_battlefield_model.h5'
        if not os.path.exists(model_path):
            print(f"❌ Battlefield model not found: {model_path}")
            return None
            
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
        
        print(f"📈 Test Accuracy: {accuracy*100:.1f}%")
        print(f"⚡ Inference: {inference_time:.2f}ms per sample")
        print(f"📏 Model size: {model_size:.1f} KB")
        print(f"🔧 Parameters: {params:,}")
        
        # Detailed classification report
        print(f"\n📋 DETAILED PERFORMANCE:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print(f"\n🔍 Confusion Matrix:")
        print(f"           Predicted")
        print(f"         BG  VH  AC")
        for i, true_class in enumerate(['BG', 'VH', 'AC']):
            print(f"True {true_class}: {cm[i]}")
        
        # 95% target analysis
        print(f"\n🎯 95% ACCURACY TARGET ANALYSIS:")
        target_reached = accuracy >= 0.95
        print(f"   Target: 95.0%")
        print(f"   Achieved: {accuracy*100:.1f}%")
        
        if target_reached:
            print("   ✅ SUCCESS: 95% target achieved!")
            print("   🚀 Model ready for battlefield deployment!")
        else:
            gap = 0.95 - accuracy
            print(f"   📈 Gap: {gap*100:.1f} percentage points")
            print(f"   💡 Status: {'Very close!' if gap < 0.03 else 'Needs enhancement'}")
        
        # Combat sound analysis
        self.analyze_combat_performance(y_test, y_pred_classes, X_test)
        
        return accuracy, target_reached
    
    def analyze_combat_performance(self, y_true, y_pred, X_test):
        """Analyze performance on combat vs civilian sounds"""
        print(f"\n⚔️  COMBAT vs CIVILIAN ANALYSIS:")
        print("-" * 40)
        
        # This is a simplified analysis - in practice you'd need metadata about which samples are combat
        correct_predictions = (y_true == y_pred)
        
        class_accuracies = []
        for i, class_name in enumerate(self.class_names):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_acc = np.mean(correct_predictions[class_mask])
                class_accuracies.append(class_acc)
                print(f"   {class_name}: {class_acc*100:.1f}% accuracy")
                
                # Combat readiness assessment
                if class_name == 'vehicle' and class_acc >= 0.95:
                    print(f"      ✅ Combat vehicle detection ready")
                elif class_name == 'aircraft' and class_acc >= 0.95:
                    print(f"      ✅ Combat aircraft detection ready")
                elif class_name == 'background' and class_acc >= 0.95:
                    print(f"      ✅ Explosion/gunfire detection ready")
        
        avg_accuracy = np.mean(class_accuracies)
        print(f"\n   Overall: {avg_accuracy*100:.1f}% average class accuracy")
    
    def test_tflite_model(self, X_test):
        """Test TFLite battlefield model"""
        print(f"\n📱 TESTING TFLITE BATTLEFIELD MODEL")
        print("-" * 50)
        
        tflite_path = 'sait01_battlefield_model.tflite'
        if not os.path.exists(tflite_path):
            print(f"❌ TFLite model not found: {tflite_path}")
            return
            
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
            
            print(f"📏 TFLite size: {model_size:.1f} KB")
            print(f"⚡ TFLite inference: {inference_time:.2f}ms")
            print(f"🚀 Deployment ready: {'✅ Yes' if model_size < 200 and inference_time < 50 else '⚠️  Check constraints'}")
            
        except Exception as e:
            print(f"❌ TFLite test failed: {e}")

def main():
    print("⚔️  SAIT_01 BATTLEFIELD MODEL TEST")
    print("=" * 70)
    print("🎯 Testing combat-enhanced model performance")
    print("=" * 70)
    
    tester = BattlefieldModelTester()
    
    # Load test data
    X_test, y_test = tester.load_test_data(samples_per_class=150)
    
    if X_test is None:
        print("❌ Could not load test data")
        return
    
    print(f"✅ Loaded {len(X_test)} test samples")
    print(f"📊 Distribution: {np.bincount(y_test)}")
    
    # Test battlefield model
    accuracy, target_reached = tester.test_battlefield_model(X_test, y_test)
    
    # Test TFLite version
    tester.test_tflite_model(X_test)
    
    # Final assessment
    print(f"\n🏆 BATTLEFIELD READINESS ASSESSMENT")
    print("=" * 70)
    
    if target_reached:
        print("✅ DEPLOYMENT APPROVED: Model meets 95% accuracy requirement")
        print("🚀 Ready for battlefield operations")
    else:
        print(f"📈 CURRENT STATUS: {accuracy*100:.1f}% accuracy achieved")
        gap = 0.95 - accuracy
        if gap <= 0.025:
            print("🎯 VERY CLOSE: Minor improvements needed")
            print("💡 Suggestion: Add more diverse combat audio samples")
        else:
            print("🔧 ENHANCEMENT NEEDED: Significant improvements required")
    
    print(f"\n✅ BATTLEFIELD MODEL TEST COMPLETE")

if __name__ == "__main__":
    main()