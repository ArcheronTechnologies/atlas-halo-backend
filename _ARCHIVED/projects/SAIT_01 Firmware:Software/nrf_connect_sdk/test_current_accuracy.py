#!/usr/bin/env python3
"""
Test Current Model Accuracy
Evaluate the trained ensemble models and individual specialists
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, MODEL_CONFIG

class ModelAccuracyTester:
    """Test and evaluate current model accuracy"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_test_dataset(self, max_per_class=500):
        """Load a subset of the dataset for testing"""
        print("📊 Loading test dataset...")
        
        expanded_dir = Path("expanded_sait01_dataset")
        if not expanded_dir.exists():
            print("❌ Expanded dataset not found")
            return None, None
        
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = expanded_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            
            # Limit samples for quick testing
            if len(audio_files) > max_per_class:
                np.random.shuffle(audio_files)
                audio_files = audio_files[:max_per_class]
            
            print(f"   Loading {class_name}: {len(audio_files)} samples")
            
            for audio_file in audio_files:
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    X.append(features)
                    y.append(class_idx)
                except Exception:
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Test dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def test_individual_models(self, X_test, y_test):
        """Test individual specialist models"""
        print(f"\n🔍 TESTING INDIVIDUAL MODELS")
        print("-" * 50)
        
        model_files = [
            ('CNN_Specialist', 'sait01_cnn_specialist_final.h5'),
            ('MultiScale_Specialist', 'sait01_multiscale_specialist_final.h5'),
            ('Temporal_Specialist', 'sait01_temporal_specialist_final.h5')
        ]
        
        results = {}
        
        for name, filename in model_files:
            if os.path.exists(filename):
                print(f"\n🤖 Testing {name}...")
                try:
                    model = keras.models.load_model(filename)
                    
                    # Predict
                    start_time = time.time()
                    y_pred = model.predict(X_test, verbose=0)
                    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
                    
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    accuracy = accuracy_score(y_test, y_pred_classes)
                    
                    print(f"   📈 Accuracy: {accuracy*100:.1f}%")
                    print(f"   ⚡ Inference: {inference_time:.2f}ms per sample")
                    
                    # Detailed metrics
                    report = classification_report(y_test, y_pred_classes, 
                                                 target_names=self.class_names, 
                                                 output_dict=True)
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'inference_time_ms': inference_time,
                        'predictions': y_pred,
                        'report': report
                    }
                    
                    # Show per-class performance
                    for i, class_name in enumerate(self.class_names):
                        if str(i) in report:
                            precision = report[str(i)]['precision']
                            recall = report[str(i)]['recall']
                            print(f"      {class_name}: {precision*100:.1f}% precision, {recall*100:.1f}% recall")
                    
                except Exception as e:
                    print(f"   ❌ Error loading {name}: {e}")
            else:
                print(f"   ⚠️  {name} model not found: {filename}")
        
        return results
    
    def test_ensemble(self, individual_results, X_test, y_test):
        """Test ensemble performance"""
        print(f"\n🤝 TESTING ENSEMBLE PERFORMANCE")
        print("-" * 50)
        
        if len(individual_results) < 2:
            print("❌ Need at least 2 models for ensemble testing")
            return None
        
        # Collect predictions
        predictions = []
        model_names = []
        accuracies = []
        
        for name, result in individual_results.items():
            predictions.append(result['predictions'])
            model_names.append(name)
            accuracies.append(result['accuracy'])
        
        # Simple average ensemble
        print("📊 Simple Average Ensemble:")
        avg_pred = np.mean(predictions, axis=0)
        avg_classes = np.argmax(avg_pred, axis=1)
        avg_accuracy = accuracy_score(y_test, avg_classes)
        print(f"   📈 Accuracy: {avg_accuracy*100:.1f}%")
        
        # Weighted ensemble (weight by individual accuracy)
        print("\n⚖️  Weighted Ensemble:")
        weights = np.array(accuracies)
        weights = weights / np.sum(weights)
        
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        weighted_classes = np.argmax(weighted_pred, axis=1)
        weighted_accuracy = accuracy_score(y_test, weighted_classes)
        
        print(f"   📈 Accuracy: {weighted_accuracy*100:.1f}%")
        print(f"   ⚖️  Weights: {[f'{name}: {w:.3f}' for name, w in zip(model_names, weights)]}")
        
        # Detailed ensemble analysis
        print(f"\n📋 ENSEMBLE CLASSIFICATION REPORT:")
        print(classification_report(y_test, weighted_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, weighted_classes)
        print(f"\n🔍 Ensemble Confusion Matrix:")
        print(f"           Predicted")
        print(f"         BG  VH  AC")
        for i, true_class in enumerate(['BG', 'VH', 'AC']):
            print(f"True {true_class}: {cm[i]}")
        
        # Check 95% target
        target_reached = weighted_accuracy >= 0.95
        print(f"\n🎯 95% ACCURACY TARGET:")
        print(f"   Target: 95.0%")
        print(f"   Achieved: {weighted_accuracy*100:.1f}%")
        print(f"   Status: {'✅ TARGET REACHED!' if target_reached else '❌ Need improvements'}")
        
        if not target_reached:
            gap = 0.95 - weighted_accuracy
            print(f"   📈 Gap: {gap*100:.1f} percentage points")
        
        return {
            'simple_ensemble_accuracy': avg_accuracy,
            'weighted_ensemble_accuracy': weighted_accuracy,
            'weights': weights,
            'model_names': model_names,
            'target_reached': target_reached
        }
    
    def test_tflite_models(self, X_test, y_test):
        """Test TFLite model performance"""
        print(f"\n📱 TESTING TFLITE MODELS")
        print("-" * 50)
        
        tflite_files = [
            'sait01_cnn_specialist_final.tflite',
            'sait01_multiscale_specialist_final.tflite', 
            'sait01_temporal_specialist_final.tflite'
        ]
        
        for filename in tflite_files:
            if os.path.exists(filename):
                print(f"\n🔧 Testing {filename}...")
                try:
                    # Load TFLite model
                    interpreter = tf.lite.Interpreter(model_path=filename)
                    interpreter.allocate_tensors()
                    
                    # Get input and output details
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    # Test inference time
                    sample = X_test[:1].astype(np.float32)
                    start_time = time.time()
                    
                    interpreter.set_tensor(input_details[0]['index'], sample)
                    interpreter.invoke()
                    
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Get model size
                    model_size = os.path.getsize(filename) / 1024  # KB
                    
                    print(f"   📏 Model size: {model_size:.1f} KB")
                    print(f"   ⚡ Inference: {inference_time:.2f}ms")
                    print(f"   ✅ TFLite conversion working")
                    
                except Exception as e:
                    print(f"   ❌ Error testing TFLite: {e}")
    
    def generate_summary_report(self, individual_results, ensemble_results):
        """Generate comprehensive summary report"""
        print(f"\n📊 COMPREHENSIVE ACCURACY SUMMARY")
        print("=" * 70)
        
        print(f"\n🤖 INDIVIDUAL MODEL PERFORMANCE:")
        for name, result in individual_results.items():
            accuracy = result['accuracy']
            inference_time = result['inference_time_ms']
            print(f"   {name:<25}: {accuracy*100:>6.1f}% accuracy, {inference_time:>6.2f}ms")
        
        if ensemble_results:
            print(f"\n🤝 ENSEMBLE PERFORMANCE:")
            simple_acc = ensemble_results['simple_ensemble_accuracy']
            weighted_acc = ensemble_results['weighted_ensemble_accuracy'] 
            print(f"   Simple Average Ensemble  : {simple_acc*100:>6.1f}% accuracy")
            print(f"   Weighted Ensemble        : {weighted_acc*100:>6.1f}% accuracy")
            
            print(f"\n🎯 95% TARGET STATUS:")
            if ensemble_results['target_reached']:
                print("   ✅ SUCCESS: 95% accuracy target achieved!")
                print("   🚀 Model ready for battlefield deployment!")
            else:
                gap = 0.95 - weighted_acc
                print(f"   📈 PROGRESS: {weighted_acc*100:.1f}% achieved")
                print(f"   🔧 GAP: {gap*100:.1f}% improvement needed")
                print(f"   💡 NEXT: Implement battlefield audio enhancement")
        
        print(f"\n📋 RECOMMENDATIONS:")
        best_individual = max(individual_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"   • Best individual model: {best_individual[0]} ({best_individual[1]['accuracy']*100:.1f}%)")
        
        if ensemble_results and ensemble_results['weighted_ensemble_accuracy'] > best_individual[1]['accuracy']:
            improvement = (ensemble_results['weighted_ensemble_accuracy'] - best_individual[1]['accuracy']) * 100
            print(f"   • Ensemble improves by: +{improvement:.1f}% over best individual")
            print(f"   • Recommendation: Use ensemble for production")
        else:
            print(f"   • Recommendation: Focus on improving individual models")
        
        if not (ensemble_results and ensemble_results['target_reached']):
            print(f"   • Next steps: Add battlefield audio (gunshots, explosions)")
            print(f"   • Priority: Vehicle detection enhancement")

def main():
    """Run comprehensive accuracy testing"""
    print("🎯 SAIT_01 CURRENT MODEL ACCURACY TEST")
    print("=" * 70)
    
    tester = ModelAccuracyTester()
    
    # Load test dataset
    X_test, y_test = tester.load_test_dataset(max_per_class=300)
    
    if X_test is None:
        print("❌ Could not load test dataset")
        return
    
    # Test individual models
    individual_results = tester.test_individual_models(X_test, y_test)
    
    # Test ensemble
    ensemble_results = tester.test_ensemble(individual_results, X_test, y_test) if individual_results else None
    
    # Test TFLite models
    tester.test_tflite_models(X_test, y_test)
    
    # Generate summary
    tester.generate_summary_report(individual_results, ensemble_results)
    
    print(f"\n✅ ACCURACY TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()