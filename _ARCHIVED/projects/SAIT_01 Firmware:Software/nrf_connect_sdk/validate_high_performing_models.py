#!/usr/bin/env python3
"""
Comprehensive validation of all high-performing models
Tests the models that claimed 95%+ accuracy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class HighPerformingModelValidator:
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
        # Models to validate
        self.models_to_test = {
            'ultimate_95_plus': {
                'path': 'sait01_ultimate_95_plus_model.h5',
                'claimed_accuracy': 0.9638,
                'description': 'Ultimate 95%+ System with ensemble'
            },
            'battlefield': {
                'path': 'sait01_battlefield_model.h5', 
                'claimed_accuracy': 0.9611,
                'description': 'Battlefield Model for combat scenarios'
            },
            'balanced_multiclass_best': {
                'path': 'balanced_multiclass_best.h5',
                'claimed_accuracy': 0.9122,
                'description': 'Best balanced multiclass checkpoint'
            }
        }
    
    def load_validation_dataset(self, samples_per_class=400):
        """Load fresh validation dataset"""
        print(f"📊 Loading validation dataset...")
        dataset_dir = Path("massive_enhanced_dataset")
        audio_files = []
        labels = []
        
        # Use different samples for validation (skip first 1000 to avoid overlap)
        start_offset = 1000
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = dataset_dir / class_name
            if class_dir.exists():
                files = list(class_dir.glob("*.wav"))
                np.random.shuffle(files)
                
                # Use different samples than any previous training/testing
                validation_files = files[start_offset:start_offset + samples_per_class]
                print(f"   {class_name}: {len(validation_files)} validation samples")
                
                for audio_file in validation_files:
                    audio_files.append(audio_file)
                    labels.append(class_idx)
        
        print(f"Total validation samples: {len(audio_files)}")
        return audio_files, labels
    
    def extract_standard_features(self, audio_files, labels):
        """Extract standard mel-spectrogram features for validation"""
        print("🔬 Extracting standard mel-spectrogram features...")
        
        X_features = []
        y_labels = []
        
        for i, (audio_file, label) in enumerate(zip(audio_files, labels)):
            if i % 100 == 0:
                print(f"   Processing {i}/{len(audio_files)} samples...")
            
            try:
                # Load and extract standard features
                audio = self.preprocessor.load_and_resample(audio_file)
                mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                
                # Ensure correct shape for model input
                if len(mel_spec.shape) == 3 and mel_spec.shape[-1] == 1:
                    # Shape is (freq, time, 1) - correct
                    X_features.append(mel_spec)
                elif len(mel_spec.shape) == 2:
                    # Shape is (freq, time) - add channel dimension
                    X_features.append(np.expand_dims(mel_spec, axis=-1))
                else:
                    print(f"⚠️  Unexpected shape: {mel_spec.shape}")
                    continue
                
                y_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        X = np.array(X_features)
        y = np.array(y_labels)
        
        print(f"✅ Feature extraction complete:")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        return X, y
    
    def validate_model(self, model_name, model_info, X_val, y_val):
        """Validate a single model"""
        print(f"\n🔍 VALIDATING: {model_info['description']}")
        print(f"   Model: {model_info['path']}")
        print(f"   Claimed accuracy: {model_info['claimed_accuracy']*100:.2f}%")
        print("-" * 60)
        
        # Check if model exists
        if not os.path.exists(model_info['path']):
            print(f"❌ Model file not found: {model_info['path']}")
            return None
        
        # Load model
        try:
            model = keras.models.load_model(model_info['path'], compile=False)
            
            # Try to compile with standard settings
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {model.count_params():,}")
            print(f"   Input shape: {model.input_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
        
        # Validate input shape compatibility
        expected_shape = model.input_shape[1:]  # Remove batch dimension
        actual_shape = X_val.shape[1:]
        
        if expected_shape != actual_shape:
            print(f"⚠️  Shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Actual: {actual_shape}")
            
            # Try to reshape if possible
            if len(expected_shape) == 3 and len(actual_shape) == 3:
                if expected_shape[2] == actual_shape[2]:  # Same channels
                    # Try to resize
                    print(f"   Attempting to resize features...")
                    X_resized = []
                    for i in range(len(X_val)):
                        feature = X_val[i]
                        # Resize using tensorflow
                        resized = tf.image.resize(feature, expected_shape[:2]).numpy()
                        X_resized.append(resized)
                    X_val = np.array(X_resized)
                    print(f"   Resized to: {X_val.shape}")
                else:
                    print(f"❌ Cannot reshape - incompatible channels")
                    return None
            else:
                print(f"❌ Cannot reshape - incompatible dimensions")
                return None
        
        # Make predictions
        print(f"\n🎯 Running inference...")
        start_time = time.time()
        
        try:
            y_pred_proba = model.predict(X_val, verbose=1, batch_size=32)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            inference_time = time.time() - start_time
            print(f"⏱️  Inference time: {inference_time:.2f}s ({inference_time/len(X_val)*1000:.1f}ms per sample)")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
        
        # Calculate metrics
        overall_accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"   Claimed Accuracy: {model_info['claimed_accuracy']*100:.2f}%")
        
        accuracy_diff = overall_accuracy - model_info['claimed_accuracy']
        if accuracy_diff >= 0:
            print(f"   Difference: +{accuracy_diff*100:.2f}% ✅")
        else:
            print(f"   Difference: {accuracy_diff*100:.2f}% ❌")
        
        # Per-class analysis
        class_accuracies = {}
        class_metrics = {}
        
        print(f"\n📈 Per-class performance:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y_val == i
            
            if np.sum(class_mask) > 0:
                # Accuracy
                class_acc = accuracy_score(y_val[class_mask], y_pred[class_mask])
                class_accuracies[class_name] = class_acc
                
                # Precision, Recall, F1
                true_positives = np.sum((y_val == i) & (y_pred == i))
                false_positives = np.sum((y_val != i) & (y_pred == i))
                false_negatives = np.sum((y_val == i) & (y_pred != i))
                
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                class_metrics[class_name] = {
                    'accuracy': class_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'samples': int(np.sum(class_mask))
                }
                
                status = '✅' if class_acc >= 0.95 else '❌'
                print(f"{status} {class_name}:")
                print(f"     Accuracy: {class_acc:.4f} ({class_acc*100:.2f}%)")
                print(f"     Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                print(f"     Samples: {np.sum(class_mask)}")
        
        # Confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print("     Predicted:")
        print("        BG    VEH   AIR")
        for i, class_name in enumerate(['BG', 'VEH', 'AIR']):
            print(f"{class_name:>3}: {cm[i]}")
        
        # Confidence analysis
        confidence_scores = np.max(y_pred_proba, axis=1)
        print(f"\n🎯 Prediction Confidence:")
        print(f"   Mean: {np.mean(confidence_scores):.4f}")
        print(f"   Min:  {np.min(confidence_scores):.4f}")
        print(f"   Max:  {np.max(confidence_scores):.4f}")
        
        # Check 95% target
        meets_target = overall_accuracy >= 0.95 and all(acc >= 0.95 for acc in class_accuracies.values())
        print(f"\n🎯 95% TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_val, y_pred, target_names=self.class_names, digits=4))
        
        return {
            'model_name': model_name,
            'model_path': model_info['path'],
            'claimed_accuracy': model_info['claimed_accuracy'],
            'actual_accuracy': float(overall_accuracy),
            'accuracy_difference': float(accuracy_diff),
            'meets_95_target': meets_target,
            'class_accuracies': {k: float(v) for k, v in class_accuracies.items()},
            'class_metrics': {k: {mk: float(mv) if isinstance(mv, (int, float, np.number)) else mv 
                                 for mk, mv in v.items()} for k, v in class_metrics.items()},
            'confusion_matrix': cm.tolist(),
            'confidence_stats': {
                'mean': float(np.mean(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            },
            'inference_time_per_sample_ms': float(inference_time/len(X_val)*1000),
            'model_parameters': int(model.count_params())
        }
    
    def run_comprehensive_validation(self):
        """Run validation on all high-performing models"""
        print("🚀 COMPREHENSIVE HIGH-PERFORMING MODEL VALIDATION")
        print("=" * 80)
        
        # Load validation dataset
        audio_files, labels = self.load_validation_dataset()
        
        if len(audio_files) == 0:
            print("❌ No validation data found!")
            return None
        
        # Extract features
        X_val, y_val = self.extract_standard_features(audio_files, labels)
        
        # Validate each model
        results = {}
        
        for model_name, model_info in self.models_to_test.items():
            try:
                result = self.validate_model(model_name, model_info, X_val, y_val)
                if result:
                    results[model_name] = result
            except Exception as e:
                print(f"❌ Failed to validate {model_name}: {e}")
                continue
        
        # Summary comparison
        print(f"\n🏆 VALIDATION SUMMARY")
        print("=" * 80)
        
        successful_models = []
        
        for model_name, result in results.items():
            print(f"\n📊 {result['model_name'].upper()}:")
            print(f"   Claimed: {result['claimed_accuracy']*100:.2f}%")
            print(f"   Actual:  {result['actual_accuracy']*100:.2f}%")
            print(f"   Target:  {'✅ ACHIEVED' if result['meets_95_target'] else '❌ NOT MET'}")
            
            if result['meets_95_target']:
                successful_models.append(result)
        
        print(f"\n🎯 MODELS MEETING 95% TARGET: {len(successful_models)}/{len(results)}")
        
        if successful_models:
            # Sort by actual accuracy
            successful_models.sort(key=lambda x: x['actual_accuracy'], reverse=True)
            best_model = successful_models[0]
            
            print(f"\n🏅 BEST PERFORMING MODEL:")
            print(f"   Model: {best_model['model_name']}")
            print(f"   Accuracy: {best_model['actual_accuracy']*100:.2f}%")
            print(f"   Parameters: {best_model['model_parameters']:,}")
            print(f"   Inference: {best_model['inference_time_per_sample_ms']:.1f}ms/sample")
        
        # Save comprehensive results
        validation_results = {
            'validation_date': str(np.datetime64('today')),
            'validation_samples': len(y_val),
            'models_tested': len(results),
            'models_meeting_target': len(successful_models),
            'results': results
        }
        
        with open('high_performing_model_validation.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved: high_performing_model_validation.json")
        
        return validation_results

def main():
    validator = HighPerformingModelValidator()
    results = validator.run_comprehensive_validation()
    
    if results and results['models_meeting_target'] > 0:
        print(f"\n✅ SUCCESS: {results['models_meeting_target']} model(s) confirmed to meet 95% target!")
        return 0
    else:
        print(f"\n❌ WARNING: No models confirmed to meet 95% target!")
        return 1

if __name__ == "__main__":
    exit(main())