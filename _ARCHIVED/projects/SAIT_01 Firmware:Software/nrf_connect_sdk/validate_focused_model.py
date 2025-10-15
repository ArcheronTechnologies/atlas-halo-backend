#!/usr/bin/env python3
"""
Comprehensive validation of the focused spectrographic model
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
import matplotlib.pyplot as plt

sys.path.append('.')
from focused_spectrographic_model import FocusedSpectrogramicModel

def validate_focused_model():
    print("🔍 COMPREHENSIVE FOCUSED MODEL VALIDATION")
    print("=" * 70)
    
    # Check if model exists
    model_path = "focused_spectrographic_best.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"📊 Loading model: {model_path}")
    
    # Load model
    try:
        model = keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Initialize model builder for feature extraction
    model_builder = FocusedSpectrogramicModel()
    
    # Load test dataset
    print("\n📊 Loading validation dataset...")
    dataset_dir = Path("massive_enhanced_dataset")
    audio_files = []
    labels = []
    
    # Use different samples for validation (higher indices)
    samples_per_class = 300  # Separate validation set
    start_offset = 800  # Skip first 800 used for training
    
    for class_idx, class_name in enumerate(model_builder.class_names):
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            np.random.shuffle(files)
            
            # Use different samples than training
            validation_files = files[start_offset:start_offset + samples_per_class]
            print(f"   {class_name}: {len(validation_files)} validation samples")
            
            for audio_file in validation_files:
                audio_files.append(audio_file)
                labels.append(class_idx)
    
    print(f"Total validation samples: {len(audio_files)}")
    
    if len(audio_files) == 0:
        print("❌ No validation files found")
        return None
    
    # Extract features for validation
    print("\n🔬 Extracting features for validation...")
    X_features, y = model_builder.prepare_focused_data(audio_files, labels, max_samples=len(audio_files))
    
    # Make predictions
    print("\n🎯 Running inference...")
    y_pred_proba = model.predict(X_features, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate comprehensive metrics
    overall_accuracy = accuracy_score(y, y_pred)
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Per-class analysis
    class_accuracies = {}
    class_metrics = {}
    
    print(f"\n📈 Per-class performance:")
    for i, class_name in enumerate(model_builder.class_names):
        class_mask = y == i
        class_pred_mask = y_pred == i
        
        if np.sum(class_mask) > 0:
            # Accuracy
            class_acc = accuracy_score(y[class_mask], y_pred[class_mask])
            class_accuracies[class_name] = class_acc
            
            # Precision, Recall, F1
            true_positives = np.sum((y == i) & (y_pred == i))
            false_positives = np.sum((y != i) & (y_pred == i))
            false_negatives = np.sum((y == i) & (y_pred != i))
            
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
        else:
            print(f"⚠️  {class_name}: No validation samples")
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print("     Predicted:")
    print("        BG    VEH   AIR")
    for i, class_name in enumerate(['BG', 'VEH', 'AIR']):
        print(f"{class_name:>3}: {cm[i]}")
    
    # Calculate confidence statistics
    print(f"\n🎯 Prediction Confidence Analysis:")
    confidence_scores = np.max(y_pred_proba, axis=1)
    print(f"   Mean confidence: {np.mean(confidence_scores):.4f}")
    print(f"   Min confidence: {np.min(confidence_scores):.4f}")
    print(f"   Max confidence: {np.max(confidence_scores):.4f}")
    
    # Low confidence predictions (potential issues)
    low_confidence_threshold = 0.7
    low_confidence_mask = confidence_scores < low_confidence_threshold
    if np.sum(low_confidence_mask) > 0:
        print(f"   Low confidence predictions (<{low_confidence_threshold}): {np.sum(low_confidence_mask)}")
        low_conf_accuracy = accuracy_score(y[low_confidence_mask], y_pred[low_confidence_mask])
        print(f"   Low confidence accuracy: {low_conf_accuracy:.4f}")
    
    # Check 95% target
    meets_target = overall_accuracy >= 0.95 and all(acc >= 0.95 for acc in class_accuracies.values())
    print(f"\n🎯 95% ACCURACY TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
    
    if not meets_target:
        print(f"\n💡 Analysis:")
        if overall_accuracy < 0.95:
            gap = 0.95 - overall_accuracy
            print(f"   Overall accuracy gap: {gap:.3f} ({gap*100:.1f}%)")
        
        for class_name, acc in class_accuracies.items():
            if acc < 0.95:
                gap = 0.95 - acc
                print(f"   {class_name} accuracy gap: {gap:.3f} ({gap*100:.1f}%)")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y, y_pred, target_names=model_builder.class_names, digits=4))
    
    # Save comprehensive results
    results = {
        'validation_date': str(np.datetime64('today')),
        'model_file': model_path,
        'model_parameters': int(model.count_params()),
        'validation_samples': len(y),
        'overall_accuracy': float(overall_accuracy),
        'meets_95_target': meets_target,
        'class_metrics': {k: {mk: float(mv) if isinstance(mv, (int, float, np.number)) else mv 
                             for mk, mv in v.items()} for k, v in class_metrics.items()},
        'confusion_matrix': cm.tolist(),
        'confidence_stats': {
            'mean': float(np.mean(confidence_scores)),
            'min': float(np.min(confidence_scores)),
            'max': float(np.max(confidence_scores)),
            'low_confidence_count': int(np.sum(low_confidence_mask)),
            'low_confidence_accuracy': float(low_conf_accuracy) if np.sum(low_confidence_mask) > 0 else 1.0
        }
    }
    
    with open('comprehensive_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved: comprehensive_validation_results.json")
    
    return results

if __name__ == "__main__":
    results = validate_focused_model()
    
    if results:
        if results['meets_95_target']:
            print("\n🎉 Model validation successful - 95% target achieved!")
        else:
            print(f"\n⚠️ Model needs improvement - {results['overall_accuracy']*100:.2f}% vs 95% target")
    else:
        print("\n❌ Validation failed")