#!/usr/bin/env python3
"""
Cycle Validator - Test models from each training cycle
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class CycleValidator:
    """Validate models from different training cycles"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_test_data(self, samples_per_class=200):
        """Load fresh test data"""
        print(f"📊 Loading test data ({samples_per_class} per class)...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            np.random.shuffle(audio_files)
            
            count = 0
            for audio_file in audio_files[:samples_per_class]:
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    if len(features.shape) == 2:
                        features = np.expand_dims(features, axis=-1)
                    
                    X.append(features)
                    y.append(class_idx)
                    count += 1
                    
                except Exception as e:
                    continue
            
            print(f"   {class_name}: {count} samples")
        
        return np.array(X), np.array(y)
    
    def validate_model(self, model_path, cycle_name):
        """Validate a specific model"""
        print(f"\\n🔍 VALIDATING {cycle_name.upper()}")
        print("=" * 50)
        
        try:
            model = keras.models.load_model(model_path, compile=False)
            print(f"✅ Loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {e}")
            return None
        
        # Load test data
        X, y = self.load_test_data()
        
        # Make predictions
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        overall_accuracy = accuracy_score(y, y_pred)
        print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Per-class accuracy
        print("\\n📈 PER-CLASS ACCURACY:")
        class_accuracies = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = y == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y[class_mask], y_pred[class_mask])
                class_accuracies[class_name] = class_acc
                status = '✅' if class_acc >= 0.95 else '❌'
                print(f"{status} {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        # Check 95% target
        meets_target = all(acc >= 0.95 for acc in class_accuracies.values()) and overall_accuracy >= 0.95
        print(f"\\n🎯 95% TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
        
        # Confusion matrix
        print("\\n🔍 CONFUSION MATRIX:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Find worst performing class
        worst_class = min(class_accuracies.keys(), key=lambda k: class_accuracies[k])
        worst_accuracy = class_accuracies[worst_class]
        print(f"\\n⚠️  WORST CLASS: {worst_class} ({worst_accuracy*100:.1f}%)")
        
        return {
            'cycle': cycle_name,
            'overall_accuracy': float(overall_accuracy),
            'class_accuracies': {k: float(v) for k, v in class_accuracies.items()},
            'meets_95_target': meets_target,
            'worst_class': worst_class,
            'worst_accuracy': float(worst_accuracy),
            'model_path': model_path
        }
    
    def compare_cycles(self):
        """Compare all available models"""
        print("🔄 ITERATIVE IMPROVEMENT COMPARISON")
        print("=" * 60)
        
        # Models to test in order
        models_to_test = [
            ('sait01_aircraft_quick_fix.h5', 'Cycle 1 - Aircraft Fix'),
            ('sait01_balanced_multiclass.h5', 'Cycle 2 - Balanced Multi-class'),
            ('balanced_multiclass_best.h5', 'Cycle 2 - Best Checkpoint'),
            ('sait01_background_specialist.h5', 'Cycle 3 - Background Specialist'),
            ('sait01_vehicle_specialist.h5', 'Cycle 3 - Vehicle Specialist'),
            ('sait01_aircraft_specialist.h5', 'Cycle 3 - Aircraft Specialist')
        ]
        
        results = []
        
        for model_file, cycle_name in models_to_test:
            if os.path.exists(model_file):
                result = self.validate_model(model_file, cycle_name)
                if result:
                    results.append(result)
            else:
                print(f"\\n⏭️  SKIPPING {cycle_name} - {model_file} not found")
        
        # Summary comparison
        if results:
            print("\\n📊 CYCLE COMPARISON SUMMARY")
            print("=" * 60)
            
            best_overall = max(results, key=lambda x: x['overall_accuracy'])
            
            for result in results:
                status = '🏆' if result == best_overall else '📈'
                target_status = '✅' if result['meets_95_target'] else '❌'
                print(f"{status} {result['cycle']}")
                print(f"   Overall: {result['overall_accuracy']*100:.1f}% | Target: {target_status}")
                print(f"   Worst: {result['worst_class']} ({result['worst_accuracy']*100:.1f}%)")
                print()
            
            # Final recommendation
            if best_overall['meets_95_target']:
                print(f"🎉 SUCCESS! {best_overall['cycle']} achieved 95% target!")
                print(f"   Recommended model: {best_overall['model_path']}")
            else:
                print(f"🔄 CONTINUE: Best so far is {best_overall['cycle']} with {best_overall['overall_accuracy']*100:.1f}%")
                print(f"   Next focus: Improve {best_overall['worst_class']} detection")
        
        return results

def main():
    validator = CycleValidator()
    results = validator.compare_cycles()
    
    # Save comparison results
    with open('cycle_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\\n💾 Results saved to cycle_comparison_results.json")

if __name__ == "__main__":
    main()