#!/usr/bin/env python3
"""
Final Cycle Validator - Test all cycles including Cycle 4
Complete validation with 95% target verification
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

class FinalCycleValidator:
    """Final comprehensive validation of all training cycles"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_fresh_test_data(self, samples_per_class=300):
        """Load completely fresh test data for final validation"""
        print(f"📊 Loading fresh test data ({samples_per_class} per class)...")
        
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
        """Validate a specific model with detailed metrics"""
        print(f"\n🔍 VALIDATING {cycle_name.upper()}")
        print("=" * 60)
        
        try:
            model = keras.models.load_model(model_path, compile=False)
            print(f"✅ Loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {e}")
            return None
        
        # Load test data
        X, y = self.load_fresh_test_data()
        
        # Make predictions
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        overall_accuracy = accuracy_score(y, y_pred)
        print(f"🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Per-class accuracy
        print("\n📈 PER-CLASS ACCURACY:")
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
        print(f"\n🎯 95% TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
        
        if meets_target:
            print("🎉 SUCCESS! This model achieves the 95% target!")
        
        # Confusion matrix
        print("\n🔍 CONFUSION MATRIX:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Find worst performing class
        worst_class = min(class_accuracies.keys(), key=lambda k: class_accuracies[k])
        worst_accuracy = class_accuracies[worst_class]
        print(f"\n⚠️  WORST CLASS: {worst_class} ({worst_accuracy*100:.1f}%)")
        
        # Gap analysis
        if not meets_target:
            gap = 0.95 - worst_accuracy
            print(f"📉 Gap to 95%: {gap*100:.1f}%")
        
        return {
            'cycle': cycle_name,
            'overall_accuracy': float(overall_accuracy),
            'class_accuracies': {k: float(v) for k, v in class_accuracies.items()},
            'meets_95_target': meets_target,
            'worst_class': worst_class,
            'worst_accuracy': float(worst_accuracy),
            'model_path': model_path
        }
    
    def final_comparison(self):
        """Run final comprehensive comparison of all cycles"""
        print("🏁 FINAL ITERATIVE IMPROVEMENT VALIDATION")
        print("=" * 70)
        
        # All models to test in order
        models_to_test = [
            ('sait01_aircraft_quick_fix.h5', 'Cycle 1 - Aircraft Fix'),
            ('sait01_balanced_multiclass.h5', 'Cycle 2 - Balanced Multi-class'),
            ('balanced_multiclass_best.h5', 'Cycle 2 - Best Checkpoint'),
            ('sait01_background_specialist.h5', 'Cycle 3 - Background Specialist'),
            ('sait01_vehicle_specialist.h5', 'Cycle 3 - Vehicle Specialist'),
            ('sait01_aircraft_specialist.h5', 'Cycle 3 - Aircraft Specialist'),
            ('sait01_cycle_4_advanced.h5', 'Cycle 4 - Advanced Training'),
            ('cycle_4_best.h5', 'Cycle 4 - Best Checkpoint'),
            ('sait01_cycle_4_final_best.h5', 'Cycle 4 - Final Best')
        ]
        
        results = []
        successful_models = []
        
        for model_file, cycle_name in models_to_test:
            if os.path.exists(model_file):
                result = self.validate_model(model_file, cycle_name)
                if result:
                    results.append(result)
                    if result['meets_95_target']:
                        successful_models.append(result)
            else:
                print(f"\n⏭️  SKIPPING {cycle_name} - {model_file} not found")
        
        # Final summary
        if results:
            print("\n🏆 FINAL CYCLE COMPARISON SUMMARY")
            print("=" * 70)
            
            # Sort by overall accuracy
            results.sort(key=lambda x: x['overall_accuracy'], reverse=True)
            
            best_overall = results[0]
            
            for i, result in enumerate(results):
                rank = '🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else f"{i+1:2d}."
                target_status = '✅' if result['meets_95_target'] else '❌'
                print(f"{rank} {result['cycle']}")
                print(f"   Overall: {result['overall_accuracy']*100:.1f}% | Target: {target_status}")
                print(f"   Worst: {result['worst_class']} ({result['worst_accuracy']*100:.1f}%)")
                print()
            
            # Success check
            if successful_models:
                print("🎉🎉🎉 SUCCESS! THE FOLLOWING MODELS ACHIEVED 95% TARGET: 🎉🎉🎉")
                for model in successful_models:
                    print(f"✅ {model['cycle']} - {model['overall_accuracy']*100:.1f}%")
                    print(f"   Model: {model['model_path']}")
                
                # Recommend best successful model
                best_successful = max(successful_models, key=lambda x: x['overall_accuracy'])
                print(f"\n🏆 RECOMMENDED MODEL: {best_successful['cycle']}")
                print(f"   File: {best_successful['model_path']}")
                print(f"   Accuracy: {best_successful['overall_accuracy']*100:.1f}%")
                
            else:
                print("❌ 95% TARGET NOT YET ACHIEVED")
                print(f"🥇 Best model: {best_overall['cycle']} with {best_overall['overall_accuracy']*100:.1f}%")
                gap = 0.95 - best_overall['worst_accuracy']
                print(f"📉 Remaining gap: {gap*100:.1f}% (focus on {best_overall['worst_class']})")
                
                if best_overall['overall_accuracy'] >= 0.90:
                    print("🔄 Close to target - recommend Cycle 5 with:")
                    print("   • Even more aggressive augmentation")
                    print("   • Larger model architecture")
                    print("   • Transfer learning from pre-trained models")
                    print("   • Advanced loss functions (focal + triplet)")
        
        return results, successful_models

def main():
    validator = FinalCycleValidator()
    results, successful_models = validator.final_comparison()
    
    # Save final results
    final_results = {
        'all_cycles': results,
        'successful_models': successful_models,
        'target_achieved': len(successful_models) > 0,
        'best_model': results[0] if results else None
    }
    
    with open('final_cycle_comparison.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n💾 Final results saved to final_cycle_comparison.json")
    
    return final_results

if __name__ == "__main__":
    main()