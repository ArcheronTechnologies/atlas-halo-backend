#!/usr/bin/env python3
"""
Automated Cycle Runner - Continue the iterative improvement loop
Automatically runs the next cycle based on previous results
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def check_model_exists(model_path):
    """Check if a model file exists"""
    return os.path.exists(model_path)

def run_validation(model_path, cycle_name):
    """Run validation on a specific model"""
    print(f"🔍 VALIDATING {cycle_name}")
    
    validation_script = f"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

# Load model
model = keras.models.load_model('{model_path}', compile=False)
preprocessor = SaitAudioPreprocessor()
class_names = ['background', 'vehicle', 'aircraft']

# Load test data
dataset_dir = Path('massive_enhanced_dataset')
X, y = [], []
samples_per_class = 100

for class_idx, class_name in enumerate(class_names):
    class_dir = dataset_dir / class_name
    if not class_dir.exists():
        continue
    
    audio_files = list(class_dir.glob('*.wav'))
    np.random.shuffle(audio_files)
    
    count = 0
    for audio_file in audio_files[:samples_per_class]:
        try:
            audio = preprocessor.load_and_resample(audio_file)
            features = preprocessor.extract_mel_spectrogram(audio)
            if len(features.shape) == 2:
                features = np.expand_dims(features, axis=-1)
            X.append(features)
            y.append(class_idx)
            count += 1
        except:
            continue

X = np.array(X)
y = np.array(y)

# Make predictions
y_pred_proba = model.predict(X, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate metrics
overall_accuracy = accuracy_score(y, y_pred)
class_accuracies = {{}}

for i, class_name in enumerate(class_names):
    class_mask = y == i
    if np.sum(class_mask) > 0:
        class_acc = accuracy_score(y[class_mask], y_pred[class_mask])
        class_accuracies[class_name] = float(class_acc)

# Check 95% target
meets_target = all(acc >= 0.95 for acc in class_accuracies.values()) and overall_accuracy >= 0.95

print(f"Overall Accuracy: {{overall_accuracy:.4f}} ({{overall_accuracy*100:.2f}}%)")
for class_name, acc in class_accuracies.items():
    status = '✅' if acc >= 0.95 else '❌'
    print(f"{{status}} {{class_name}}: {{acc:.4f}} ({{acc*100:.2f}}%)")

print(f"95% TARGET: {{'✅ ACHIEVED' if meets_target else '❌ NOT MET'}}")

# Save results
import json
results = {{
    'cycle': '{cycle_name}',
    'overall_accuracy': float(overall_accuracy),
    'class_accuracies': class_accuracies,
    'meets_95_target': meets_target,
    'model_path': '{model_path}'
}}

with open('{cycle_name.lower().replace(" ", "_")}_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("VALIDATION_COMPLETE")
"""
    
    try:
        result = subprocess.run(['python3', '-c', validation_script], 
                              capture_output=True, text=True, timeout=300)
        return result.stdout, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "Validation timed out", False

def should_continue_to_next_cycle():
    """Determine if we should continue to the next cycle"""
    # Check latest results
    result_files = [
        'cycle_2_results.json',
        'balanced_multiclass_validation_results.json'
    ]
    
    for result_file in result_files:
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)
                
            if results.get('meets_95_target', False):
                print(f"🎉 SUCCESS! {result_file} achieved 95% target!")
                return False, results
            else:
                worst_class = min(results['class_accuracies'].keys(), 
                                key=lambda k: results['class_accuracies'][k])
                worst_acc = results['class_accuracies'][worst_class]
                print(f"🔄 CONTINUE: Best {results['overall_accuracy']*100:.1f}%, worst {worst_class} {worst_acc*100:.1f}%")
    
    return True, None

def run_next_cycle():
    """Run the next cycle based on current status"""
    print("🔄 DETERMINING NEXT CYCLE...")
    
    # Check if Cycle 2 completed
    if check_model_exists('sait01_balanced_multiclass.h5'):
        print("✅ Cycle 2 model found, validating...")
        output, success = run_validation('sait01_balanced_multiclass.h5', 'Cycle 2')
        print(output)
        
        if success and "VALIDATION_COMPLETE" in output:
            continue_needed, results = should_continue_to_next_cycle()
            
            if not continue_needed:
                print("🎉 TARGET ACHIEVED - STOPPING ITERATIONS")
                return True
            else:
                print("🚀 STARTING CYCLE 3...")
                # Run Cycle 3
                try:
                    subprocess.run(['python3', 'advanced_ensemble_fix.py'], timeout=1800)
                    print("✅ Cycle 3 completed")
                    
                    # Validate Cycle 3 results
                    if check_model_exists('sait01_background_specialist.h5'):
                        print("🔍 Validating Cycle 3 ensemble...")
                        # Run ensemble validation here
                        return run_cycle_comparison()
                        
                except subprocess.TimeoutExpired:
                    print("⏰ Cycle 3 timed out")
                    return False
        else:
            print("❌ Cycle 2 validation failed")
            return False
    else:
        print("⏳ Cycle 2 still training...")
        return False

def run_cycle_comparison():
    """Run final comparison of all cycles"""
    print("🏁 RUNNING FINAL CYCLE COMPARISON...")
    
    try:
        result = subprocess.run(['python3', 'cycle_validator.py'], 
                              capture_output=True, text=True, timeout=600)
        print(result.stdout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Cycle comparison timed out")
        return False

def main():
    print("🤖 AUTOMATED ITERATIVE IMPROVEMENT")
    print("=" * 50)
    
    # Wait for current training to complete and run next steps
    max_attempts = 10
    for attempt in range(max_attempts):
        print(f"\\n🔄 Attempt {attempt + 1}/{max_attempts}")
        
        success = run_next_cycle()
        if success:
            print("🎉 ITERATIVE IMPROVEMENT COMPLETE!")
            break
        
        if attempt < max_attempts - 1:
            print("⏳ Waiting 2 minutes before next check...")
            time.sleep(120)  # Wait 2 minutes
    else:
        print("⏰ Maximum attempts reached")

if __name__ == "__main__":
    main()