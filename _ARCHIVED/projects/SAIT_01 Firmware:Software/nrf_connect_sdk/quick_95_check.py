#!/usr/bin/env python3
"""
Quick 95% Target Check - Test best model with larger sample size
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import accuracy_score

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

def quick_validation_check():
    """Quick check of best model with different sample sizes"""
    print("🔍 QUICK 95% TARGET CHECK")
    print("=" * 50)
    
    # Test best model
    model_path = 'balanced_multiclass_best.h5'
    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found")
        return
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ Loaded: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    preprocessor = SaitAudioPreprocessor()
    class_names = ['background', 'vehicle', 'aircraft']
    
    # Test with different sample sizes
    for samples_per_class in [100, 200, 500]:
        print(f"\n📊 Testing with {samples_per_class} samples per class...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        X, y = [], []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
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
            
            print(f"   {class_name}: {count} samples")
        
        X = np.array(X)
        y = np.array(y)
        
        # Make predictions
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        overall_accuracy = accuracy_score(y, y_pred)
        print(f"🎯 Overall: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        class_accuracies = {}
        all_meet_target = True
        
        for i, class_name in enumerate(class_names):
            class_mask = y == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y[class_mask], y_pred[class_mask])
                class_accuracies[class_name] = class_acc
                status = '✅' if class_acc >= 0.95 else '❌'
                print(f"{status} {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
                if class_acc < 0.95:
                    all_meet_target = False
        
        meets_target = all_meet_target and overall_accuracy >= 0.95
        print(f"🎯 95% TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
        
        if meets_target:
            print("🎉 SUCCESS! 95% target achieved!")
            return True
    
    return False

if __name__ == "__main__":
    success = quick_validation_check()
    if success:
        print("\n🎉 95% TARGET ACHIEVED! STOPPING ITERATIVE IMPROVEMENT")
    else:
        print("\n🔄 Continue with Cycle 4...")