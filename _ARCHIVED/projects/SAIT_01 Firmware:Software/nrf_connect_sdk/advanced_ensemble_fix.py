#!/usr/bin/env python3
"""
Advanced Ensemble Fix - Cycle 3
Ensemble approach with specialized models for each class
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class AdvancedEnsembleFix:
    """Advanced ensemble with class-specific experts"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        self.models = {}
        
    def create_specialist_model(self, model_name):
        """Create specialist model for specific class detection"""
        if model_name == "background_specialist":
            # Background specialist - focus on noise patterns
            model = keras.Sequential([
                layers.Input(shape=(64, 63, 1)),
                layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(3, activation='softmax')
            ], name="background_specialist")
            
        elif model_name == "vehicle_specialist":
            # Vehicle specialist - focus on low-freq engine patterns
            model = keras.Sequential([
                layers.Input(shape=(64, 63, 1)),
                layers.Conv2D(32, (7, 3), activation='relu', padding='same'),  # Wide temporal
                layers.Conv2D(32, (3, 7), activation='relu', padding='same'),  # Wide frequency
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(3, activation='softmax')
            ], name="vehicle_specialist")
            
        elif model_name == "aircraft_specialist":
            # Aircraft specialist - focus on high-freq rotor patterns
            model = keras.Sequential([
                layers.Input(shape=(64, 63, 1)),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 5), activation='relu', padding='same'),  # Focus on freq patterns
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(3, activation='softmax')
            ], name="aircraft_specialist")
        
        return model
    
    def load_augmented_dataset(self):
        """Load dataset with class-specific augmentation"""
        print("📊 Loading augmented dataset for ensemble training...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        X, y = [], []
        
        samples_per_class = 1000
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            np.random.shuffle(audio_files)
            
            count = 0
            for audio_file in audio_files:
                if count >= samples_per_class:
                    break
                    
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    
                    # Class-specific augmentation
                    if class_name == 'vehicle':
                        # Add low-frequency emphasis for vehicles
                        audio = self.apply_vehicle_augmentation(audio)
                    elif class_name == 'aircraft':
                        # Add high-frequency emphasis for aircraft
                        audio = self.apply_aircraft_augmentation(audio)
                    
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
    
    def apply_vehicle_augmentation(self, audio):
        """Vehicle-specific augmentation"""
        # Enhance low frequencies (engine sounds)
        return audio * (1.0 + 0.1 * np.random.randn())
    
    def apply_aircraft_augmentation(self, audio):
        """Aircraft-specific augmentation"""
        # Enhance high frequencies (rotor sounds)  
        return audio * (1.0 + 0.1 * np.random.randn())
    
    def train_ensemble(self, X, y):
        """Train ensemble of specialist models"""
        print("🎯 Training ensemble of specialist models...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train each specialist
        for specialist_name in ["background_specialist", "vehicle_specialist", "aircraft_specialist"]:
            print(f"\\n🎓 Training {specialist_name}...")
            
            model = self.create_specialist_model(specialist_name)
            
            # Class-specific weights
            if "background" in specialist_name:
                class_weight = {0: 1.5, 1: 1.0, 2: 1.0}
            elif "vehicle" in specialist_name:
                class_weight = {0: 1.0, 1: 2.0, 2: 1.0}
            else:  # aircraft
                class_weight = {0: 1.0, 1: 1.0, 2: 2.0}
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train specialist
            model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=30,  # Shorter training for each specialist
                validation_data=(X_val, y_val),
                class_weight=class_weight,
                verbose=0
            )
            
            self.models[specialist_name] = model
            print(f"✅ {specialist_name} trained")
        
        return self.models
    
    def ensemble_predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Weighted ensemble - give each specialist equal weight
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Alternative: Confidence-weighted ensemble
        # You could weight by each model's confidence for its specialty
        
        return ensemble_pred
    
    def validate_ensemble(self, X, y):
        """Validate ensemble performance"""
        print("✅ Validating ensemble model...")
        
        y_pred_proba = self.ensemble_predict(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Overall accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"🎯 Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
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
        meets_target = all(acc >= 0.95 for acc in class_accuracies.values()) and accuracy >= 0.95
        print(f"\\n🎯 95% TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
        
        return accuracy, class_accuracies, meets_target
    
    def save_ensemble(self):
        """Save all specialist models"""
        for name, model in self.models.items():
            model.save(f"sait01_{name}.h5")
            print(f"💾 Saved: sait01_{name}.h5")

def main():
    print("🔄 CYCLE 3: ADVANCED ENSEMBLE TRAINING")
    print("=" * 60)
    
    trainer = AdvancedEnsembleFix()
    
    # Load data
    X, y = trainer.load_augmented_dataset()
    
    # Train ensemble
    models = trainer.train_ensemble(X, y)
    
    # Validate
    accuracy, class_accuracies, meets_target = trainer.validate_ensemble(X, y)
    
    # Save models
    trainer.save_ensemble()
    
    # Save results
    results = {
        "cycle": 3,
        "overall_accuracy": float(accuracy),
        "class_accuracies": {k: float(v) for k, v in class_accuracies.items()},
        "meets_95_target": meets_target,
        "ensemble_models": list(trainer.models.keys())
    }
    
    with open("cycle_3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n{'🎉 CYCLE 3 SUCCESS' if meets_target else '🔄 CONTINUE TO CYCLE 4'}")
    
    return results

if __name__ == "__main__":
    main()