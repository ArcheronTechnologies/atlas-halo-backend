#!/usr/bin/env python3
"""
Balanced Multi-Class Fix - Cycle 2
Fix vehicle confusion while maintaining aircraft detection
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class BalancedMultiClassFix:
    """Balanced training for all three classes"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_stratified_dataset(self):
        """Load perfectly stratified dataset"""
        print("📊 Loading stratified dataset for balanced training...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        X, y = [], []
        
        # EQUAL samples per class to prevent any bias
        samples_per_class = 1200
        
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
    
    def create_balanced_model(self):
        """Create model with balanced multi-class focus"""
        model = keras.Sequential([
            layers.Input(shape=(64, 63, 1)),
            
            # Multi-scale feature extraction
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Enhanced feature extraction
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Deep feature extraction
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classification with class-specific branches
            layers.GlobalAveragePooling2D(),
            
            # Shared dense layer
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Class-discriminative layer
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output with equal treatment for all classes
            layers.Dense(3, activation='softmax')
        ], name="balanced_multiclass_model")
        
        return model
    
    def train_balanced_model(self, X, y):
        """Train with carefully balanced class weights"""
        print("🎯 Training balanced multi-class model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Moderate class weights - less extreme than before
        class_weight = {
            0: 1.0,    # Background - normal weight
            1: 1.5,    # Vehicle - slight boost to help with confusion
            2: 2.0     # Aircraft - moderate boost (down from 5x)
        }
        print(f"Class weights: {class_weight}")
        
        # Create model
        model = self.create_balanced_model()
        
        # Compile with balanced approach
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0008),  # Slightly lower LR
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for balanced training
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.7, patience=5, min_lr=1e-6
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'balanced_multiclass_best.h5', 
                monitor='val_accuracy', 
                save_best_only=True
            )
        ]
        
        # Train with balanced approach
        print("🎯 Training with balanced multi-class focus...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=60,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def validate_balanced_model(self, model, X, y):
        """Comprehensive validation"""
        print("✅ Validating balanced multi-class model...")
        
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Overall accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
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
        
        # Confusion matrix
        print("\\n🔍 CONFUSION MATRIX:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Detailed report
        print("\\n📊 CLASSIFICATION REPORT:")
        print(classification_report(y, y_pred, target_names=self.class_names))
        
        return accuracy, class_accuracies, meets_target

def main():
    print("🔄 CYCLE 2: BALANCED MULTI-CLASS TRAINING")
    print("=" * 60)
    
    trainer = BalancedMultiClassFix()
    
    # Load data
    X, y = trainer.load_stratified_dataset()
    
    # Train
    model, history = trainer.train_balanced_model(X, y)
    
    # Validate
    accuracy, class_accuracies, meets_target = trainer.validate_balanced_model(model, X, y)
    
    # Save models
    model.save("sait01_balanced_multiclass.h5")
    print("💾 Saved: sait01_balanced_multiclass.h5")
    
    # Convert to TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open("sait01_balanced_multiclass.tflite", "wb") as f:
            f.write(tflite_model)
        print("💾 Saved: sait01_balanced_multiclass.tflite")
    except Exception as e:
        print(f"TFLite conversion error: {e}")
    
    # Save results
    results = {
        "cycle": 2,
        "overall_accuracy": float(accuracy),
        "class_accuracies": {k: float(v) for k, v in class_accuracies.items()},
        "meets_95_target": meets_target,
        "model_path": "sait01_balanced_multiclass.h5"
    }
    
    with open("cycle_2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n{'🎉 CYCLE 2 SUCCESS' if meets_target else '🔄 CONTINUE TO CYCLE 3'}")
    
    return results

if __name__ == "__main__":
    main()