#!/usr/bin/env python3
"""
Quick Aircraft Classification Fix
Simple approach to fix aircraft detection immediately
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

class QuickAircraftFix:
    """Quick fix for aircraft classification"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_balanced_dataset(self):
        """Load perfectly balanced dataset"""
        print("🚨 Loading balanced dataset for aircraft fix...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        X, y = [], []
        
        # FORCE EQUAL SAMPLING
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
    
    def create_simple_model(self):
        """Simple CNN model for reliable classification"""
        model = keras.Sequential([
            layers.Input(shape=(64, 63, 1)),
            
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ], name="aircraft_fix_model")
        
        return model
    
    def train_fix_model(self, X, y):
        """Train with heavy aircraft focus"""
        print("🚀 Training aircraft fix model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Extreme class weights favoring aircraft
        class_weight = {0: 1.0, 1: 1.0, 2: 5.0}  # 5x weight for aircraft
        print(f"Class weights: {class_weight}")
        
        # Create model
        model = self.create_simple_model()
        
        # Compile with standard categorical crossentropy
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with aggressive aircraft focus
        print("🎯 Training with aircraft focus...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            verbose=1
        )
        
        return model, history
    
    def validate_fix(self, model, X, y):
        """Quick validation"""
        print("✅ Validating aircraft fix...")
        
        y_pred = model.predict(X, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y, y_pred_classes)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class accuracy
        print("\\n📈 PER-CLASS ACCURACY:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y[class_mask], y_pred_classes[class_mask])
                print(f"{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        # Confusion matrix
        print("\\n🔍 CONFUSION MATRIX:")
        cm = confusion_matrix(y, y_pred_classes)
        print(cm)
        
        # Aircraft detection rate
        aircraft_mask = y == 2
        aircraft_detected = np.sum(y_pred_classes[aircraft_mask] == 2)
        aircraft_total = np.sum(aircraft_mask)
        aircraft_rate = aircraft_detected / aircraft_total
        
        print(f"\\n✈️  AIRCRAFT DETECTION: {aircraft_detected}/{aircraft_total} ({aircraft_rate*100:.1f}%)")
        
        return accuracy, aircraft_rate

def main():
    print("🚨 QUICK AIRCRAFT FIX")
    print("=" * 40)
    
    fixer = QuickAircraftFix()
    
    # Load data
    X, y = fixer.load_balanced_dataset()
    print(f"Dataset shape: {X.shape}, Labels: {y.shape}")
    
    # Train
    model, history = fixer.train_fix_model(X, y)
    
    # Validate
    accuracy, aircraft_rate = fixer.validate_fix(model, X, y)
    
    # Save
    model.save("sait01_aircraft_quick_fix.h5")
    
    # Convert to TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open("sait01_aircraft_quick_fix.tflite", "wb") as f:
            f.write(tflite_model)
        print("💾 TFLite model saved")
    except Exception as e:
        print(f"TFLite error: {e}")
    
    # Results
    success = aircraft_rate > 0.7
    print(f"\\n{'🎉 SUCCESS' if success else '❌ FAILED'}: Aircraft rate {aircraft_rate*100:.1f}%")
    
    return {"success": success, "aircraft_rate": aircraft_rate, "accuracy": accuracy}

if __name__ == "__main__":
    main()