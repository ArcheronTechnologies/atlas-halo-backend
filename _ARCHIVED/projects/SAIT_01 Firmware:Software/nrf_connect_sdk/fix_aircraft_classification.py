#!/usr/bin/env python3
"""
Emergency Aircraft Classification Fix
Address catastrophic background class bias
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
import time

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class AircraftClassificationFix:
    """Emergency fix for aircraft classification failure"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_balanced_dataset(self):
        """Load perfectly balanced dataset with aggressive aircraft focus"""
        print("🚨 EMERGENCY: Loading balanced dataset for aircraft fix...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        X, y = [], []
        
        # FORCE EQUAL SAMPLING - critical for aircraft detection
        samples_per_class = 1500  # Equal samples per class
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            np.random.shuffle(audio_files)
            
            # Take exactly samples_per_class from each
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
            
            print(f"   {class_name}: {count} samples (target: {samples_per_class})")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Balanced dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_aircraft_focused_model(self):
        """Create model architecture specifically designed to detect aircraft"""
        print("🛠️  Creating aircraft-focused model architecture...")
        
        inputs = keras.Input(shape=(64, 63, 1), name="aircraft_input")
        
        # Aggressive feature extraction for aircraft detection
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Second block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Third block - more focused on aircraft patterns
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Final classification layer
        outputs = layers.Dense(3, activation='softmax', name="aircraft_output")(x)
        
        model = keras.Model(inputs, outputs, name="aircraft_focused_model")
        return model
    
    def create_focal_loss(self, alpha=1.0, gamma=2.0):
        """Focal loss to address class imbalance and hard examples"""
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Convert to one-hot if needed
            if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
            
            ce = -y_true * tf.math.log(y_pred)
            weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
            fl = weight * ce
            return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
        
        return focal_loss_fn
    
    def train_aircraft_fix_model(self, X, y):
        """Train model with extreme focus on aircraft detection"""
        print("🚀 Training aircraft-focused model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class weights - HEAVILY favor aircraft class
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        # Boost aircraft weight even more
        class_weights[2] *= 3.0  # Triple the aircraft weight
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Create model
        model = self.create_aircraft_focused_model()
        
        # Compile with focal loss and aggressive aircraft focus
        focal_loss = self.create_focal_loss(alpha=2.0, gamma=3.0)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR for stability
            loss=focal_loss,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks for aggressive aircraft training
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'aircraft_fix_model_best.h5', 
                monitor='val_accuracy', 
                save_best_only=True
            )
        ]
        
        # Train with class weights
        print("🎯 Starting aircraft-focused training...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=80,  # More epochs for aircraft learning
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def validate_aircraft_fix(self, model, X, y):
        """Validate the aircraft fix"""
        print("✅ Validating aircraft classification fix...")
        
        # Make predictions
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Overall accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class accuracy - focus on aircraft
        print("\n📈 PER-CLASS ACCURACY:")
        for i, class_name in enumerate(self.class_names):
            class_mask = y == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y[class_mask], y_pred[class_mask])
                print(f"{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
                
                if class_name == 'aircraft' and class_acc < 0.8:
                    print(f"⚠️  WARNING: Aircraft accuracy still too low!")
        
        # Confusion matrix
        print("\n🔍 CONFUSION MATRIX:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Aircraft-specific analysis
        aircraft_mask = y == 2
        aircraft_predictions = y_pred[aircraft_mask]
        aircraft_true_positives = np.sum(aircraft_predictions == 2)
        aircraft_total = np.sum(aircraft_mask)
        
        print(f"\n✈️  AIRCRAFT DETECTION ANALYSIS:")
        print(f"Aircraft samples: {aircraft_total}")
        print(f"Correctly detected: {aircraft_true_positives}")
        print(f"Aircraft detection rate: {aircraft_true_positives/aircraft_total*100:.1f}%")
        
        return accuracy, aircraft_true_positives/aircraft_total

def main():
    print("🚨 EMERGENCY AIRCRAFT CLASSIFICATION FIX")
    print("=" * 60)
    
    fixer = AircraftClassificationFix()
    
    # Load balanced dataset
    X, y = fixer.load_balanced_dataset()
    
    # Train aircraft-focused model
    model, history = fixer.train_aircraft_fix_model(X, y)
    
    # Validate the fix
    accuracy, aircraft_detection_rate = fixer.validate_aircraft_fix(model, X, y)
    
    # Save the fixed model
    model.save("sait01_aircraft_fixed_model.h5")
    print(f"\n💾 Fixed model saved: sait01_aircraft_fixed_model.h5")
    
    # Convert to TFLite for deployment
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open("sait01_aircraft_fixed_model.tflite", "wb") as f:
            f.write(tflite_model)
        print(f"💾 TFLite model saved: sait01_aircraft_fixed_model.tflite")
    except Exception as e:
        print(f"⚠️  TFLite conversion failed: {e}")
    
    # Save results
    results = {
        "overall_accuracy": float(accuracy),
        "aircraft_detection_rate": float(aircraft_detection_rate),
        "model_path": "sait01_aircraft_fixed_model.h5",
        "tflite_path": "sait01_aircraft_fixed_model.tflite",
        "fix_successful": aircraft_detection_rate > 0.8
    }
    
    with open("aircraft_fix_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if aircraft_detection_rate > 0.8:
        print(f"\n🎉 AIRCRAFT FIX SUCCESSFUL!")
        print(f"✅ Aircraft detection rate: {aircraft_detection_rate*100:.1f}%")
        print(f"✅ Overall accuracy: {accuracy*100:.1f}%")
    else:
        print(f"\n❌ AIRCRAFT FIX FAILED - Need more aggressive measures")
    
    return results

if __name__ == "__main__":
    main()