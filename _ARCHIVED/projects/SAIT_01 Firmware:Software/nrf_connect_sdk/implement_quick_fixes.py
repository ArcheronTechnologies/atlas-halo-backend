#!/usr/bin/env python3
"""
Implement Quick Fixes for SAIT_01 Accuracy Issues
Apply immediate improvements based on diagnosis
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import librosa
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time
import json

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, MODEL_CONFIG

class QuickFixTrainer:
    """Implement quick fixes for accuracy improvement"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def create_improved_model(self):
        """Create improved model with better regularization"""
        print("🏗️  Creating improved model with quick fixes...")
        
        inputs = keras.Input(shape=MODEL_CONFIG['input_shape'], name='mel_input')
        
        # Improved CNN architecture with better regularization
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)  # Reduced from 0.5
        
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)  # Reduced from 0.5
        
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)  # Reduced from 0.5
        
        # Improved dense layers with L2 regularization
        x = keras.layers.Dense(256, activation='relu', 
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.3)(x)  # Reduced from 0.4
        
        x = keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.2)(x)  # Reduced from 0.3
        
        x = keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.2)(x)
        
        outputs = keras.layers.Dense(3, activation='softmax', name='classification')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SAIT01_QuickFix')
        
        # Compile with class weights to emphasize vehicle detection
        class_weights = {
            0: 1.0,    # background - normal weight
            1: 3.0,    # vehicle - 3x weight to boost detection 
            2: 1.2     # aircraft - slightly higher weight
        }
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Slightly lower LR
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Improved model created: {model.count_params()} parameters")
        print(f"🎯 Class weights applied: {class_weights}")
        
        return model, class_weights
    
    def load_balanced_dataset_from_expanded(self, max_samples_per_class=1000):
        """Load balanced dataset from expanded dataset with cleaner background samples"""
        print("📊 Loading balanced dataset with quality filtering...")
        
        expanded_dir = Path("expanded_sait01_dataset")
        if not expanded_dir.exists():
            print("❌ Expanded dataset not found")
            return None, None
        
        X = []
        y = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = expanded_dir / class_name
            if not class_dir.exists():
                continue
            
            audio_files = list(class_dir.glob("*.wav"))
            
            # Filter background samples to get cleaner ones
            if class_name == 'background':
                # Prefer original and synthetic samples over augmented ones for background
                filtered_files = []
                
                # Prioritize synthetic and original background sounds
                for f in audio_files:
                    filename = f.name.lower()
                    if 'synthetic' in filename or 'original' in filename:
                        filtered_files.append(f)
                    elif 'generated' in filename:
                        # Only include generated sounds that are clearly environmental
                        if any(keyword in filename for keyword in ['water', 'wind', 'rain', 'nature']):
                            filtered_files.append(f)
                
                # If not enough, add some augmented files
                if len(filtered_files) < max_samples_per_class:
                    remaining = max_samples_per_class - len(filtered_files)
                    other_files = [f for f in audio_files if f not in filtered_files]
                    filtered_files.extend(other_files[:remaining])
                
                audio_files = filtered_files
            
            # Limit samples per class for balanced training
            if len(audio_files) > max_samples_per_class:
                # Shuffle and select subset
                np.random.shuffle(audio_files)
                audio_files = audio_files[:max_samples_per_class]
            
            print(f"   Loading {class_name}: {len(audio_files)} samples")
            
            for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
                try:
                    # Load and preprocess audio
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    X.append(features)
                    y.append(class_idx)
                    
                except Exception as e:
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Balanced dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_improved_model(self, model, class_weights, X, y):
        """Train model with class weights and improved strategy"""
        print("🚀 Training improved model with quick fixes...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        print(f"📊 Data split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=12,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'sait01_quickfix_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with class weights
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=40,  # More epochs with early stopping
            validation_data=(X_val, y_val),
            class_weight=class_weights,  # Apply class weights
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Test model
        print("🔍 Testing improved model...")
        test_start = time.time()
        
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        inference_time = (time.time() - test_start) / len(X_test) * 1000  # ms per sample
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"\n📊 Quick Fix Results:")
        print(f"   Training time: {training_time/60:.1f} minutes")
        print(f"   Test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Inference time: {inference_time:.2f}ms per sample")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print(f"\n🔍 Improved Confusion Matrix:")
        print(f"           Predicted")
        print(f"         BG  VH  AC")
        for i, true_class in enumerate(['BG', 'VH', 'AC']):
            print(f"True {true_class}: {cm[i]}")
        
        # Compare with original performance
        original_accuracy = 0.473
        improvement = (accuracy - original_accuracy) * 100
        
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"   Original Accuracy: {original_accuracy*100:.1f}%")
        print(f"   Quick Fix Accuracy: {accuracy*100:.1f}%")
        print(f"   Improvement: {improvement:+.1f} percentage points")
        
        if improvement > 15:
            print("   🏆 EXCELLENT: Major improvement achieved!")
        elif improvement > 8:
            print("   ✅ GOOD: Significant improvement")
        elif improvement > 3:
            print("   📈 MODERATE: Some improvement")
        else:
            print("   ⚠️  MINIMAL: Need more fixes")
        
        # Analyze class-specific improvements
        print(f"\n🎯 CLASS-SPECIFIC ANALYSIS:")
        
        # Calculate per-class recall
        for i, class_name in enumerate(self.class_names):
            if i in y_test:
                class_mask = (y_test == i)
                class_correct = np.sum(y_pred_classes[class_mask] == i)
                class_total = np.sum(class_mask)
                class_recall = class_correct / class_total if class_total > 0 else 0
                
                print(f"   {class_name.upper():<10}: {class_recall*100:.1f}% recall ({class_correct}/{class_total})")
                
                if class_name == 'vehicle':
                    original_vehicle_recall = 0.16
                    vehicle_improvement = (class_recall - original_vehicle_recall) * 100
                    print(f"     {'':12} Vehicle improvement: {vehicle_improvement:+.1f}%")
        
        return {
            'accuracy': accuracy,
            'improvement': improvement,
            'training_time': training_time,
            'inference_time': inference_time,
            'confusion_matrix': cm,
            'model': model
        }
    
    def save_improved_model(self, model):
        """Save the improved model"""
        print("💾 Saving improved model...")
        
        # Save Keras model
        model.save('sait01_quickfix_model.h5')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open('sait01_quickfix_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        model_size = os.path.getsize('sait01_quickfix_model.h5') / 1024
        tflite_size = len(tflite_model) / 1024
        
        print(f"✅ Models saved:")
        print(f"   Keras: sait01_quickfix_model.h5 ({model_size:.1f} KB)")
        print(f"   TFLite: sait01_quickfix_model.tflite ({tflite_size:.1f} KB)")

def main():
    """Main quick fix implementation"""
    print("⚡ SAIT_01 QUICK FIXES IMPLEMENTATION")
    print("=" * 60)
    
    # Initialize trainer
    trainer = QuickFixTrainer()
    
    # Create improved model
    model, class_weights = trainer.create_improved_model()
    
    # Load balanced dataset
    X, y = trainer.load_balanced_dataset_from_expanded(max_samples_per_class=800)
    
    if X is None:
        print("❌ Cannot load dataset")
        return
    
    # Train improved model
    results = trainer.train_improved_model(model, class_weights, X, y)
    
    # Save model
    trainer.save_improved_model(results['model'])
    
    # Final summary
    print(f"\n🎉 QUICK FIXES IMPLEMENTATION COMPLETE!")
    print("=" * 60)
    print(f"🎯 Accuracy Improvement: {results['improvement']:+.1f} percentage points")
    print(f"📈 New Accuracy: {results['accuracy']*100:.1f}%")
    print(f"⏱️  Training Time: {results['training_time']/60:.1f} minutes")
    print(f"⚡ Inference Time: {results['inference_time']:.1f}ms")
    
    if results['improvement'] >= 15:
        print("\n🚀 READY FOR NEXT PHASE: Medium-term improvements")
    else:
        print("\n🔧 NEED MORE FIXES: Consider implementing medium-term solutions")

if __name__ == "__main__":
    main()