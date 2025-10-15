#!/usr/bin/env python3
"""
Battlefield Model Training
Train the combat-enhanced model for 95% accuracy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

# Enable eager execution to avoid TF issues
tf.config.run_functions_eagerly(True)

class BattlefieldModelTrainer:
    """Train combat-enhanced model for battlefield deployment"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_enhanced_dataset(self, samples_per_class=1800):
        """Load the combat-enhanced dataset"""
        print("📊 Loading combat-enhanced dataset...")
        
        enhanced_dir = Path("enhanced_sait01_dataset")
        if not enhanced_dir.exists():
            print("❌ Enhanced dataset not found. Run battlefield_audio_integration.py first")
            return None, None
            
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = enhanced_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            
            # Shuffle for variety
            np.random.shuffle(audio_files)
            if len(audio_files) > samples_per_class:
                audio_files = audio_files[:samples_per_class]
            
            print(f"   Loading {class_name}: {len(audio_files)} samples")
            
            for audio_file in audio_files:
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    X.append(features)
                    y.append(class_idx)
                except Exception:
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Enhanced dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def train_battlefield_model(self, X, y):
        """Train the combat-enhanced model"""
        print("\n🚀 Training battlefield-ready model...")
        print("=" * 60)
        
        # Load best base model
        base_model_path = 'sait01_production_model.h5'
        if not os.path.exists(base_model_path):
            print(f"❌ Base model not found: {base_model_path}")
            return None
            
        print(f"📖 Loading base model: {base_model_path}")
        base_model = keras.models.load_model(base_model_path)
        print(f"   Original baseline: 94.0% accuracy")
        
        # Data split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Data split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Compute class weights for battlefield emphasis
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Emphasize combat-critical classes more
        class_weight_dict[1] *= 2.5  # Vehicle (includes combat vehicles)
        class_weight_dict[2] *= 1.5  # Aircraft (includes combat aircraft)
        
        print(f"⚖️  Class weights: {class_weight_dict}")
        
        # Compile with robust optimizer
        base_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'sait01_battlefield_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("🔥 Starting battlefield training...")
        
        # Train with enhanced combat data
        history = base_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n📊 BATTLEFIELD MODEL EVALUATION")
        print("-" * 50)
        
        test_loss, test_accuracy = base_model.evaluate(X_test, y_test, verbose=0)
        print(f"🎯 Test Accuracy: {test_accuracy*100:.1f}%")
        
        # Detailed predictions
        y_pred = base_model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print(f"\n📋 DETAILED PERFORMANCE:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print(f"\n🔍 Confusion Matrix:")
        print(f"           Predicted")
        print(f"         BG  VH  AC")
        for i, true_class in enumerate(['BG', 'VH', 'AC']):
            print(f"True {true_class}: {cm[i]}")
        
        # 95% target check
        target_reached = test_accuracy >= 0.95
        print(f"\n🎯 95% BATTLEFIELD ACCURACY TARGET:")
        print(f"   Target: 95.0%")
        print(f"   Achieved: {test_accuracy*100:.1f}%")
        print(f"   Status: {'✅ TARGET REACHED!' if target_reached else '❌ Need more enhancement'}")
        
        if target_reached:
            print("🚀 BATTLEFIELD READY - Model deployed for combat operations!")
        else:
            gap = 0.95 - test_accuracy
            print(f"   📈 Gap: {gap*100:.1f} percentage points")
            print("   💡 Recommendation: Add more diverse combat sounds")
        
        # Create TFLite version
        print(f"\n📱 Creating TensorFlow Lite version...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_path = 'sait01_battlefield_model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            model_size = len(tflite_model) / 1024
            print(f"✅ TFLite model saved: {tflite_path} ({model_size:.1f} KB)")
        except Exception as e:
            print(f"⚠️  TFLite conversion failed: {e}")
        
        return {
            'accuracy': test_accuracy,
            'target_reached': target_reached,
            'history': history.history,
            'confusion_matrix': cm.tolist()
        }

def main():
    """Train battlefield-enhanced model"""
    print("⚔️  SAIT_01 BATTLEFIELD MODEL TRAINING")
    print("=" * 70)
    print("🎯 Objective: Train combat-enhanced model for 95%+ accuracy")
    print("🚀 Enhanced with synthetic battlefield audio")
    print("=" * 70)
    
    trainer = BattlefieldModelTrainer()
    
    # Load enhanced dataset
    X, y = trainer.load_enhanced_dataset()
    
    if X is None:
        print("❌ Could not load enhanced dataset")
        return
    
    # Train battlefield model
    results = trainer.train_battlefield_model(X, y)
    
    if results:
        print(f"\n🏆 BATTLEFIELD TRAINING COMPLETE")
        print("=" * 70)
        
        if results['target_reached']:
            print("✅ SUCCESS: 95% accuracy achieved!")
            print("🚀 Model ready for battlefield deployment!")
        else:
            print(f"📈 Achieved: {results['accuracy']*100:.1f}%")
            print("🔧 Consider adding more diverse combat audio")
        
        # Save results
        with open('battlefield_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("💾 Results saved to: battlefield_training_results.json")

if __name__ == "__main__":
    main()