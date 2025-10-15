#!/usr/bin/env python3
"""
Break the 95% Accuracy Barrier
Advanced techniques to push past 95% accuracy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class Elite95PercentTrainer:
    """Elite training techniques to break 95% barrier"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def load_elite_dataset(self, samples_per_class=3000):
        """Load maximum available data"""
        print("📊 Loading ELITE dataset (maximum samples)...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        if not dataset_dir.exists():
            dataset_dir = Path("enhanced_sait01_dataset")
        
        if not dataset_dir.exists():
            print("❌ No enhanced dataset found")
            return None, None
            
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            
            # Use ALL available samples (no limit for elite training)
            print(f"   Loading {class_name}: {len(audio_files)} samples (ALL)")
            
            for audio_file in audio_files:
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    X.append(features)
                    y.append(class_idx)
                except Exception:
                    continue
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        print(f"✅ ELITE dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_elite_model_architecture(self):
        """Create enhanced model architecture optimized for 95%+"""
        print("🏗️  Creating ELITE model architecture...")
        
        # Enhanced input processing
        inputs = keras.layers.Input(shape=(64, 63, 1), name='mel_input')
        
        # Multi-scale feature extraction
        # Scale 1: Fine details
        conv1a = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1b = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1a)
        pool1 = keras.layers.MaxPooling2D((2, 2))(conv1b)
        drop1 = keras.layers.Dropout(0.25)(pool1)
        
        # Scale 2: Medium details
        conv2a = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
        conv2b = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2a)
        pool2 = keras.layers.MaxPooling2D((2, 2))(conv2b)
        drop2 = keras.layers.Dropout(0.25)(pool2)
        
        # Scale 3: Coarse details
        conv3a = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
        conv3b = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3a)
        pool3 = keras.layers.MaxPooling2D((2, 2))(conv3b)
        drop3 = keras.layers.Dropout(0.3)(pool3)
        
        # Additional deep layer for complex patterns
        conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(drop3)
        conv4b = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = keras.layers.MaxPooling2D((2, 2))(conv4b)
        drop4 = keras.layers.Dropout(0.4)(pool4)
        
        # Global feature extraction
        global_avg = keras.layers.GlobalAveragePooling2D()(drop4)
        global_max = keras.layers.GlobalMaxPooling2D()(drop4)
        
        # Flatten and combine
        flatten = keras.layers.Flatten()(drop4)
        combined = keras.layers.Concatenate()([flatten, global_avg, global_max])
        
        # Enhanced dense layers with residual connections
        dense1 = keras.layers.Dense(512, activation='relu')(combined)
        drop_dense1 = keras.layers.Dropout(0.5)(dense1)
        
        dense2 = keras.layers.Dense(256, activation='relu')(drop_dense1)
        drop_dense2 = keras.layers.Dropout(0.4)(dense2)
        
        dense3 = keras.layers.Dense(128, activation='relu')(drop_dense2)
        drop_dense3 = keras.layers.Dropout(0.3)(dense3)
        
        # Output with enhanced regularization
        outputs = keras.layers.Dense(3, activation='softmax', dtype='float32')(drop_dense3)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='elite_sait01_model')
        return model
    
    def create_advanced_training_strategy(self, X, y):
        """Advanced training strategy for 95%+ accuracy"""
        print("🎯 Creating ADVANCED training strategy...")
        
        # Stratified split with more test data for robust validation
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
        )
        
        print(f"📊 ELITE data split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Advanced class weighting to fix background classification
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # CRITICAL: Boost background class to fix 85.3% issue
        class_weight_dict[0] *= 3.0  # Background needs major boost
        class_weight_dict[1] *= 1.2  # Vehicle slight boost
        class_weight_dict[2] *= 1.5  # Aircraft moderate boost
        
        print(f"⚖️  ELITE class weights: {class_weight_dict}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict
    
    def train_elite_model(self, X, y):
        """Train elite model with advanced techniques"""
        print("\n🚀 ELITE MODEL TRAINING - TARGET: 95%+")
        print("=" * 70)
        
        # Create elite model
        model = self.create_elite_model_architecture()
        
        # Advanced training strategy
        X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict = self.create_advanced_training_strategy(X, y)
        
        # Advanced optimizer with learning rate scheduling
        initial_lr = 0.001
        optimizer = keras.optimizers.Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile with standard loss (label smoothing not available in this TF version)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Advanced callbacks
        callbacks = [
            # Aggressive early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Cosine annealing learning rate
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=1e-8,
                verbose=1
            ),
            
            # Save best model
            keras.callbacks.ModelCheckpoint(
                'sait01_elite_95_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            ),
            
            # Learning rate warmup and decay
            keras.callbacks.LearningRateScheduler(
                lambda epoch: initial_lr * (0.95 ** epoch) if epoch > 5 else initial_lr * (epoch + 1) / 6,
                verbose=0
            )
        ]
        
        print("🔥 Starting ELITE training with advanced techniques...")
        
        # Train with extended epochs and advanced techniques
        history = model.fit(
            X_train, y_train,
            batch_size=16,  # Smaller batch for better gradients
            epochs=50,      # More epochs
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        best_model = keras.models.load_model('sait01_elite_95_model.h5')
        
        # Final evaluation
        print(f"\n🎯 ELITE MODEL EVALUATION")
        print("-" * 50)
        
        test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
        y_pred = best_model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"🏆 ELITE Test Accuracy: {test_accuracy*100:.1f}%")
        
        # Detailed analysis
        print(f"\n📋 ELITE PERFORMANCE BREAKDOWN:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print(f"\n🔍 ELITE Confusion Matrix:")
        print(f"           Predicted")
        print(f"         BG  VH  AC")
        for i, true_class in enumerate(['BG', 'VH', 'AC']):
            print(f"True {true_class}: {cm[i]}")
        
        # 95% barrier analysis
        barrier_broken = test_accuracy >= 0.95
        print(f"\n🚀 95% BARRIER STATUS:")
        print(f"   Target: 95.0%")
        print(f"   Achieved: {test_accuracy*100:.1f}%")
        
        if barrier_broken:
            print("   🎉 BARRIER BROKEN! 95%+ ACHIEVED!")
            print("   ✅ BATTLEFIELD DEPLOYMENT APPROVED!")
        else:
            gap = 0.95 - test_accuracy
            print(f"   📈 Gap remaining: {gap*100:.1f} percentage points")
            if gap < 0.01:
                print("   🔥 EXTREMELY CLOSE - Elite techniques working!")
            elif gap < 0.02:
                print("   💪 VERY CLOSE - Continue elite training!")
        
        # Create elite TFLite
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            
            with open('sait01_elite_95_model.tflite', 'wb') as f:
                f.write(tflite_model)
            
            print(f"📱 Elite TFLite created: {len(tflite_model)/1024:.1f} KB")
        except Exception as e:
            print(f"⚠️  TFLite conversion: {e}")
        
        return {
            'accuracy': test_accuracy,
            'barrier_broken': barrier_broken,
            'gap': 0.95 - test_accuracy if not barrier_broken else 0,
            'confusion_matrix': cm.tolist(),
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }

def main():
    print("🚀 BREAKING THE 95% ACCURACY BARRIER")
    print("=" * 70)
    print("🎯 ELITE training techniques for 95%+ battlefield accuracy")
    print("💪 Advanced architecture + Enhanced training strategy")
    print("=" * 70)
    
    trainer = Elite95PercentTrainer()
    
    # Load maximum available data
    X, y = trainer.load_elite_dataset()
    
    if X is None:
        print("❌ Could not load dataset")
        return
    
    # Train elite model
    results = trainer.train_elite_model(X, y)
    
    # Save elite results
    with open('elite_95_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 ELITE TRAINING COMPLETE")
    print("=" * 70)
    
    if results['barrier_broken']:
        print("🎉 SUCCESS: 95% BARRIER BROKEN!")
        print("🚀 BATTLEFIELD DEPLOYMENT READY!")
    else:
        print(f"📈 Progress: {results['accuracy']*100:.1f}% achieved")
        print(f"🔧 Gap: {results['gap']*100:.1f}% remaining")
    
    print("💾 Results saved: elite_95_results.json")

if __name__ == "__main__":
    main()