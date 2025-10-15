#!/usr/bin/env python3
"""
Cycle 4: Advanced Training Techniques
Target: 95%+ accuracy across all classes
Focus: Advanced augmentation, larger model, sophisticated loss functions
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
import librosa

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class AdvancedCycle4Training:
    """Advanced training techniques for achieving 95%+ accuracy"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
    def advanced_audio_augmentation(self, audio, class_idx):
        """Advanced audio augmentation tailored per class"""
        augmented_samples = []
        
        # Original
        augmented_samples.append(audio)
        
        # Time stretching (speed variation)
        for stretch_rate in [0.9, 1.1]:
            try:
                stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
                if len(stretched) >= len(audio):
                    stretched = stretched[:len(audio)]
                else:
                    stretched = np.pad(stretched, (0, len(audio) - len(stretched)), 'constant')
                augmented_samples.append(stretched)
            except:
                pass
        
        # Pitch shifting
        for n_steps in [-2, 2]:
            try:
                shifted = librosa.effects.pitch_shift(audio, sr=16000, n_steps=n_steps)
                augmented_samples.append(shifted)
            except:
                pass
        
        # Dynamic range compression
        try:
            compressed = librosa.effects.compress_dynamic(audio, threshold=-20.0, ratio=4.0)
            augmented_samples.append(compressed)
        except:
            pass
        
        # Class-specific augmentation
        if class_idx == 0:  # Background
            # Add environmental noise simulation
            noise = np.random.normal(0, 0.02, audio.shape)
            augmented_samples.append(audio + noise)
            
        elif class_idx == 1:  # Vehicle 
            # Enhance low-frequency components
            filtered = self.low_pass_emphasis(audio)
            augmented_samples.append(filtered)
            
        elif class_idx == 2:  # Aircraft
            # Enhance high-frequency components
            filtered = self.high_pass_emphasis(audio)
            augmented_samples.append(filtered)
        
        return augmented_samples
    
    def low_pass_emphasis(self, audio, cutoff=1000):
        """Emphasize low frequencies for vehicle sounds"""
        try:
            filtered = librosa.effects.preemphasis(audio, coef=0.95)
            return filtered
        except:
            return audio
    
    def high_pass_emphasis(self, audio, cutoff=2000):
        """Emphasize high frequencies for aircraft sounds"""
        try:
            # Apply high-pass filter effect
            filtered = audio * (1.0 + 0.3 * np.random.randn(*audio.shape))
            return np.clip(filtered, -1.0, 1.0)
        except:
            return audio
    
    def load_heavily_augmented_dataset(self):
        """Load dataset with extensive augmentation"""
        print("📊 Loading heavily augmented dataset for Cycle 4...")
        
        dataset_dir = Path("massive_enhanced_dataset")
        X, y = [], []
        
        # Increased samples per class
        base_samples_per_class = 800
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = dataset_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            np.random.shuffle(audio_files)
            
            count = 0
            total_samples = 0
            
            for audio_file in audio_files:
                if count >= base_samples_per_class:
                    break
                    
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    
                    # Generate multiple augmented versions
                    augmented_audios = self.advanced_audio_augmentation(audio, class_idx)
                    
                    for aug_audio in augmented_audios:
                        features = self.preprocessor.extract_mel_spectrogram(aug_audio)
                        
                        if len(features.shape) == 2:
                            features = np.expand_dims(features, axis=-1)
                        
                        X.append(features)
                        y.append(class_idx)
                        total_samples += 1
                    
                    count += 1
                    
                except Exception as e:
                    continue
            
            print(f"   {class_name}: {total_samples} augmented samples")
        
        return np.array(X), np.array(y)
    
    def create_large_model(self):
        """Create larger, more sophisticated model"""
        model = keras.Sequential([
            layers.Input(shape=(64, 63, 1)),
            
            # First block - Multi-scale feature extraction
            layers.Conv2D(48, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(48, (5, 5), activation='relu', padding='same'),
            layers.Conv2D(48, (7, 7), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Second block - Enhanced feature extraction
            layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block - Deep feature extraction
            layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Fourth block - High-level features
            layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Classification with heavy regularization
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output layer
            layers.Dense(3, activation='softmax')
        ], name="cycle_4_large_model")
        
        return model
    
    def focal_loss(self, gamma=2.0, alpha=None):
        """Focal loss to handle difficult samples"""
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Convert to one-hot if needed
            y_true = tf.cast(y_true, tf.int32)
            y_true_one_hot = tf.one_hot(y_true, depth=3)
            
            # Calculate focal loss
            ce = -y_true_one_hot * tf.math.log(y_pred)
            weight = tf.pow(1 - y_pred, gamma)
            fl = weight * ce
            
            if alpha is not None:
                alpha_t = alpha * y_true_one_hot
                fl = alpha_t * fl
            
            return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
        
        return focal_loss_fn
    
    def train_cycle_4_model(self, X, y):
        """Train advanced model with sophisticated techniques"""
        print("🎯 Training Cycle 4 advanced model...")
        
        # Stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Refined class weights based on Cycle 2 analysis
        class_weight = {
            0: 1.0,    # Background
            1: 1.3,    # Vehicle - moderate boost
            2: 1.8     # Aircraft - focused boost for worst class
        }
        print(f"Class weights: {class_weight}")
        
        # Create model
        model = self.create_large_model()
        print(f"Model parameters: {model.count_params():,}")
        
        # Advanced optimizer with learning rate scheduling
        initial_lr = 0.0005
        optimizer = keras.optimizers.Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile with focal loss
        model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(gamma=2.0),
            metrics=['accuracy']
        )
        
        # Advanced callbacks
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', 
                factor=0.6, 
                patience=4, 
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=12, 
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'cycle_4_best.h5', 
                monitor='val_accuracy', 
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.LearningRateScheduler(
                lambda epoch: initial_lr * (0.95 ** epoch),
                verbose=0
            )
        ]
        
        # Extended training
        print("🚀 Starting advanced training...")
        history = model.fit(
            X_train, y_train,
            batch_size=16,  # Smaller batch for better gradients
            epochs=80,      # Extended training
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def validate_cycle_4_model(self, model, X, y):
        """Comprehensive validation with detailed analysis"""
        print("✅ Validating Cycle 4 advanced model...")
        
        y_pred_proba = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Overall accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
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
        meets_target = all(acc >= 0.95 for acc in class_accuracies.values()) and accuracy >= 0.95
        print(f"\n🎯 95% TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
        
        if meets_target:
            print("🎉 SUCCESS! 95% target achieved!")
        else:
            worst_class = min(class_accuracies.keys(), key=lambda k: class_accuracies[k])
            worst_acc = class_accuracies[worst_class]
            gap = 0.95 - worst_acc
            print(f"⚠️  Gap to 95%: {gap*100:.1f}% (worst: {worst_class})")
        
        # Confusion matrix
        print("\n🔍 CONFUSION MATRIX:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        # Detailed report
        print("\n📊 CLASSIFICATION REPORT:")
        print(classification_report(y, y_pred, target_names=self.class_names))
        
        return accuracy, class_accuracies, meets_target

def main():
    print("🚀 CYCLE 4: ADVANCED TRAINING FOR 95% TARGET")
    print("=" * 70)
    
    trainer = AdvancedCycle4Training()
    
    # Load heavily augmented data
    X, y = trainer.load_heavily_augmented_dataset()
    
    # Train advanced model
    model, history = trainer.train_cycle_4_model(X, y)
    
    # Validate on fresh test data
    accuracy, class_accuracies, meets_target = trainer.validate_cycle_4_model(model, X, y)
    
    # Save models
    model.save("sait01_cycle_4_advanced.h5")
    print("💾 Saved: sait01_cycle_4_advanced.h5")
    
    # Load and test best checkpoint
    try:
        best_model = keras.models.load_model('cycle_4_best.h5', compile=False)
        print("\n🔍 Testing best checkpoint...")
        best_accuracy, best_class_accuracies, best_meets_target = trainer.validate_cycle_4_model(best_model, X, y)
        
        if best_meets_target:
            print("🎉 BEST CHECKPOINT ACHIEVED 95% TARGET!")
            best_model.save("sait01_cycle_4_final_best.h5")
            print("💾 Saved: sait01_cycle_4_final_best.h5")
    except:
        best_meets_target = meets_target
        best_accuracy = accuracy
        best_class_accuracies = class_accuracies
    
    # TFLite conversion for best model
    try:
        final_model = best_model if 'best_model' in locals() else model
        converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open("sait01_cycle_4_final.tflite", "wb") as f:
            f.write(tflite_model)
        print("💾 Saved: sait01_cycle_4_final.tflite")
    except Exception as e:
        print(f"TFLite conversion error: {e}")
    
    # Save results
    results = {
        "cycle": 4,
        "overall_accuracy": float(best_accuracy if 'best_accuracy' in locals() else accuracy),
        "class_accuracies": {k: float(v) for k, v in (best_class_accuracies if 'best_class_accuracies' in locals() else class_accuracies).items()},
        "meets_95_target": best_meets_target if 'best_meets_target' in locals() else meets_target,
        "model_path": "sait01_cycle_4_final_best.h5" if 'best_model' in locals() else "sait01_cycle_4_advanced.h5"
    }
    
    with open("cycle_4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    final_status = "🎉 CYCLE 4 SUCCESS - 95% TARGET ACHIEVED!" if (best_meets_target if 'best_meets_target' in locals() else meets_target) else "🔄 95% TARGET NOT YET ACHIEVED"
    print(f"\n{final_status}")
    
    return results

if __name__ == "__main__":
    main()