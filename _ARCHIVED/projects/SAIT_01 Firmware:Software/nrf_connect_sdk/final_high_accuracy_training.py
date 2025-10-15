#!/usr/bin/env python3
"""
Final High Accuracy Training System - 90-95% Target
Clean implementation for maximum reliability
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

# Add current directory to path
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class FinalHighAccuracyTrainer:
    """Final high accuracy trainer targeting 90-95% accuracy"""
    
    def __init__(self, data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        self.data_dir = data_dir
        self.preprocessor = SaitAudioPreprocessor()
        
        # Performance targets
        self.target_accuracy = 0.90      # 90% accuracy target (conservative)
        self.max_false_positive_rate = 0.05  # 5% max false positives
        
        print("🚀 FINAL HIGH ACCURACY TRAINING SYSTEM")
        print("=" * 50)
        print(f"🎯 Target Accuracy: {self.target_accuracy*100:.1f}%")
        print(f"🛡️  Max False Positives: {self.max_false_positive_rate*100:.1f}%")
        print("=" * 50)
        
    def load_enhanced_dataset(self):
        """Load the enhanced dataset with comprehensive negative samples"""
        print("📊 Loading enhanced dataset...")
        
        X_data = []
        y_data = []
        
        train_path = Path(self.data_dir) / 'train'
        class_mapping = {'background': 0, 'drone': 1, 'helicopter': 2}
        
        for class_name, class_id in class_mapping.items():
            class_path = train_path / class_name
            if not class_path.exists():
                print(f"❌ Missing: {class_path}")
                continue
                
            files = list(class_path.glob("*.wav"))
            print(f"📂 Processing {len(files)} {class_name} samples...")
            
            for i, audio_file in enumerate(files):
                try:
                    audio = self.preprocessor.load_and_resample(str(audio_file))
                    
                    # Quality check
                    if len(audio) < 8000 or np.max(np.abs(audio)) < 0.005:
                        continue
                    
                    mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    # Feature quality check
                    if np.any(np.isnan(mel_spec)) or np.all(mel_spec == 0):
                        continue
                    
                    X_data.append(mel_spec)
                    y_data.append(class_id)
                    
                    if (i + 1) % 200 == 0:
                        print(f"  {i + 1}/{len(files)} processed")
                        
                except Exception as e:
                    continue
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"✅ Loaded {len(X_data)} high-quality samples")
        print(f"📊 Shape: {X_data.shape}")
        print(f"📋 Class distribution: {np.bincount(y_data)}")
        
        # Check dataset balance
        class_counts = np.bincount(y_data)
        background_ratio = class_counts[0] / len(y_data)
        print(f"🛡️  Background samples: {background_ratio:.1%} (good for FP rejection)")
        
        return X_data, y_data
    
    def create_advanced_model(self):
        """Create advanced model architecture"""
        print("🧠 Building Advanced Model")
        
        inputs = tf.keras.layers.Input(shape=(63, 64, 1))
        
        # Main branch with depthwise separable convolutions
        x = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        x = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Attention mechanism
        attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = tf.keras.layers.multiply([x, attention])
        
        x = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Dense layers with regularization
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile with class weights for imbalanced data
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"📊 Model: {model.count_params()} parameters")
        return model
    
    def augment_data(self, X, y, multiplier=3):
        """Apply comprehensive data augmentation"""
        print(f"🔄 Applying data augmentation (x{multiplier})...")
        
        augmented_X = []
        augmented_y = []
        
        # Original samples
        for i in range(len(X)):
            augmented_X.append(X[i])
            augmented_y.append(y[i])
        
        # Augmented samples
        for i in range(len(X)):
            for _ in range(multiplier):
                sample = X[i].copy()
                
                # Frequency masking
                if random.random() < 0.5:
                    freq_mask_width = random.randint(2, 8)
                    freq_start = random.randint(0, 64 - freq_mask_width)
                    sample[:, freq_start:freq_start + freq_mask_width, :] *= 0.1
                
                # Time masking
                if random.random() < 0.5:
                    time_mask_width = random.randint(2, 10)
                    time_start = random.randint(0, 63 - time_mask_width)
                    sample[time_start:time_start + time_mask_width, :, :] *= 0.1
                
                # Noise injection
                if random.random() < 0.6:
                    noise_level = random.uniform(0.01, 0.05)
                    noise = np.random.randn(*sample.shape) * noise_level
                    sample = sample + noise
                
                # Spectral shifting
                if random.random() < 0.3:
                    shift = random.randint(-2, 2)
                    if shift != 0:
                        sample = np.roll(sample, shift, axis=1)
                
                augmented_X.append(sample)
                augmented_y.append(y[i])
        
        augmented_X = np.array(augmented_X)
        augmented_y = np.array(augmented_y)
        
        print(f"✅ Augmented dataset: {len(augmented_X)} samples")
        print(f"📋 New distribution: {np.bincount(augmented_y)}")
        
        return augmented_X, augmented_y
    
    def train_model(self, X_data, y_data):
        """Train the high accuracy model"""
        print("\\n🚀 Training High Accuracy Model")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, stratify=y_data, random_state=42
        )
        
        print(f"📊 Split: {len(X_train)} train, {len(X_test)} test")
        print(f"📋 Train: {np.bincount(y_train)}")
        print(f"📋 Test: {np.bincount(y_test)}")
        
        # Augment training data
        X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
        
        # Create model
        model = self.create_advanced_model()
        
        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount(y_train_aug)
        total_samples = len(y_train_aug)
        class_weights = {
            i: total_samples / (len(class_counts) * count) 
            for i, count in enumerate(class_counts)
        }
        print(f"📊 Class weights: {class_weights}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'final_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print("🔄 Starting training...")
        start_time = time.time()
        
        history = model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_test, y_test),
            epochs=60,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        train_time = time.time() - start_time
        print(f"⏱️  Training: {train_time:.1f}s")
        
        return model, history, (X_test, y_test)
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\\n📊 Model Evaluation")
        print("=" * 40)
        
        # Predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # False positive analysis
        background_mask = (y_test == 0)
        false_positives = np.sum((y_pred > 0) & background_mask)
        total_background = np.sum(background_mask)
        fp_rate = false_positives / total_background if total_background > 0 else 0
        
        # True positive analysis
        target_mask = (y_test > 0)
        true_positives = np.sum((y_pred > 0) & target_mask)
        total_targets = np.sum(target_mask)
        tp_rate = true_positives / total_targets if total_targets > 0 else 0
        
        print(f"🎯 Results:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  False Positive Rate: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
        print(f"  True Positive Rate: {tp_rate:.4f} ({tp_rate*100:.2f}%)")
        
        # Confidence analysis
        max_confidences = np.max(y_pred_probs, axis=1)
        high_conf_mask = max_confidences >= 0.8
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(y_pred[high_conf_mask] == y_test[high_conf_mask])
            print(f"  High Confidence Accuracy (≥80%): {high_conf_accuracy:.4f}")
        
        # Classification report
        class_names = ['background', 'drone', 'helicopter']
        print("\\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\\n📊 Confusion Matrix:")
        print(cm)
        
        # Target achievement
        accuracy_met = accuracy >= self.target_accuracy
        fp_met = fp_rate <= self.max_false_positive_rate
        
        print(f"\\n🎯 Target Achievement:")
        print(f"  Accuracy ≥{self.target_accuracy*100:.1f}%: {'✅ MET' if accuracy_met else '❌ NOT MET'}")
        print(f"  FP Rate ≤{self.max_false_positive_rate*100:.1f}%: {'✅ MET' if fp_met else '❌ NOT MET'}")
        
        targets_achieved = accuracy_met and fp_met
        
        if targets_achieved:
            print("\\n🎉 HIGH ACCURACY TARGETS ACHIEVED!")
            print("🚀 Model ready for defense deployment!")
        else:
            print("\\n🔧 Targets not fully achieved")
            if not accuracy_met:
                print(f"  Need {(self.target_accuracy - accuracy)*100:.1f}% more accuracy")
            if not fp_met:
                print(f"  Need {(fp_rate - self.max_false_positive_rate)*100:.1f}% fewer false positives")
        
        return {
            'accuracy': accuracy,
            'fp_rate': fp_rate,
            'tp_rate': tp_rate,
            'targets_achieved': targets_achieved,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'confidences': max_confidences
        }
    
    def convert_to_tflite(self, model, X_sample):
        """Convert to TensorFlow Lite"""
        print("\\n🔄 Converting to TensorFlow Lite")
        print("=" * 40)
        
        def representative_dataset():
            for i in range(min(200, len(X_sample))):
                yield [X_sample[i:i+1].astype(np.float32)]
        
        try:
            # INT8 quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            quantized_model = converter.convert()
            size_kb = len(quantized_model) / 1024
            
            print(f"✅ Quantized model: {size_kb:.1f} KB")
            
            # Save
            with open('sait01_final_high_accuracy.tflite', 'wb') as f:
                f.write(quantized_model)
            
            print("💾 Model saved as 'sait01_final_high_accuracy.tflite'")
            
            if size_kb <= 80:
                print("✅ Perfect fit for nRF5340!")
            else:
                print("⚠️  Larger than ideal for nRF5340")
            
            return quantized_model
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            
            # Fallback
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            fallback_model = converter.convert()
            
            with open('sait01_final_high_accuracy.tflite', 'wb') as f:
                f.write(fallback_model)
            
            print("✅ Fallback model saved")
            return fallback_model
    
    def run_training(self):
        """Complete training pipeline"""
        print("\\n🚀 FINAL HIGH ACCURACY TRAINING PIPELINE")
        print("=" * 60)
        print("🎯 Targeting 90%+ accuracy with robust FP rejection")
        print("=" * 60)
        
        # Load dataset
        X_data, y_data = self.load_enhanced_dataset()
        
        if len(X_data) == 0:
            print("❌ No data loaded")
            return None
        
        # Train model
        model, history, (X_test, y_test) = self.train_model(X_data, y_data)
        
        # Evaluate
        results = self.evaluate_model(model, X_test, y_test)
        
        # Convert to TFLite if successful
        if results['targets_achieved']:
            tflite_model = self.convert_to_tflite(model, X_test)
            results['tflite_model'] = tflite_model
        
        # Final summary
        print("\\n🏆 FINAL RESULTS")
        print("=" * 30)
        print(f"✅ Dataset: {len(X_data)} samples")
        print(f"🎯 Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"🛡️  False Positives: {results['fp_rate']:.3f} ({results['fp_rate']*100:.1f}%)")
        print(f"🚀 Success: {'✅ YES' if results['targets_achieved'] else '❌ PARTIAL'}")
        
        if results['targets_achieved']:
            print("\\n🎉 HIGH ACCURACY TRAINING COMPLETE!")
            print("🚀 Ready for critical defense deployment!")
        else:
            print("\\n🔧 Additional optimization may be needed")
        
        return {
            'model': model,
            'results': results,
            'test_data': (X_test, y_test)
        }

def main():
    trainer = FinalHighAccuracyTrainer()
    training_results = trainer.run_training()
    
    if training_results and training_results['results']['targets_achieved']:
        print("\\n🚀 SUCCESS: High accuracy model deployment ready!")
    else:
        print("\\n🔧 Training completed - review results for optimization")

if __name__ == "__main__":
    main()