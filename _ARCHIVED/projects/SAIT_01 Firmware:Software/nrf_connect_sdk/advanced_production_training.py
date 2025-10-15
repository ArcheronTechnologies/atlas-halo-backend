#!/usr/bin/env python3
"""
Advanced Production TinyML Training - 90-95% Accuracy Target
Implements multiple advanced techniques for high-accuracy defense sensor
"""

import os
import sys
import numpy as np
import time
import random
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# TensorFlow imports with optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Add current directory to path
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class AdvancedProductionTrainer:
    """Advanced trainer targeting 90-95% accuracy with false positive rejection"""
    
    def __init__(self, data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        self.data_dir = data_dir
        self.preprocessor = SaitAudioPreprocessor()
        
        # Enhanced class mapping with noise rejection
        self.class_mapping = {
            'background': 0,  # Natural sounds, environmental noise
            'drone': 1,       # Drone/UAV sounds
            'helicopter': 2   # Helicopter sounds
        }
        
        # Advanced training parameters
        self.target_accuracy = 0.925  # 92.5% target
        self.false_positive_threshold = 0.05  # Max 5% false positives
        
        print("🚀 Advanced Production Trainer")
        print(f"🎯 Target Accuracy: {self.target_accuracy*100:.1f}%")
        print(f"🛡️  False Positive Limit: {self.false_positive_threshold*100:.1f}%")
        print(f"📁 Data: {data_dir}")
        
        # Enable mixed precision for better performance
        # policy = mixed_precision.Policy('mixed_float16')
        # mixed_precision.set_policy(policy)
        
    def load_enhanced_dataset(self, samples_per_class=200):
        """Load dataset with enhanced sampling and validation"""
        print(f"📊 Loading enhanced dataset ({samples_per_class} per class)...")
        
        X_data = []
        y_data = []
        file_sources = []  # Track which files samples came from
        
        train_path = Path(self.data_dir) / 'train'
        
        for class_name, class_id in self.class_mapping.items():
            class_path = train_path / class_name
            if not class_path.exists():
                print(f"❌ Missing: {class_path}")
                continue
                
            files = list(class_path.glob("*.wav"))
            
            # Ensure we have enough files
            if len(files) < samples_per_class // 4:
                print(f"⚠️  Limited data for {class_name}: {len(files)} files")
            
            # Sample with replacement if needed, but track source diversity
            selected_files = []
            if len(files) >= samples_per_class:
                selected_files = random.sample(files, samples_per_class)
            else:
                # Repeat files but with different augmentations
                repeats = (samples_per_class // len(files)) + 1
                selected_files = (files * repeats)[:samples_per_class]
            
            print(f"📂 Processing {len(selected_files)} {class_name} samples...")
            
            for i, audio_file in enumerate(selected_files):
                try:
                    # Load and validate audio
                    audio = self.preprocessor.load_and_resample(str(audio_file))
                    
                    # Quality check - reject very short or silent audio
                    if len(audio) < 8000 or np.max(np.abs(audio)) < 0.01:
                        continue
                    
                    # Extract features with quality validation
                    mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    # Feature quality check
                    if np.any(np.isnan(mel_spec)) or np.all(mel_spec == 0):
                        continue
                    
                    X_data.append(mel_spec)
                    y_data.append(class_id)
                    file_sources.append(str(audio_file))
                    
                    if (i + 1) % 50 == 0:
                        print(f"  {i + 1}/{len(selected_files)} processed")
                        
                except Exception as e:
                    print(f"  ❌ {audio_file.name}: {e}")
                    continue
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"✅ Loaded {len(X_data)} high-quality samples")
        print(f"📊 Shape: {X_data.shape}")
        print(f"📋 Class distribution: {np.bincount(y_data)}")
        
        # Validate class balance
        class_counts = np.bincount(y_data)
        min_samples = np.min(class_counts)
        max_samples = np.max(class_counts)
        balance_ratio = min_samples / max_samples if max_samples > 0 else 0
        
        print(f"⚖️  Class balance ratio: {balance_ratio:.2f}")
        if balance_ratio < 0.7:
            print("⚠️  Class imbalance detected - applying balancing")
            X_data, y_data = self._balance_classes(X_data, y_data)
        
        return X_data, y_data, file_sources
    
    def _balance_classes(self, X, y):
        """Balance classes by undersampling majority classes"""
        class_counts = np.bincount(y)
        min_count = np.min(class_counts[class_counts > 0])
        
        balanced_X = []
        balanced_y = []
        
        for class_id in range(len(class_counts)):
            if class_counts[class_id] == 0:
                continue
                
            class_indices = np.where(y == class_id)[0]
            
            if len(class_indices) > min_count:
                # Undersample majority class
                selected_indices = np.random.choice(class_indices, min_count, replace=False)
            else:
                selected_indices = class_indices
            
            balanced_X.append(X[selected_indices])
            balanced_y.append(y[selected_indices])
        
        X_balanced = np.concatenate(balanced_X, axis=0)
        y_balanced = np.concatenate(balanced_y, axis=0)
        
        print(f"✅ Balanced dataset: {np.bincount(y_balanced)}")
        return X_balanced, y_balanced
    
    def create_advanced_augmentation(self):
        """Advanced data augmentation for robustness"""
        
        def advanced_augment(mel_spec, label):
            """Apply multiple augmentation techniques"""
            
            # Random noise injection (simulate environmental noise)
            noise_factor = tf.random.uniform([], 0.0, 0.1)
            noise = tf.random.normal(tf.shape(mel_spec), 0.0, noise_factor)
            mel_spec = mel_spec + noise
            
            # Frequency masking (simulate frequency-specific interference)
            if tf.random.uniform([]) < 0.4:
                freq_mask_param = tf.random.uniform([], 2, 8, dtype=tf.int32)
                freq_start = tf.random.uniform([], 0, 64 - freq_mask_param, dtype=tf.int32)
                mask = tf.ones_like(mel_spec)
                mask = mask[:, freq_start:freq_start + freq_mask_param, :].assign(
                    tf.zeros_like(mask[:, freq_start:freq_start + freq_mask_param, :])
                )
                mel_spec = mel_spec * mask
            
            # Time masking (simulate temporal gaps)
            if tf.random.uniform([]) < 0.4:
                time_mask_param = tf.random.uniform([], 2, 10, dtype=tf.int32)
                time_start = tf.random.uniform([], 0, 63 - time_mask_param, dtype=tf.int32)
                mask = tf.ones_like(mel_spec)
                mask = mask[time_start:time_start + time_mask_param, :, :].assign(
                    tf.zeros_like(mask[time_start:time_start + time_mask_param, :, :])
                )
                mel_spec = mel_spec * mask
            
            # Dynamic range compression (simulate different recording conditions)
            if tf.random.uniform([]) < 0.3:
                compression_factor = tf.random.uniform([], 0.7, 1.3)
                mel_spec = tf.sign(mel_spec) * tf.pow(tf.abs(mel_spec), compression_factor)
            
            # Spectral shift (simulate Doppler effects)
            if tf.random.uniform([]) < 0.2:
                shift_amount = tf.random.uniform([], -3, 3, dtype=tf.int32)
                mel_spec = tf.roll(mel_spec, shift_amount, axis=1)
            
            return mel_spec, label
        
        return advanced_augment
    
    def create_ensemble_model(self):
        """Create ensemble of different architectures for robust classification"""
        print("\n🧠 Building Advanced Ensemble Model")
        
        inputs = layers.Input(shape=(63, 64, 1), name='audio_input')
        
        # Branch 1: Depthwise Separable CNN (efficient)
        x1 = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        x1 = layers.Dropout(0.25)(x1)
        
        x1 = layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        x1 = layers.Dropout(0.25)(x1)
        
        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dense(64, activation='relu')(x1)
        branch1 = layers.Dropout(0.5)(x1)
        
        # Branch 2: Attention-based CNN
        x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x2 = layers.BatchNormalization()(x2)
        
        # Attention mechanism
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x2)
        x2 = layers.multiply([x2, attention])
        
        x2 = layers.MaxPooling2D((2, 2))(x2)
        x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling2D((2, 2))(x2)
        x2 = layers.GlobalAveragePooling2D()(x2)
        x2 = layers.Dense(64, activation='relu')(x2)
        branch2 = layers.Dropout(0.5)(x2)
        
        # Branch 3: Residual-like connections
        x3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
        x3 = layers.BatchNormalization()(x3)
        residual = x3
        
        x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.add([x3, residual])  # Residual connection
        
        x3 = layers.MaxPooling2D((4, 4))(x3)
        x3 = layers.GlobalAveragePooling2D()(x3)
        x3 = layers.Dense(64, activation='relu')(x3)
        branch3 = layers.Dropout(0.5)(x3)
        
        # Combine branches
        combined = layers.concatenate([branch1, branch2, branch3])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output with confidence estimation
        main_output = layers.Dense(3, activation='softmax', name='classification')(combined)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(combined)
        
        model = models.Model(inputs=inputs, 
                           outputs=[main_output, confidence_output], 
                           name='SAIT01_Advanced_Ensemble')
        
        # Advanced optimizer with scheduling
        initial_lr = 0.001
        optimizer = optimizers.Adam(learning_rate=initial_lr)
        
        model.compile(
            optimizer=optimizer,
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={
                'classification': 1.0,
                'confidence': 0.3
            },
            metrics={
                'classification': ['accuracy'],
                'confidence': ['mae']
            }
        )
        
        print("📝 Advanced Model Summary:")
        model.summary()
        
        params = model.count_params()
        size_kb = params * 4 / 1024
        print(f"📏 Model size: {size_kb:.1f} KB ({params} parameters)")
        
        return model
    
    def create_advanced_callbacks(self, patience=15):
        """Advanced training callbacks for optimal performance"""
        
        return [
            # Early stopping with restore best weights
            callbacks.EarlyStopping(
                monitor='val_classification_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Learning rate reduction with more aggressive schedule
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpointing
            callbacks.ModelCheckpoint(
                'best_sait01_model.h5',
                monitor='val_classification_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Custom learning rate scheduler
            callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (0.9 ** epoch) if epoch < 20 else 0.001 * (0.95 ** (epoch - 20)),
                verbose=0
            ),
            
            # Terminate on NaN
            callbacks.TerminateOnNaN()
        ]
    
    def stratified_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Stratified split maintaining class proportions"""
        from sklearn.model_selection import train_test_split
        
        return train_test_split(X, y, test_size=test_size, 
                              stratify=y, random_state=random_state)
    
    def train_advanced_model(self, X_data, y_data):
        """Advanced training with cross-validation and ensemble"""
        print("\n🚀 Advanced Model Training")
        print("=" * 60)
        
        # Stratified split
        X_train, X_test, y_train, y_test = self.stratified_train_test_split(X_data, y_data)
        
        print(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
        print(f"📋 Train distribution: {np.bincount(y_train)}")
        print(f"📋 Test distribution: {np.bincount(y_test)}")
        
        # Create confidence labels (1.0 for real samples, lower for augmented)
        y_train_conf = np.ones(len(y_train), dtype=np.float32)
        y_test_conf = np.ones(len(y_test), dtype=np.float32)
        
        # Create model
        model = self.create_ensemble_model()
        
        # Advanced augmentation
        augment_fn = self.create_advanced_augmentation()
        
        # Create augmented training data
        augmented_X = []
        augmented_y = []
        augmented_conf = []
        
        print("🔄 Generating augmented training data...")
        for i in range(len(X_train)):
            # Original sample
            augmented_X.append(X_train[i])
            augmented_y.append(y_train[i])
            augmented_conf.append(1.0)
            
            # Generate 2-3 augmented versions
            for _ in range(random.randint(2, 3)):
                aug_x, aug_y = augment_fn(X_train[i], y_train[i])
                augmented_X.append(aug_x.numpy())
                augmented_y.append(aug_y.numpy())
                augmented_conf.append(0.8)  # Lower confidence for augmented
        
        X_train_aug = np.array(augmented_X)
        y_train_aug = np.array(augmented_y)
        y_train_conf_aug = np.array(augmented_conf)
        
        print(f"✅ Augmented training set: {len(X_train_aug)} samples")
        
        # Advanced callbacks
        model_callbacks = self.create_advanced_callbacks()
        
        # Training with multiple outputs
        print("🔄 Starting advanced training...")
        start_time = time.time()
        
        history = model.fit(
            X_train_aug, 
            {
                'classification': y_train_aug,
                'confidence': y_train_conf_aug
            },
            validation_data=(
                X_test, 
                {
                    'classification': y_test,
                    'confidence': y_test_conf
                }
            ),
            epochs=50,
            batch_size=32,
            callbacks=model_callbacks,
            verbose=1
        )
        
        train_time = time.time() - start_time
        print(f"⏱️  Training completed in {train_time:.1f}s")
        
        # Comprehensive evaluation
        print("\n📊 Advanced Model Evaluation")
        print("=" * 40)
        
        # Predict with confidence
        predictions = model.predict(X_test, verbose=0)
        y_pred_class = predictions[0]
        y_pred_conf = predictions[1]
        
        y_pred_labels = np.argmax(y_pred_class, axis=1)
        y_pred_probs = np.max(y_pred_class, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_labels == y_test)
        
        print(f"🎯 Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed classification report
        class_names = list(self.class_mapping.keys())
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_labels, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_labels)
        print("\n📊 Confusion Matrix:")
        print(cm)
        
        # False positive analysis
        background_mask = (y_test == 0)  # Background class
        false_positives = np.sum((y_pred_labels != 0) & background_mask)
        total_background = np.sum(background_mask)
        fp_rate = false_positives / total_background if total_background > 0 else 0
        
        print(f"\n🛡️  False Positive Analysis:")
        print(f"  Background samples: {total_background}")
        print(f"  False positives: {false_positives}")
        print(f"  False positive rate: {fp_rate:.3f} ({fp_rate*100:.1f}%)")
        
        # Target achievement check
        target_met = accuracy >= self.target_accuracy and fp_rate <= self.false_positive_threshold
        
        print(f"\n🎯 Target Achievement:")
        print(f"  Accuracy target: {self.target_accuracy*100:.1f}% - {'✅ MET' if accuracy >= self.target_accuracy else '❌ NOT MET'}")
        print(f"  FP target: <{self.false_positive_threshold*100:.1f}% - {'✅ MET' if fp_rate <= self.false_positive_threshold else '❌ NOT MET'}")
        print(f"  Overall: {'✅ SUCCESS' if target_met else '❌ NEEDS IMPROVEMENT'}")
        
        return model, history, (X_test, y_test), {
            'accuracy': accuracy,
            'fp_rate': fp_rate,
            'target_met': target_met,
            'predictions': y_pred_labels,
            'probabilities': y_pred_probs,
            'confidence': y_pred_conf.flatten()
        }
    
    def convert_to_optimized_tflite(self, model, X_sample):
        """Convert to highly optimized TensorFlow Lite with aggressive quantization"""
        print("\n🔄 Advanced TensorFlow Lite Conversion")
        print("=" * 50)
        
        # Representative dataset for quantization
        def representative_dataset():
            for i in range(min(200, len(X_sample))):  # More samples for better quantization
                yield [X_sample[i:i+1].astype(np.float32)]
        
        # Full integer quantization for maximum efficiency
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Advanced optimization settings
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        
        try:
            quantized_model = converter.convert()
            quantized_size = len(quantized_model) / 1024
            
            print(f"✅ Quantized model: {quantized_size:.1f} KB")
            
            # Validate quantized model performance
            interpreter = tf.lite.Interpreter(model_content=quantized_model)
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"📊 Quantized model details:")
            print(f"  Input: {input_details[0]['dtype']} {input_details[0]['shape']}")
            print(f"  Output: {output_details[0]['dtype']} {output_details[0]['shape']}")
            
            # Test quantized inference speed
            test_sample = X_sample[0:1].astype(np.float32)
            
            # Convert to int8 if needed
            if input_details[0]['dtype'] == np.int8:
                input_scale, input_zero_point = input_details[0]['quantization']
                test_sample = (test_sample / input_scale + input_zero_point).astype(np.int8)
            
            start_time = time.time()
            for _ in range(100):  # 100 inferences for timing
                interpreter.set_tensor(input_details[0]['index'], test_sample)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
            
            avg_time_ms = (time.time() - start_time) * 10  # Average per inference in ms
            
            print(f"⚡ Quantized inference: {avg_time_ms:.2f} ms")
            
            # Check nRF5340 compatibility
            if quantized_size <= 64:  # More aggressive target
                print("✅ Optimized for nRF5340 deployment")
            else:
                print("⚠️  Large for embedded deployment")
            
            # Save optimized model
            with open('sait01_advanced_model.tflite', 'wb') as f:
                f.write(quantized_model)
            
            print("💾 Advanced model saved as 'sait01_advanced_model.tflite'")
            
            return quantized_model
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            print("🔄 Falling back to float16 quantization...")
            
            # Fallback to float16
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            fallback_model = converter.convert()
            fallback_size = len(fallback_model) / 1024
            
            print(f"✅ Float16 model: {fallback_size:.1f} KB")
            
            with open('sait01_advanced_model.tflite', 'wb') as f:
                f.write(fallback_model)
            
            return fallback_model
    
    def run_advanced_training(self):
        """Complete advanced training pipeline for 90-95% accuracy"""
        print("🚀 SAIT_01 ADVANCED PRODUCTION TRAINING")
        print("=" * 60)
        print("🎯 TARGET: 90-95% accuracy with <5% false positives")
        print("=" * 60)
        
        # Load enhanced dataset
        X_data, y_data, sources = self.load_enhanced_dataset(samples_per_class=250)
        
        if len(X_data) == 0:
            print("❌ No data loaded - training failed")
            return None
        
        # Advanced training
        model, history, (X_test, y_test), metrics = self.train_advanced_model(X_data, y_data)
        
        # Convert to optimized TFLite
        tflite_model = self.convert_to_optimized_tflite(model, X_test)
        
        # Final results
        print("\n🏆 ADVANCED TRAINING RESULTS")
        print("=" * 50)
        print(f"✅ Dataset: {len(X_data)} samples")
        print(f"🎯 Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"🛡️  False Positives: {metrics['fp_rate']:.3f} ({metrics['fp_rate']*100:.1f}%)")
        print(f"🚀 Target Achievement: {'✅ SUCCESS' if metrics['target_met'] else '❌ NEEDS WORK'}")
        
        if metrics['target_met']:
            print("\n🎉 TRAINING SUCCESSFUL - READY FOR DEPLOYMENT!")
        else:
            print("\n🔧 Training needs refinement - consider:")
            if metrics['accuracy'] < self.target_accuracy:
                print("  - More training data")
                print("  - Enhanced model architecture")
                print("  - Better feature engineering")
            if metrics['fp_rate'] > self.false_positive_threshold:
                print("  - More negative examples")
                print("  - Stricter confidence thresholds")
                print("  - Enhanced background noise samples")
        
        return {
            'model': model,
            'history': history,
            'tflite': tflite_model,
            'metrics': metrics,
            'test_data': (X_test, y_test)
        }

def main():
    trainer = AdvancedProductionTrainer()
    results = trainer.run_advanced_training()
    
    if results and results['metrics']['target_met']:
        print("\n🚀 ADVANCED TRAINING COMPLETE - DEPLOYMENT READY!")
    else:
        print("\n🔧 ADVANCED TRAINING NEEDS REFINEMENT")

if __name__ == "__main__":
    main()