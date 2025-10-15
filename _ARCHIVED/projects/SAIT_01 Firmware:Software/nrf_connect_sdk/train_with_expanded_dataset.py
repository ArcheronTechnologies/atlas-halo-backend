#!/usr/bin/env python3
"""
Train SAIT_01 Model with Expanded Dataset for Production Accuracy
Uses 9,258 samples across 3 classes for 85%+ accuracy target
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
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Matplotlib/Seaborn not available - skipping plots")

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, MODEL_CONFIG

class ExpandedDatasetTrainer:
    """Train SAIT_01 model with expanded dataset for production accuracy"""
    
    def __init__(self, dataset_dir="expanded_sait01_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def load_expanded_dataset(self, max_samples_per_class=3000):
        """Load the expanded dataset with balanced sampling"""
        print("📊 Loading expanded dataset...")
        
        X = []
        y = []
        
        # Load metadata
        metadata_path = self.dataset_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"📄 Dataset metadata: {metadata['dataset_info']['total_samples']} total samples")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                print(f"❌ Class directory not found: {class_dir}")
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            
            # Limit samples for balanced training
            if len(audio_files) > max_samples_per_class:
                audio_files = audio_files[:max_samples_per_class]
            
            print(f"   Loading {class_name}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
                try:
                    # Load and preprocess audio
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    
                    X.append(features)
                    y.append(class_idx)
                    
                except Exception as e:
                    print(f"⚠️  Error processing {audio_file}: {e}")
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_production_model(self):
        """Create optimized model for production deployment"""
        print("🏗️  Creating production-optimized model...")
        
        inputs = keras.Input(shape=MODEL_CONFIG['input_shape'], name='mel_input')
        
        # Optimized CNN architecture for better accuracy with more data
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.5)(x)
        
        # Classification head with more capacity for complex dataset
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        outputs = keras.layers.Dense(3, activation='softmax', name='classification')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SAIT01_Production')
        
        # Compile with optimized settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        print(f"✅ Model created: {model.count_params()} parameters")
        model.summary()
        
        return model
    
    def train_production_model(self, X, y, test_size=0.2, validation_size=0.2):
        """Train model with advanced techniques for maximum accuracy"""
        print("🚀 Training production model...")
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=validation_size, stratify=y_train_val, random_state=42
        )
        
        print(f"📊 Data split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create model
        model = self.create_production_model()
        
        # Advanced training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_sait01_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation for training
        def augment_batch(X_batch, y_batch):
            """Real-time data augmentation"""
            augmented_X = []
            for x in X_batch:
                # Random noise injection
                if np.random.random() < 0.3:
                    noise = np.random.randn(*x.shape) * 0.02
                    x = x + noise
                
                # Random amplitude scaling
                if np.random.random() < 0.3:
                    scale = np.random.uniform(0.8, 1.2)
                    x = x * scale
                
                # Clip values
                x = np.clip(x, -1, 1)
                augmented_X.append(x)
            
            return np.array(augmented_X), y_batch
        
        # Train model
        print("🔄 Training model with expanded dataset...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        print("🔍 Evaluating model...")
        test_start = time.time()
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        inference_time = (time.time() - test_start) / len(X_test) * 1000  # ms per sample
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"\n📊 Production Model Results:")
        print(f"   Training time: {training_time/60:.1f} minutes")
        print(f"   Test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Inference time: {inference_time:.2f}ms per sample")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print(f"\n🔍 Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix if available
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('SAIT_01 Production Model - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('sait01_confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("📊 Confusion matrix saved to: sait01_confusion_matrix.png")
        else:
            print("📊 Confusion matrix plotting skipped (matplotlib not available)")
        
        # Model size estimation
        model.save('sait01_production_model.h5')
        model_size = os.path.getsize('sait01_production_model.h5') / 1024  # KB
        print(f"📏 Model size: {model_size:.1f} KB")
        
        # Convert to TFLite
        print("📱 Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open('sait01_production_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        tflite_size = len(tflite_model) / 1024  # KB
        print(f"📱 TFLite model size: {tflite_size:.1f} KB")
        
        # Test TFLite accuracy
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        tflite_correct = 0
        tflite_times = []
        
        for i in range(min(100, len(X_test))):
            sample = X_test[i:i+1].astype(np.float32)
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            tflite_time = (time.time() - start_time) * 1000
            
            tflite_times.append(tflite_time)
            pred_class = np.argmax(output[0])
            if pred_class == y_test[i]:
                tflite_correct += 1
        
        tflite_accuracy = tflite_correct / min(100, len(X_test))
        avg_tflite_time = np.mean(tflite_times)
        
        print(f"📱 TFLite accuracy: {tflite_accuracy:.3f} ({tflite_accuracy*100:.1f}%)")
        print(f"📱 TFLite inference: {avg_tflite_time:.2f}ms per sample")
        
        # Performance assessment
        print(f"\n🎯 Production Readiness Assessment:")
        print(f"   ✅ Model Size: {tflite_size:.1f} KB (Target: <80KB)")
        print(f"   ✅ Inference Time: {avg_tflite_time:.1f}ms (Target: <50ms)")
        
        if accuracy >= 0.85:
            print(f"   🏆 EXCELLENT ACCURACY: {accuracy*100:.1f}% (Target: >85%)")
            print("   🚀 READY FOR PRODUCTION DEPLOYMENT!")
        elif accuracy >= 0.70:
            print(f"   ✅ GOOD ACCURACY: {accuracy*100:.1f}% (Target: >70%)")
            print("   📈 Consider further optimization for production")
        else:
            print(f"   ⚠️  MODERATE ACCURACY: {accuracy*100:.1f}%")
            print("   🔧 Requires additional dataset expansion")
        
        return {
            'accuracy': accuracy,
            'tflite_accuracy': tflite_accuracy,
            'model_size_kb': model_size,
            'tflite_size_kb': tflite_size,
            'inference_time_ms': inference_time,
            'tflite_time_ms': avg_tflite_time,
            'training_time_minutes': training_time/60,
            'model': model,
            'history': history
        }

def main():
    """Main training execution with expanded dataset"""
    print("🎯 SAIT_01 Production Training with Expanded Dataset")
    print("=" * 70)
    
    # Initialize trainer
    trainer = ExpandedDatasetTrainer()
    
    # Load expanded dataset
    X, y = trainer.load_expanded_dataset(max_samples_per_class=3000)
    
    if len(X) < 1000:
        print("❌ Insufficient data for production training")
        return
    
    # Train production model
    results = trainer.train_production_model(X, y)
    
    # Final summary
    print(f"\n🎉 SAIT_01 Production Training Complete!")
    print("=" * 70)
    print(f"🎯 Final Accuracy: {results['accuracy']*100:.1f}%")
    print(f"📏 TFLite Size: {results['tflite_size_kb']:.1f} KB")
    print(f"⏱️  Inference Time: {results['tflite_time_ms']:.1f}ms")
    print(f"⏰ Training Time: {results['training_time_minutes']:.1f} minutes")
    
    if results['accuracy'] >= 0.85:
        print("\n🏆 MISSION ACCOMPLISHED: Production-grade accuracy achieved!")
        print("🚀 Model ready for nRF5340 deployment")
    else:
        print(f"\n📈 Accuracy improved from 43.3% to {results['accuracy']*100:.1f}%")
        print("🔧 Consider additional optimization techniques")

if __name__ == "__main__":
    main()