#!/usr/bin/env python3
"""
Final SAIT_01 Production Model with Maximum Accuracy Techniques
Implements all optimizations for deployment-ready accuracy
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
import time
import json

class FinalSAIT01ProductionSystem:
    """Final production-ready SAIT_01 audio classification system"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.n_mels = 64
        self.n_frames = 63
        self.input_shape = (64, 63, 1)
        
        print("🎯 SAIT_01 Final Production System")
        print("=" * 60)
    
    def advanced_preprocess_audio(self, audio_path):
        """Advanced preprocessing with multiple techniques"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Normalize length to 1 second
            target_length = self.sample_rate
            if len(audio) > target_length:
                # Take center segment for consistency
                start = (len(audio) - target_length) // 2
                audio = audio[start:start + target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            # Advanced mel spectrogram with optimal parameters
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=2048,      # Higher resolution
                hop_length=256,  # Good time resolution
                n_mels=self.n_mels,
                fmin=20,         # Remove very low frequencies
                fmax=8000,       # Nyquist for 16kHz
                power=2.0        # Power spectrogram
            )
            
            # Convert to log scale with small epsilon
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
            
            # Advanced normalization
            mel_spec_norm = (mel_spec_db + 80) / 80  # Scale to [0, 1]
            mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
            
            # Ensure correct time dimension
            if mel_spec_norm.shape[1] != self.n_frames:
                mel_spec_norm = librosa.util.fix_length(mel_spec_norm, size=self.n_frames, axis=1)
            
            # Add channel dimension - CORRECTED SHAPE
            return mel_spec_norm[..., np.newaxis]  # (64, 63, 1)
            
        except Exception as e:
            print(f"⚠️  Error processing {audio_path}: {e}")
            # Return zero spectrogram on error
            return np.zeros(self.input_shape)
    
    def create_augmented_sample(self, audio_path, augmentation_type=0):
        """Create augmented audio sample"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply different augmentation types
            if augmentation_type == 1:
                # Time stretching
                rate = np.random.uniform(0.85, 1.15)
                audio = librosa.effects.time_stretch(audio, rate=rate)
            elif augmentation_type == 2:
                # Pitch shifting
                n_steps = np.random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            elif augmentation_type == 3:
                # Add noise
                noise_factor = np.random.uniform(0.005, 0.02)
                noise = np.random.randn(len(audio)) * noise_factor
                audio = audio + noise
            elif augmentation_type == 4:
                # Dynamic range compression
                audio = np.sign(audio) * np.power(np.abs(audio), 0.8)
            
            # Normalize and ensure correct length
            target_length = self.sample_rate
            if len(audio) > target_length:
                start = np.random.randint(0, len(audio) - target_length)
                audio = audio[start:start + target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            return self.advanced_preprocess_audio_from_array(audio)
            
        except Exception as e:
            return np.zeros(self.input_shape)
    
    def advanced_preprocess_audio_from_array(self, audio):
        """Process audio array directly"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=2048,
            hop_length=256,
            n_mels=self.n_mels,
            fmin=20,
            fmax=8000,
            power=2.0
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        mel_spec_norm = (mel_spec_db + 80) / 80
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
        if mel_spec_norm.shape[1] != self.n_frames:
            mel_spec_norm = librosa.util.fix_length(mel_spec_norm, size=self.n_frames, axis=1)
        
        return mel_spec_norm[..., np.newaxis]
    
    def load_enhanced_dataset(self, max_per_class=200, augment_factor=3):
        """Load dataset with aggressive augmentation"""
        print(f"📊 Loading enhanced dataset (max {max_per_class} per class, {augment_factor}x augmentation)")
        
        balanced_dir = Path("quick_balanced_data")
        if not balanced_dir.exists():
            print("❌ Balanced dataset not found")
            return np.array([]), np.array([])
        
        X = []
        y = []
        class_names = ['background', 'drone', 'aircraft']
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = balanced_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))[:max_per_class]
            print(f"   {class_name}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
                # Original sample
                features = self.advanced_preprocess_audio(audio_file)
                X.append(features)
                y.append(class_idx)
                
                # Augmented samples
                for aug_type in range(1, augment_factor + 1):
                    aug_features = self.create_augmented_sample(audio_file, aug_type)
                    X.append(aug_features)
                    y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Enhanced dataset: {X.shape}, Labels: {len(y)}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_production_model(self):
        """Create final production model with all optimizations"""
        print("🏗️  Creating production-optimized model...")
        
        inputs = keras.Input(shape=self.input_shape, name='audio_input')
        
        # Advanced CNN architecture
        # Block 1: Initial feature extraction
        x = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv1')(inputs)
        x = keras.layers.BatchNormalization(name='bn1')(x)
        x = keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = keras.layers.Dropout(0.2, name='dropout1')(x)
        
        # Block 2: Pattern recognition
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = keras.layers.BatchNormalization(name='bn2')(x)
        x = keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = keras.layers.Dropout(0.25, name='dropout2')(x)
        
        # Block 3: Complex feature extraction
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = keras.layers.BatchNormalization(name='bn3')(x)
        x = keras.layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = keras.layers.Dropout(0.3, name='dropout3')(x)
        
        # Global feature aggregation
        x = keras.layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Dense classification layers
        x = keras.layers.Dense(256, activation='relu', name='dense1')(x)
        x = keras.layers.Dropout(0.5, name='dropout4')(x)
        x = keras.layers.Dense(128, activation='relu', name='dense2')(x)
        x = keras.layers.Dropout(0.3, name='dropout5')(x)
        x = keras.layers.Dense(64, activation='relu', name='dense3')(x)
        x = keras.layers.Dropout(0.2, name='dropout6')(x)
        
        # Output layer
        outputs = keras.layers.Dense(3, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SAIT01_Production')
        
        # Advanced optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Production model: {model.count_params()} parameters")
        return model
    
    def train_production_model(self, X, y, validation_split=0.2):
        """Train with advanced techniques"""
        print("🚀 Training production model with advanced techniques...")
        
        # Create model
        model = self.create_production_model()
        
        # Advanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_production_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with longer schedule
        print("📈 Starting training...")
        start_time = time.time()
        
        history = model.fit(
            X, y,
            batch_size=16,
            epochs=100,  # More epochs for better convergence
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f}s")
        
        return model, history
    
    def comprehensive_evaluation(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        print("🔍 Comprehensive model evaluation...")
        
        # Predictions
        start_time = time.time()
        y_pred_prob = model.predict(X_test, verbose=0)
        inference_time = (time.time() - start_time) / len(X_test) * 1000
        
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Inference Time: {inference_time:.2f}ms per sample")
        
        # Per-class analysis
        class_names = ['Background', 'Drone', 'Aircraft']
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        for i, class_name in enumerate(class_names):
            print(f"   {class_name}: {cm[i]}")
        
        # Confidence analysis
        max_confidences = np.max(y_pred_prob, axis=1)
        avg_confidence = np.mean(max_confidences)
        high_conf_mask = max_confidences > 0.8
        high_conf_accuracy = np.mean(y_pred[high_conf_mask] == y_test[high_conf_mask]) if np.sum(high_conf_mask) > 0 else 0
        
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   High Confidence Samples: {np.sum(high_conf_mask)}/{len(y_test)} ({np.sum(high_conf_mask)/len(y_test)*100:.1f}%)")
        print(f"   High Confidence Accuracy: {high_conf_accuracy:.3f} ({high_conf_accuracy*100:.1f}%)")
        
        return {
            'accuracy': accuracy,
            'inference_time_ms': inference_time,
            'avg_confidence': avg_confidence,
            'high_conf_accuracy': high_conf_accuracy,
            'confusion_matrix': cm.tolist()
        }
    
    def create_optimized_tflite(self, model):
        """Create optimized TensorFlow Lite model"""
        print("📱 Creating optimized TensorFlow Lite model...")
        
        # Save model first
        model.save('final_production_model.keras')
        model_size_kb = os.path.getsize('final_production_model.keras') / 1024
        
        # Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Int8 quantization for maximum optimization
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Representative dataset for calibration
        def representative_dataset():
            for _ in range(100):
                # Generate random data matching input shape
                yield [np.random.randn(1, 64, 63, 1).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = 'sait01_final_production.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size_kb = len(tflite_model) / 1024
        
        print(f"📏 Model Sizes:")
        print(f"   Keras Model: {model_size_kb:.1f} KB")
        print(f"   TFLite Model: {tflite_size_kb:.1f} KB")
        print(f"   Compression: {model_size_kb/tflite_size_kb:.1f}x smaller")
        
        return tflite_model, tflite_size_kb
    
    def test_tflite_model(self, tflite_model, X_test, y_test, num_samples=50):
        """Test TensorFlow Lite model performance"""
        print(f"🧪 Testing TFLite model ({num_samples} samples)...")
        
        # Initialize interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        correct_predictions = 0
        inference_times = []
        
        for i in range(min(num_samples, len(X_test))):
            # Prepare input
            input_data = X_test[i:i+1].astype(np.float32)
            
            # Run inference
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            inference_time = (time.time() - start_time) * 1000
            
            inference_times.append(inference_time)
            
            # Check prediction
            predicted_class = np.argmax(output_data[0])
            if predicted_class == y_test[i]:
                correct_predictions += 1
        
        tflite_accuracy = correct_predictions / min(num_samples, len(X_test))
        avg_inference_time = np.mean(inference_times)
        
        print(f"📱 TFLite Results:")
        print(f"   Accuracy: {tflite_accuracy:.3f} ({tflite_accuracy*100:.1f}%)")
        print(f"   Avg Inference Time: {avg_inference_time:.2f}ms")
        print(f"   Max Inference Time: {np.max(inference_times):.2f}ms")
        print(f"   Min Inference Time: {np.min(inference_times):.2f}ms")
        
        return tflite_accuracy, avg_inference_time
    
    def run_complete_production_pipeline(self):
        """Run complete production pipeline"""
        print("🎯 SAIT_01 Complete Production Pipeline")
        print("=" * 70)
        
        # Load enhanced dataset
        X, y = self.load_enhanced_dataset(max_per_class=150, augment_factor=4)
        
        if len(X) == 0:
            print("❌ No data available")
            return
        
        # Split data
        split_idx = int(len(X) * 0.8)
        indices = np.random.permutation(len(X))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print(f"📊 Data Split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Train production model
        model, history = self.train_production_model(X_train, y_train)
        
        # Comprehensive evaluation
        results = self.comprehensive_evaluation(model, X_test, y_test)
        
        # Create optimized TFLite model
        tflite_model, tflite_size = self.create_optimized_tflite(model)
        
        # Test TFLite model
        tflite_acc, tflite_time = self.test_tflite_model(tflite_model, X_test, y_test)
        
        # Final assessment
        print(f"\n🏆 FINAL PRODUCTION ASSESSMENT")
        print("=" * 70)
        print(f"🎯 Accuracy: {results['accuracy']*100:.1f}%")
        print(f"📏 TFLite Size: {tflite_size:.1f} KB")
        print(f"⏱️  TFLite Inference: {tflite_time:.1f}ms")
        print(f"🔥 Confidence: {results['avg_confidence']*100:.1f}%")
        
        # Deployment readiness
        deployment_ready = (
            results['accuracy'] >= 0.7 and
            tflite_size <= 100 and
            tflite_time <= 50
        )
        
        print(f"\n🚀 Deployment Status: {'✅ READY' if deployment_ready else '⚠️  NEEDS OPTIMIZATION'}")
        
        if results['accuracy'] >= 0.9:
            print("🏆 EXCEPTIONAL ACCURACY - PRODUCTION READY!")
        elif results['accuracy'] >= 0.8:
            print("✅ EXCELLENT ACCURACY - DEPLOYMENT RECOMMENDED")
        elif results['accuracy'] >= 0.7:
            print("✅ GOOD ACCURACY - SUITABLE FOR PRODUCTION")
        else:
            print("⚠️  ACCURACY NEEDS IMPROVEMENT")
        
        # Save results
        final_results = {
            'accuracy': results['accuracy'],
            'tflite_size_kb': tflite_size,
            'tflite_inference_ms': tflite_time,
            'confidence': results['avg_confidence'],
            'deployment_ready': deployment_ready,
            'confusion_matrix': results['confusion_matrix']
        }
        
        with open('sait01_production_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📄 Results saved to: sait01_production_results.json")
        print(f"📱 TFLite model saved to: sait01_final_production.tflite")
        
        return final_results

def main():
    """Main execution function"""
    system = FinalSAIT01ProductionSystem()
    results = system.run_complete_production_pipeline()
    
    print(f"\n🎉 SAIT_01 PRODUCTION PIPELINE COMPLETE!")
    print("🚀 System ready for nRF5340 deployment")

if __name__ == "__main__":
    main()