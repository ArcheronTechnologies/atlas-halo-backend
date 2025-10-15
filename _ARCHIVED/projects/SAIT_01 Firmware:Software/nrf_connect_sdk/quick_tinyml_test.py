#!/usr/bin/env python3
"""
Quick TinyML Testing - Minimal model training for rapid validation
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, MODEL_CONFIG

def quick_audio_test():
    """Quick test of audio preprocessing pipeline"""
    print("🎵 Quick Audio Preprocessing Test")
    print("=" * 40)
    
    data_dir = "edth-copenhagen-drone-acoustics/data/raw"
    if not os.path.exists(data_dir):
        print("❌ Dataset not found")
        return False
        
    preprocessor = SaitAudioPreprocessor()
    
    # Test on a few sample files
    test_files = []
    for class_name in ['background', 'drone', 'helicopter']:
        class_path = Path(data_dir) / 'train' / class_name
        if class_path.exists():
            files = list(class_path.glob("*.wav"))[:2]
            test_files.extend([(f, class_name) for f in files])
            
    if not test_files:
        print("❌ No test files found")
        return False
        
    print(f"Testing {len(test_files)} audio files...")
    
    processing_times = []
    processed_shapes = []
    
    for audio_file, class_name in test_files:
        try:
            start_time = time.time()
            
            # Load and preprocess
            audio = preprocessor.load_and_resample(str(audio_file))
            mel_spec = preprocessor.extract_mel_spectrogram(audio)
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            processed_shapes.append(mel_spec.shape)
            
            print(f"  ✅ {class_name}: {audio_file.name} -> {mel_spec.shape} ({processing_time:.1f}ms)")
            
        except Exception as e:
            print(f"  ❌ {class_name}: {audio_file.name} -> Error: {e}")
            return False
            
    # Verify shapes are consistent
    if len(set(processed_shapes)) == 1:
        print(f"✅ All shapes consistent: {processed_shapes[0]}")
    else:
        print(f"❌ Inconsistent shapes: {set(processed_shapes)}")
        return False
        
    avg_time = np.mean(processing_times)
    print(f"⏱️  Average preprocessing: {avg_time:.1f}ms")
    
    if avg_time < 50:  # 50ms threshold
        print("✅ Preprocessing is fast enough for real-time")
        return True
    else:
        print("⚠️  Preprocessing may be slow for real-time")
        return True

def create_simple_model():
    """Create a simplified model for quick testing"""
    print("\n🧠 Building Simplified CNN Model")
    print("=" * 40)
    
    # Simplified architecture for quick testing
    inputs = tf.keras.layers.Input(shape=(63, 64, 1), name='audio_input')
    
    # Simple CNN layers
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output for 3 classes (background, drone, helicopter)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='classification')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SimpleCNN')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📝 Model Summary:")
    model.summary()
    
    # Calculate model size
    model_size = model.count_params() * 4 / 1024  # KB
    print(f"📏 Model size: {model_size:.1f} KB")
    
    return model

def load_quick_dataset():
    """Load a small subset of the dataset for quick testing"""
    print("\n📊 Loading Quick Dataset")
    print("=" * 40)
    
    data_dir = "edth-copenhagen-drone-acoustics/data/raw"
    preprocessor = SaitAudioPreprocessor()
    
    # Simplified class mapping (0: background, 1: drone, 2: helicopter)
    class_mapping = {'background': 0, 'drone': 1, 'helicopter': 2}
    
    X_data = []
    y_data = []
    
    # Load small subset from each class
    samples_per_class = 10  # Only 10 samples per class for quick test
    
    for class_name, class_id in class_mapping.items():
        class_path = Path(data_dir) / 'train' / class_name
        if not class_path.exists():
            print(f"⚠️  Class directory not found: {class_name}")
            continue
            
        files = list(class_path.glob("*.wav"))[:samples_per_class]
        print(f"Loading {len(files)} {class_name} samples...")
        
        for audio_file in files:
            try:
                audio = preprocessor.load_and_resample(str(audio_file))
                mel_spec = preprocessor.extract_mel_spectrogram(audio)
                
                X_data.append(mel_spec)
                y_data.append(class_id)
                
            except Exception as e:
                print(f"  ❌ Error loading {audio_file.name}: {e}")
                
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"📊 Dataset summary:")
    print(f"  Samples: {len(X_data)}")
    print(f"  Shape: {X_data.shape}")
    print(f"  Classes: {np.unique(y_data)}")
    print(f"  Distribution: {np.bincount(y_data)}")
    
    return X_data, y_data

def quick_training_test(model, X_data, y_data):
    """Quick training test with limited epochs"""
    print("\n🚀 Quick Training Test")
    print("=" * 40)
    
    if len(X_data) < 10:
        print("❌ Not enough data for training")
        return None
        
    # Split into train/test
    split_idx = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Quick training with few epochs
    print("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=3,  # Only 3 epochs for quick test
        batch_size=8,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f} seconds")
    
    # Quick evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 Test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    return history

def test_tflite_conversion(model):
    """Test TensorFlow Lite conversion"""
    print("\n🔄 Testing TensorFlow Lite Conversion")
    print("=" * 40)
    
    try:
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        tflite_size = len(tflite_model) / 1024
        
        print(f"✅ TFLite conversion successful")
        print(f"📏 TFLite model size: {tflite_size:.1f} KB")
        
        # Test quantization
        converter.target_spec.supported_types = [tf.int8]
        quantized_model = converter.convert()
        quantized_size = len(quantized_model) / 1024
        
        print(f"✅ INT8 quantization successful")
        print(f"📏 Quantized size: {quantized_size:.1f} KB")
        print(f"🗜️  Compression: {((tflite_size - quantized_size) / tflite_size * 100):.1f}%")
        
        if quantized_size <= 80:  # 80KB target
            print("✅ Model fits nRF5340 memory constraints")
        else:
            print("⚠️  Model may be too large for nRF5340")
            
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def main():
    """Run quick TinyML validation tests"""
    print("🎯 SAIT_01 Quick TinyML Validation")
    print("=" * 50)
    
    # 1. Test audio preprocessing
    if not quick_audio_test():
        print("❌ Audio preprocessing test failed")
        return
        
    # 2. Create simple model
    model = create_simple_model()
    
    # 3. Load small dataset
    X_data, y_data = load_quick_dataset()
    
    if len(X_data) == 0:
        print("❌ No data loaded - cannot continue")
        return
        
    # 4. Quick training test
    history = quick_training_test(model, X_data, y_data)
    
    if history is None:
        print("❌ Training test failed")
        return
        
    # 5. Test TFLite conversion
    if test_tflite_conversion(model):
        print("\n✅ All quick tests passed!")
        print("\n📋 Summary:")
        print("  ✅ Audio preprocessing working")
        print("  ✅ Model architecture valid") 
        print("  ✅ Training pipeline functional")
        print("  ✅ TensorFlow Lite conversion successful")
        print("\n🚀 Ready for full model training and evaluation!")
    else:
        print("\n⚠️  Some tests had issues - check TFLite conversion")

if __name__ == "__main__":
    main()