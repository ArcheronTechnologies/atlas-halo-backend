#!/usr/bin/env python3
"""
🛡️ Fix SAIT_01 Model Loading Issues
====================================
Fix INT8/FLOAT32 data type mismatch and create properly quantized models
"""

import tensorflow as tf
import numpy as np
import os

def fix_model_quantization():
    """Fix model quantization issues for proper deployment"""
    print("🔧 Fixing SAIT_01 Model Loading Issues")
    print("=" * 50)
    
    # Check for trained model
    model_path = "final_best_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found")
        return False
    
    print(f"✅ Found trained model: {model_path}")
    
    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Loaded model successfully")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Create representative dataset for quantization
    def representative_data_gen():
        """Generate representative data for INT8 quantization"""
        for _ in range(100):
            # Generate random mel-spectrogram-like data
            sample = np.random.randn(1, 63, 64, 1).astype(np.float32)
            yield [sample]
    
    # Convert to TensorFlow Lite with proper quantization
    print("\n🔄 Converting to TensorFlow Lite with INT8 quantization...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_quantized_model = converter.convert()
        
        # Save the quantized model
        quantized_path = "sait01_fixed_quantized.tflite"
        with open(quantized_path, 'wb') as f:
            f.write(tflite_quantized_model)
        
        print(f"✅ Saved quantized model: {quantized_path}")
        print(f"📏 Model size: {len(tflite_quantized_model) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"⚠️ INT8 quantization failed: {e}")
        print("🔄 Falling back to FLOAT32 model...")
        
        # Create FLOAT32 version
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        float32_path = "sait01_fixed_float32.tflite"
        with open(float32_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Saved FLOAT32 model: {float32_path}")
        print(f"📏 Model size: {len(tflite_model) / 1024:.1f} KB")
    
    # Test the model loading
    print("\n🧪 Testing model loading...")
    
    # Test quantized model if available
    test_models = ["sait01_fixed_quantized.tflite", "sait01_fixed_float32.tflite"]
    
    for model_file in test_models:
        if os.path.exists(model_file):
            print(f"\n📊 Testing {model_file}:")
            
            try:
                interpreter = tf.lite.Interpreter(model_path=model_file)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                print(f"   Input shape: {input_details[0]['shape']}")
                print(f"   Input dtype: {input_details[0]['dtype']}")
                print(f"   Output shape: {output_details[0]['shape']}")
                print(f"   Output dtype: {output_details[0]['dtype']}")
                
                # Test inference
                if input_details[0]['dtype'] == np.int8:
                    # INT8 model
                    test_input = np.random.randint(-128, 127, input_details[0]['shape'], dtype=np.int8)
                else:
                    # FLOAT32 model
                    test_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)
                
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                print(f"   Test inference: ✅ SUCCESS")
                print(f"   Output range: [{np.min(output_data):.3f}, {np.max(output_data):.3f}]")
                
            except Exception as e:
                print(f"   Test inference: ❌ FAILED - {e}")
    
    print(f"\n🎯 Model fixing complete!")
    return True

def create_enhanced_inference_wrapper():
    """Create enhanced inference wrapper that handles data type conversion"""
    
    wrapper_code = '''#!/usr/bin/env python3
"""
🛡️ Enhanced TensorFlow Lite Inference Wrapper
==============================================
Handles data type conversions and preprocessing for SAIT_01 models
"""

import numpy as np
import tensorflow as tf
import librosa

class SAIT01ModelWrapper:
    """Enhanced model wrapper with automatic data type handling"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_scale = None
        self.input_zero_point = None
        self.output_scale = None
        self.output_zero_point = None
        
        self._load_model()
    
    def _load_model(self):
        """Load TensorFlow Lite model with proper configuration"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get quantization parameters
            input_details = self.input_details[0]
            if input_details['dtype'] == np.int8:
                self.input_scale, self.input_zero_point = input_details['quantization']
                
            output_details = self.output_details[0]
            if output_details['dtype'] == np.int8:
                self.output_scale, self.output_zero_point = output_details['quantization']
            
            print(f"✅ Model loaded: {self.model_path}")
            print(f"📊 Input: {input_details['shape']} ({input_details['dtype']})")
            print(f"📊 Output: {output_details['shape']} ({output_details['dtype']})")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def preprocess_audio(self, audio_data, sample_rate=16000):
        """Preprocess audio to model input format"""
        try:
            # Extract mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=63,
                n_fft=1024,
                hop_length=256,
                window='hann'
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
            
            # Ensure consistent shape (63, 64)
            if log_mel.shape[1] < 64:
                padding = 64 - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, padding)), mode='constant')
            elif log_mel.shape[1] > 64:
                log_mel = log_mel[:, :64]
            
            # Add batch and channel dimensions
            features = log_mel[np.newaxis, ..., np.newaxis]  # Shape: (1, 63, 64, 1)
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            return None
    
    def predict(self, audio_data):
        """Run inference on audio data with proper data type handling"""
        try:
            # Preprocess audio
            features = self.preprocess_audio(audio_data)
            if features is None:
                return None
            
            # Handle quantization if needed
            input_dtype = self.input_details[0]['dtype']
            
            if input_dtype == np.int8:
                # Convert to INT8
                if self.input_scale is not None:
                    features_quantized = features / self.input_scale + self.input_zero_point
                    features_quantized = np.clip(features_quantized, -128, 127).astype(np.int8)
                else:
                    # Fallback quantization
                    features_quantized = np.clip(features * 127, -128, 127).astype(np.int8)
                
                input_data = features_quantized
            else:
                input_data = features
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Handle dequantization if needed
            if self.output_details[0]['dtype'] == np.int8:
                if self.output_scale is not None:
                    output_data = self.output_scale * (output_data - self.output_zero_point)
                else:
                    output_data = output_data.astype(np.float32) / 127.0
            
            # Apply softmax to get probabilities
            probabilities = tf.nn.softmax(output_data[0]).numpy()
            
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            return {
                'predicted_class': int(predicted_class),
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'source': 'enhanced_wrapper'
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return None

def test_enhanced_wrapper():
    """Test the enhanced model wrapper"""
    print("🧪 Testing Enhanced Model Wrapper")
    print("=" * 40)
    
    # Test with available models
    test_models = [
        "sait01_fixed_quantized.tflite",
        "sait01_fixed_float32.tflite",
        "sait01_final_high_accuracy.tflite"
    ]
    
    for model_file in test_models:
        if os.path.exists(model_file):
            print(f"\\n📊 Testing {model_file}:")
            
            try:
                wrapper = SAIT01ModelWrapper(model_file)
                
                # Generate test audio
                test_audio = np.random.randn(16000) * 0.1  # 1 second of noise
                
                # Run prediction
                result = wrapper.predict(test_audio)
                
                if result:
                    print(f"   Prediction: Class {result['predicted_class']}")
                    print(f"   Confidence: {result['confidence']:.3f}")
                    print(f"   Status: ✅ SUCCESS")
                else:
                    print(f"   Status: ❌ PREDICTION FAILED")
                    
            except Exception as e:
                print(f"   Status: ❌ WRAPPER FAILED - {e}")

if __name__ == "__main__":
    test_enhanced_wrapper()
'''
    
    with open("enhanced_model_wrapper.py", "w") as f:
        f.write(wrapper_code)
    
    print("✅ Created enhanced model wrapper: enhanced_model_wrapper.py")

if __name__ == "__main__":
    success = fix_model_quantization()
    if success:
        create_enhanced_inference_wrapper()
    print(f"\n🎯 Model fixing {'completed' if success else 'failed'}!")