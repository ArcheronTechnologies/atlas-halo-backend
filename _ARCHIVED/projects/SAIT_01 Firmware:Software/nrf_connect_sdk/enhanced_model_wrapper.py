#!/usr/bin/env python3
"""
🛡️ Enhanced TensorFlow Lite Inference Wrapper
==============================================
Handles data type conversions and preprocessing for SAIT_01 models
"""

import numpy as np
import tensorflow as tf
import librosa
import os

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
            features = self.preprocess_audio(audio_data)
            if features is None:
                return None

            input_details = self.input_details[0]
            if input_details['dtype'] == np.int8 and self.input_scale is not None:
                features = features / self.input_scale + self.input_zero_point
                features = np.clip(features, -128, 127).astype(np.int8)

            self.interpreter.set_tensor(input_details['index'], features)
            self.interpreter.invoke()

            outputs = []
            for details in self.output_details:
                raw = self.interpreter.get_tensor(details['index'])
                outputs.append(self._dequantize_output(raw, details))

            if len(outputs) == 1:
                probs = tf.nn.softmax(outputs[0][0]).numpy()
                idx = int(np.argmax(probs))
                return {
                    'predicted_class': idx,
                    'confidence': float(probs[idx]),
                    'probabilities': probs.tolist(),
                    'source': 'enhanced_wrapper'
                }

            binary, category, specific = outputs[:3]
            escalation = outputs[3] if len(outputs) > 3 else None
            uncertainty = outputs[4] if len(outputs) > 4 else None

            binary_probs = tf.nn.softmax(binary[0]).numpy()
            category_probs = tf.nn.softmax(category[0]).numpy()
            specific_probs = tf.nn.softmax(specific[0]).numpy()

            from threat_taxonomy.threat_hierarchy import ThreatCategory as _TC

            if binary_probs[1] < 0.4:
                category_probs = np.zeros_like(category_probs)
                category_probs[_TC.NON_THREAT.value] = 1.0

            result = {
                'binary': binary_probs.tolist(),
                'category': category_probs.tolist(),
                'specific': specific_probs.tolist(),
                'threat_probability': float(binary_probs[1]),
                'category_index': int(np.argmax(category_probs)),
                'source': 'hierarchical_wrapper'
            }
            if escalation is not None:
                result['escalation'] = tf.nn.softmax(escalation[0]).numpy().tolist()
            if uncertainty is not None:
                result['uncertainty'] = float(uncertainty.squeeze())
            return result

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
            print(f"\n📊 Testing {model_file}:")
            
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
