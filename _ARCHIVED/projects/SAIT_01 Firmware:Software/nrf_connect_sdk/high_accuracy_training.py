#!/usr/bin/env python3
"""
High Accuracy Training System - 90-95% Target
Robust implementation without problematic imports
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

# Add current directory to path
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class HighAccuracyTrainer:
    """High accuracy trainer targeting 90-95% accuracy with robust false positive rejection"""
    
    def __init__(self, data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        self.data_dir = data_dir
        self.preprocessor = SaitAudioPreprocessor()
        
        # Performance targets
        self.target_accuracy = 0.92      # 92% accuracy target
        self.max_false_positive_rate = 0.05  # 5% max false positives
        self.confidence_threshold = 0.85     # 85% confidence required
        
        print("🚀 HIGH ACCURACY TRAINING SYSTEM")
        print("=" * 50)
        print(f"🎯 Target Accuracy: {self.target_accuracy*100:.1f}%")
        print(f"🛡️  Max False Positives: {self.max_false_positive_rate*100:.1f}%")
        print("=" * 50)
        
    def load_enhanced_dataset(self, samples_per_class=300):
        """Load the enhanced dataset with all negative samples"""
        print(f"📊 Loading enhanced dataset ({samples_per_class} per class)...")
        
        X_data = []
        y_data = []
        file_sources = []
        
        train_path = Path(self.data_dir) / 'train'
        class_mapping = {'background': 0, 'drone': 1, 'helicopter': 2}
        
        for class_name, class_id in class_mapping.items():
            class_path = train_path / class_name
            if not class_path.exists():
                print(f"❌ Missing: {class_path}")
                continue
                
            files = list(class_path.glob("*.wav"))
            
            # For background, use all available samples (we have many negatives)
            if class_name == 'background':
                selected_files = files  # Use all background samples
            else:
                # For targets, limit to samples_per_class
                selected_files = files[:samples_per_class] if len(files) >= samples_per_class else files
            
            print(f"📂 Processing {len(selected_files)} {class_name} samples...")
            
            for i, audio_file in enumerate(selected_files):
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
                    file_sources.append(str(audio_file))
                    
                    if (i + 1) % 100 == 0:
                        print(f"  {i + 1}/{len(selected_files)} processed")
                        
                except Exception as e:
                    # print(f"  ❌ {audio_file.name}: {e}")
                    continue
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"✅ Loaded {len(X_data)} high-quality samples")
        print(f"📊 Shape: {X_data.shape}")
        print(f"📋 Class distribution: {np.bincount(y_data)}")
        
        return X_data, y_data, file_sources
    
    def create_advanced_model(self):
        """Create advanced model architecture for high accuracy"""
        print("\\n🧠 Building Advanced High-Accuracy Model")
        
        inputs = tf.keras.layers.Input(shape=(63, 64, 1), name='audio_input')
        
        # Branch 1: Fine-grained feature extraction
        x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
        x1 = tf.keras.layers.Dropout(0.2)(x1)
        
        x1 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
        x1 = tf.keras.layers.Dropout(0.25)(x1)
        
        x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
        branch1 = tf.keras.layers.Dense(128, activation='relu')(x1)
        
        # Branch 2: Attention-enhanced processing
        x2 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        
        # Simple attention mechanism
        attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x2)
        x2 = tf.keras.layers.multiply([x2, attention])
        
        x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
        x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
        x2 = tf.keras.layers.Dropout(0.25)(x2)
        
        x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
        branch2 = tf.keras.layers.Dense(128, activation='relu')(x2)
        
        # Branch 3: Multi-scale feature extraction
        x3_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        x3_2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
        x3_3 = tf.keras.layers.Conv2D(16, (7, 7), activation='relu', padding='same')(inputs)
        
        x3 = tf.keras.layers.concatenate([x3_1, x3_2, x3_3])
        x3 = tf.keras.layers.BatchNormalization()(x3)
        x3 = tf.keras.layers.MaxPooling2D((4, 4))(x3)
        x3 = tf.keras.layers.GlobalAveragePooling2D()(x3)
        branch3 = tf.keras.layers.Dense(128, activation='relu')(x3)
        
        # Combine branches
        combined = tf.keras.layers.concatenate([branch1, branch2, branch3])
        combined = tf.keras.layers.Dense(256, activation='relu')(combined)
        combined = tf.keras.layers.Dropout(0.5)(combined)
        combined = tf.keras.layers.BatchNormalization()(combined)
        
        combined = tf.keras.layers.Dense(128, activation='relu')(combined)
        combined = tf.keras.layers.Dropout(0.4)(combined)
        
        combined = tf.keras.layers.Dense(64, activation='relu')(combined)
        combined = tf.keras.layers.Dropout(0.3)(combined)
        
        # Multi-output for enhanced training
        main_output = tf.keras.layers.Dense(3, activation='softmax', name='classification')(combined)
        confidence_output = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence')(combined)
        
        model = tf.keras.Model(
            inputs=inputs, 
            outputs=[main_output, confidence_output],
            name='HighAccuracy_Model'
        )
        
        # Advanced optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={
                'classification': 1.0,
                'confidence': 0.4
            },
            metrics={
                'classification': ['accuracy'],
                'confidence': ['mae']
            }
        )
        
        print("📝 Model Summary:")
        model.summary()
        
        params = model.count_params()
        size_kb = params * 4 / 1024
        print(f"📏 Model size: {size_kb:.1f} KB ({params} parameters)")
        
        return model
    
    def create_advanced_augmentation(self):
        """Create comprehensive augmentation pipeline"""
        
        def augment_sample(mel_spec):
            """Apply advanced augmentation to a single sample"""
            
            # Frequency masking
            if random.random() < 0.5:
                freq_mask_width = random.randint(2, 8)
                freq_start = random.randint(0, 64 - freq_mask_width)
                mel_spec[:, freq_start:freq_start + freq_mask_width, :] *= random.uniform(0.0, 0.2)
            
            # Time masking
            if random.random() < 0.5:
                time_mask_width = random.randint(2, 10)
                time_start = random.randint(0, 63 - time_mask_width)
                mel_spec[time_start:time_start + time_mask_width, :, :] *= random.uniform(0.0, 0.2)
            
            # Noise injection
            if random.random() < 0.6:
                noise_level = random.uniform(0.01, 0.05)
                noise = np.random.randn(*mel_spec.shape) * noise_level
                mel_spec = mel_spec + noise
            
            # Frequency shifting
            if random.random() < 0.3:
                shift = random.randint(-2, 2)
                if shift != 0:
                    mel_spec = np.roll(mel_spec, shift, axis=1)
            
            # Dynamic range compression
            if random.random() < 0.3:
                compression = random.uniform(0.8, 1.2)
                mel_spec = np.sign(mel_spec) * np.power(np.abs(mel_spec), compression)
            
            return mel_spec
        
        return augment_sample
    
    def apply_comprehensive_augmentation(self, X, y, multiplier=4):
        """Apply comprehensive augmentation to dataset"""
        print(f"🔄 Applying comprehensive augmentation (x{multiplier})...")
        
        augment_fn = self.create_advanced_augmentation()
        
        augmented_X = []
        augmented_y = []
        
        # Original samples
        for i in range(len(X)):
            augmented_X.append(X[i])
            augmented_y.append(y[i])
        
        # Generate augmented versions
        for i in range(len(X)):
            for _ in range(multiplier):
                aug_sample = augment_fn(X[i].copy())
                augmented_X.append(aug_sample)
                augmented_y.append(y[i])
        
        # Additional negative augmentation for background samples
        background_indices = np.where(y == 0)[0]
        for idx in background_indices:
            if random.random() < 0.5:  # 50% chance
                # Mix with another background sample
                other_idx = random.choice(background_indices)
                if other_idx != idx:
                    mix_ratio = random.uniform(0.3, 0.7)
                    mixed_sample = mix_ratio * X[idx] + (1 - mix_ratio) * X[other_idx]
                    augmented_X.append(mixed_sample)
                    augmented_y.append(0)  # Still background
        
        augmented_X = np.array(augmented_X)
        augmented_y = np.array(augmented_y)
        
        print(f"✅ Augmented dataset: {len(augmented_X)} samples (from {len(X)})")
        print(f"📋 New class distribution: {np.bincount(augmented_y)}")\n        \n        return augmented_X, augmented_y\n    \n    def create_advanced_callbacks(self):\n        \"\"\"Create advanced training callbacks\"\"\"\n        \n        return [\n            tf.keras.callbacks.EarlyStopping(\n                monitor='val_classification_accuracy',\n                patience=20,\n                restore_best_weights=True,\n                verbose=1,\n                min_delta=0.001\n            ),\n            \n            tf.keras.callbacks.ReduceLROnPlateau(\n                monitor='val_loss',\n                factor=0.3,\n                patience=7,\n                min_lr=1e-6,\n                verbose=1\n            ),\n            \n            tf.keras.callbacks.ModelCheckpoint(\n                'high_accuracy_best_model.h5',\n                monitor='val_classification_accuracy',\n                save_best_only=True,\n                save_weights_only=False,\n                verbose=1\n            )\n        ]\n    \n    def train_high_accuracy_model(self, X_data, y_data):\n        \"\"\"Train high accuracy model with advanced techniques\"\"\"\n        print(\"\\n🚀 High Accuracy Model Training\")\n        print(\"=\" * 50)\n        \n        # Stratified split to maintain class proportions\n        X_train, X_test, y_train, y_test = train_test_split(\n            X_data, y_data, test_size=0.2, stratify=y_data, random_state=42\n        )\n        \n        print(f\"📊 Training: {len(X_train)}, Testing: {len(X_test)}\")\n        print(f\"📋 Train distribution: {np.bincount(y_train)}\")\n        print(f\"📋 Test distribution: {np.bincount(y_test)}\")\n        \n        # Apply comprehensive augmentation\n        X_train_aug, y_train_aug = self.apply_comprehensive_augmentation(X_train, y_train)\n        \n        # Create confidence labels\n        y_train_conf = np.ones(len(y_train_aug), dtype=np.float32)\n        y_test_conf = np.ones(len(y_test), dtype=np.float32)\n        \n        # Reduce confidence for heavily augmented samples\n        original_count = len(y_train)\n        for i in range(original_count, len(y_train_aug)):\n            y_train_conf[i] = random.uniform(0.7, 0.9)\n        \n        # Create model\n        model = self.create_advanced_model()\n        \n        # Advanced callbacks\n        callbacks = self.create_advanced_callbacks()\n        \n        # Training\n        print(\"🔄 Starting high accuracy training...\")\n        start_time = time.time()\n        \n        history = model.fit(\n            X_train_aug,\n            {\n                'classification': y_train_aug,\n                'confidence': y_train_conf\n            },\n            validation_data=(\n                X_test,\n                {\n                    'classification': y_test,\n                    'confidence': y_test_conf\n                }\n            ),\n            epochs=80,\n            batch_size=32,\n            callbacks=callbacks,\n            verbose=1\n        )\n        \n        train_time = time.time() - start_time\n        print(f\"⏱️  Training completed in {train_time:.1f}s\")\n        \n        # Comprehensive evaluation\n        return self.comprehensive_evaluation(model, X_test, y_test, history)\n    \n    def comprehensive_evaluation(self, model, X_test, y_test, history):\n        \"\"\"Comprehensive model evaluation\"\"\"\n        print(\"\\n📊 Comprehensive Model Evaluation\")\n        print(\"=\" * 50)\n        \n        # Predict with confidence\n        predictions = model.predict(X_test, verbose=0)\n        y_pred_class = predictions[0]\n        y_pred_conf = predictions[1].flatten()\n        \n        y_pred_labels = np.argmax(y_pred_class, axis=1)\n        y_pred_probs = np.max(y_pred_class, axis=1)\n        \n        # Overall accuracy\n        overall_accuracy = np.mean(y_pred_labels == y_test)\n        \n        # High confidence accuracy\n        high_conf_mask = y_pred_conf >= self.confidence_threshold\n        if np.sum(high_conf_mask) > 0:\n            high_conf_accuracy = np.mean(y_pred_labels[high_conf_mask] == y_test[high_conf_mask])\n        else:\n            high_conf_accuracy = 0.0\n        \n        # False positive analysis (CRITICAL for defense application)\n        background_mask = (y_test == 0)\n        false_positives = np.sum((y_pred_labels > 0) & background_mask)\n        total_background = np.sum(background_mask)\n        fp_rate = false_positives / total_background if total_background > 0 else 0\n        \n        # True positive analysis\n        target_mask = (y_test > 0)\n        true_positives = np.sum((y_pred_labels > 0) & target_mask)\n        total_targets = np.sum(target_mask)\n        tp_rate = true_positives / total_targets if total_targets > 0 else 0\n        \n        print(f\"🎯 Performance Results:\")\n        print(f\"  Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\")\n        print(f\"  High Confidence Accuracy: {high_conf_accuracy:.4f} ({high_conf_accuracy*100:.2f}%)\")\n        print(f\"  False Positive Rate: {fp_rate:.4f} ({fp_rate*100:.2f}%)\")\n        print(f\"  True Positive Rate: {tp_rate:.4f} ({tp_rate*100:.2f}%)\")\n        print(f\"  Confident Predictions: {np.sum(high_conf_mask)}/{len(y_test)} ({np.sum(high_conf_mask)/len(y_test)*100:.1f}%)\")\n        \n        # Detailed classification report\n        class_names = ['background', 'drone', 'helicopter']\n        print(\"\\n📋 Classification Report:\")\n        print(classification_report(y_test, y_pred_labels, target_names=class_names))\n        \n        # Confusion matrix\n        cm = confusion_matrix(y_test, y_pred_labels)\n        print(\"\\n📊 Confusion Matrix:\")\n        print(cm)\n        \n        # Target achievement assessment\n        accuracy_target_met = overall_accuracy >= self.target_accuracy\n        fp_target_met = fp_rate <= self.max_false_positive_rate\n        \n        print(f\"\\n🎯 Target Achievement:\")\n        print(f\"  Accuracy target (≥{self.target_accuracy*100:.1f}%): {'✅ MET' if accuracy_target_met else '❌ NOT MET'}\")\n        print(f\"  FP target (≤{self.max_false_positive_rate*100:.1f}%): {'✅ MET' if fp_target_met else '❌ NOT MET'}\")\n        \n        overall_success = accuracy_target_met and fp_target_met\n        \n        if overall_success:\n            print(\"\\n🎉 HIGH ACCURACY TARGET ACHIEVED!\")\n            print(\"🚀 Model ready for defense deployment!\")\n        else:\n            print(\"\\n🔧 Targets not fully met - analyzing issues:\")\n            if not accuracy_target_met:\n                print(f\"  - Accuracy {overall_accuracy:.1%} < {self.target_accuracy:.1%}\")\n                print(\"  - Consider: more data, better architecture, longer training\")\n            if not fp_target_met:\n                print(f\"  - False positives {fp_rate:.1%} > {self.max_false_positive_rate:.1%}\")\n                print(\"  - Consider: more negative samples, higher confidence threshold\")\n        \n        return {\n            'model': model,\n            'history': history,\n            'overall_accuracy': overall_accuracy,\n            'high_conf_accuracy': high_conf_accuracy,\n            'fp_rate': fp_rate,\n            'tp_rate': tp_rate,\n            'targets_met': overall_success,\n            'predictions': y_pred_labels,\n            'confidences': y_pred_conf,\n            'test_data': (X_test, y_test)\n        }\n    \n    def convert_to_optimized_tflite(self, model, X_sample):\n        \"\"\"Convert to optimized TensorFlow Lite\"\"\"\n        print(\"\\n🔄 Converting to Optimized TensorFlow Lite\")\n        print(\"=\" * 50)\n        \n        # Representative dataset for quantization\n        def representative_dataset():\n            for i in range(min(300, len(X_sample))):\n                yield [X_sample[i:i+1].astype(np.float32)]\n        \n        try:\n            # INT8 quantization for maximum efficiency\n            converter = tf.lite.TFLiteConverter.from_keras_model(model)\n            converter.optimizations = [tf.lite.Optimize.DEFAULT]\n            converter.representative_dataset = representative_dataset\n            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]\n            converter.inference_input_type = tf.int8\n            converter.inference_output_type = tf.int8\n            \n            quantized_model = converter.convert()\n            quantized_size = len(quantized_model) / 1024\n            \n            print(f\"✅ INT8 Quantized model: {quantized_size:.1f} KB\")\n            \n            # Test inference speed\n            interpreter = tf.lite.Interpreter(model_content=quantized_model)\n            interpreter.allocate_tensors()\n            \n            input_details = interpreter.get_input_details()\n            output_details = interpreter.get_output_details()\n            \n            # Test with sample\n            test_sample = X_sample[0:1].astype(np.float32)\n            \n            # Convert to int8 if needed\n            if input_details[0]['dtype'] == np.int8:\n                input_scale, input_zero_point = input_details[0]['quantization']\n                test_sample = (test_sample / input_scale + input_zero_point).astype(np.int8)\n            \n            # Time inference\n            times = []\n            for _ in range(100):\n                start = time.time()\n                interpreter.set_tensor(input_details[0]['index'], test_sample)\n                interpreter.invoke()\n                _ = interpreter.get_tensor(output_details[0]['index'])\n                times.append((time.time() - start) * 1000)\n            \n            avg_time = np.mean(times)\n            print(f\"⚡ Average inference: {avg_time:.2f} ms\")\n            \n            # nRF5340 compatibility check\n            if quantized_size <= 80:  # 80KB target\n                print(\"✅ Excellent fit for nRF5340 deployment\")\n            elif quantized_size <= 120:\n                print(\"✅ Good fit for nRF5340 deployment\")\n            else:\n                print(\"⚠️  May be tight fit for nRF5340\")\n            \n            # Save model\n            with open('sait01_high_accuracy_model.tflite', 'wb') as f:\n                f.write(quantized_model)\n            \n            print(\"💾 High accuracy model saved as 'sait01_high_accuracy_model.tflite'\")\n            \n            return quantized_model, avg_time\n            \n        except Exception as e:\n            print(f\"❌ INT8 quantization failed: {e}\")\n            print(\"🔄 Falling back to dynamic quantization...\")\n            \n            # Fallback to dynamic quantization\n            converter = tf.lite.TFLiteConverter.from_keras_model(model)\n            converter.optimizations = [tf.lite.Optimize.DEFAULT]\n            \n            fallback_model = converter.convert()\n            fallback_size = len(fallback_model) / 1024\n            \n            print(f\"✅ Dynamic quantized model: {fallback_size:.1f} KB\")\n            \n            with open('sait01_high_accuracy_model.tflite', 'wb') as f:\n                f.write(fallback_model)\n            \n            return fallback_model, 0.0\n    \n    def run_high_accuracy_training(self):\n        \"\"\"Complete high accuracy training pipeline\"\"\"\n        print(\"\\n🚀 HIGH ACCURACY TRAINING PIPELINE\")\n        print(\"=\" * 60)\n        print(f\"🎯 Target: {self.target_accuracy*100:.1f}% accuracy, <{self.max_false_positive_rate*100:.1f}% false positives\")\n        print(\"🧠 Strategy: Advanced architecture + Comprehensive augmentation\")\n        print(\"=\" * 60)\n        \n        # Load enhanced dataset\n        X_data, y_data, sources = self.load_enhanced_dataset()\n        \n        if len(X_data) == 0:\n            print(\"❌ No data loaded - training failed\")\n            return None\n        \n        # Train model\n        results = self.train_high_accuracy_model(X_data, y_data)\n        \n        # Convert to TensorFlow Lite\n        if results['targets_met']:\n            tflite_model, inference_time = self.convert_to_optimized_tflite(\n                results['model'], results['test_data'][0]\n            )\n            results['tflite_model'] = tflite_model\n            results['inference_time_ms'] = inference_time\n        \n        # Final summary\n        print(\"\\n🏆 HIGH ACCURACY TRAINING RESULTS\")\n        print(\"=\" * 50)\n        print(f\"✅ Dataset: {len(X_data)} samples\")\n        print(f\"🎯 Accuracy: {results['overall_accuracy']:.3f} ({results['overall_accuracy']*100:.1f}%)\")\n        print(f\"🛡️  False Positives: {results['fp_rate']:.3f} ({results['fp_rate']*100:.1f}%)\")\n        print(f\"🚀 Targets Met: {'✅ YES' if results['targets_met'] else '❌ NO'}\")\n        \n        if results['targets_met']:\n            print(\"\\n🎉 HIGH ACCURACY TRAINING SUCCESSFUL!\")\n            print(\"🚀 Model ready for critical defense deployment!\")\n        else:\n            print(\"\\n🔧 Training completed but targets not fully met\")\n            print(\"💡 Consider running ultra-high accuracy training for better results\")\n        \n        return results\n\ndef main():\n    trainer = HighAccuracyTrainer()\n    results = trainer.run_high_accuracy_training()\n    \n    if results and results['targets_met']:\n        print(\"\\n🚀 SUCCESS: High accuracy model ready for deployment!\")\n    else:\n        print(\"\\n🔧 Training needs refinement for optimal results\")\n\nif __name__ == \"__main__\":\n    main()