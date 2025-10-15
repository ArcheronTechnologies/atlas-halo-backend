#!/usr/bin/env python3
"""
Multi-Stage Classification System for Ultra-High Accuracy
Implements cascaded detection with confidence thresholding and ensemble voting
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
from pathlib import Path

class MultiStageClassifier:
    """Multi-stage classifier with progressive refinement and false positive rejection"""
    
    def __init__(self):
        self.stage1_model = None  # Fast screening model
        self.stage2_model = None  # Detailed classification model
        self.stage3_model = None  # Verification model
        
        # Confidence thresholds for each stage
        self.stage1_threshold = 0.7  # High recall, some false positives OK
        self.stage2_threshold = 0.85  # Higher precision required
        self.stage3_threshold = 0.95  # Ultimate verification
        
        # False positive rejection thresholds
        self.background_confidence_min = 0.8  # Background must be very confident
        self.target_confidence_min = 0.9    # Targets must be very confident
        
        print("🎯 Multi-Stage Classification System")
        print("Stage 1: Fast screening (70% confidence)")
        print("Stage 2: Detailed analysis (85% confidence)")
        print("Stage 3: Final verification (95% confidence)")
    
    def create_stage1_model(self, input_shape=(63, 64, 1)):
        """Fast screening model - optimized for speed and high recall"""
        print("🚀 Creating Stage 1 Model (Fast Screening)")
        
        inputs = layers.Input(shape=input_shape)
        
        # Very lightweight architecture for speed
        x = layers.SeparableConv2D(16, (5, 5), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((4, 4))(x)  # Aggressive pooling
        x = layers.Dropout(0.1)(x)
        
        x = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Binary output: background vs potential target
        outputs = layers.Dense(2, activation='softmax', name='stage1_output')(x)
        
        model = models.Model(inputs, outputs, name='Stage1_FastScreening')
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Stage 1 Model: {model.count_params()} parameters")
        return model
    
    def create_stage2_model(self, input_shape=(63, 64, 1)):
        """Detailed classification model - balanced speed/accuracy"""
        print("🎯 Creating Stage 2 Model (Detailed Classification)")
        
        inputs = layers.Input(shape=input_shape)
        
        # Multi-branch architecture for robustness
        # Branch 1: Standard CNN
        x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        
        x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        
        x1 = layers.GlobalAveragePooling2D()(x1)
        branch1 = layers.Dense(64, activation='relu')(x1)
        
        # Branch 2: Attention-enhanced CNN
        x2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
        x2 = layers.BatchNormalization()(x2)
        
        # Spatial attention
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x2)
        x2 = layers.multiply([x2, attention])
        
        x2 = layers.MaxPooling2D((2, 2))(x2)
        x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = layers.GlobalAveragePooling2D()(x2)
        branch2 = layers.Dense(64, activation='relu')(x2)
        
        # Combine branches
        combined = layers.concatenate([branch1, branch2])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        
        # Three-class output: background, drone, helicopter
        outputs = layers.Dense(3, activation='softmax', name='stage2_output')(combined)
        
        model = models.Model(inputs, outputs, name='Stage2_DetailedClassification')
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Stage 2 Model: {model.count_params()} parameters")
        return model
    
    def create_stage3_model(self, input_shape=(63, 64, 1)):
        """Final verification model - maximum accuracy"""
        print("🔍 Creating Stage 3 Model (Final Verification)")
        
        inputs = layers.Input(shape=input_shape)
        
        # Advanced architecture with multiple feature extraction paths
        # Path 1: Fine-grained temporal features
        x1 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(inputs)  # Temporal
        x1 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(x1)      # Spectral
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        
        x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D((2, 2))(x1)
        
        # Path 2: Frequency-focused features
        x2 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(inputs)  # Spectral
        x2 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(x2)      # Temporal
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling2D((2, 2))(x2)
        
        x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling2D((2, 2))(x2)
        
        # Path 3: Global context
        x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x3 = layers.BatchNormalization()(x3)
        # Skip connection
        residual = x3
        
        x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.add([x3, residual])
        
        x3 = layers.MaxPooling2D((4, 4))(x3)
        
        # Combine all paths
        x1_pooled = layers.GlobalAveragePooling2D()(x1)
        x2_pooled = layers.GlobalAveragePooling2D()(x2)
        x3_pooled = layers.GlobalAveragePooling2D()(x3)
        
        combined = layers.concatenate([x1_pooled, x2_pooled, x3_pooled])
        
        # Dense layers with extensive regularization
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.4)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output with high confidence requirement
        outputs = layers.Dense(3, activation='softmax', name='stage3_output')(combined)
        
        model = models.Model(inputs, outputs, name='Stage3_FinalVerification')
        
        # Use lower learning rate for fine-tuning
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Stage 3 Model: {model.count_params()} parameters")
        return model
    
    def build_multistage_system(self):
        """Build complete multi-stage classification system"""
        print("\n🏗️  Building Multi-Stage Classification System")
        print("=" * 60)
        
        self.stage1_model = self.create_stage1_model()
        self.stage2_model = self.create_stage2_model()
        self.stage3_model = self.create_stage3_model()
        
        total_params = (self.stage1_model.count_params() + 
                       self.stage2_model.count_params() + 
                       self.stage3_model.count_params())
        
        print(f"\n📊 System Summary:")
        print(f"  Total parameters: {total_params}")
        print(f"  Stage 1: {self.stage1_model.count_params()} params (fast screening)")
        print(f"  Stage 2: {self.stage2_model.count_params()} params (classification)")
        print(f"  Stage 3: {self.stage3_model.count_params()} params (verification)")
        
        return True
    
    def prepare_stage_training_data(self, X_data, y_data):
        """Prepare training data for each stage"""
        
        # Stage 1: Binary classification (background vs target)
        y_stage1 = (y_data > 0).astype(np.int32)  # 0=background, 1=any target
        
        # Stage 2 & 3: Three-class classification
        y_stage2 = y_data.copy()
        y_stage3 = y_data.copy()
        
        print(f"📊 Stage Training Data:")
        print(f"  Stage 1 (binary): {np.bincount(y_stage1)}")
        print(f"  Stage 2 (3-class): {np.bincount(y_stage2)}")
        print(f"  Stage 3 (3-class): {np.bincount(y_stage3)}")
        
        return y_stage1, y_stage2, y_stage3
    
    def train_multistage_system(self, X_train, y_train, X_val, y_val, epochs=30):
        """Train all stages of the multi-stage system"""
        print("\n🚀 Training Multi-Stage System")
        print("=" * 50)
        
        # Prepare stage-specific labels
        y_train_s1, y_train_s2, y_train_s3 = self.prepare_stage_training_data(X_train, y_train)
        y_val_s1, y_val_s2, y_val_s3 = self.prepare_stage_training_data(X_val, y_val)
        
        histories = {}
        
        # Train Stage 1 (Fast Screening)
        print("\n🎯 Training Stage 1: Fast Screening")
        print("-" * 40)
        
        callbacks_s1 = [
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)
        ]
        
        history_s1 = self.stage1_model.fit(
            X_train, y_train_s1,
            validation_data=(X_val, y_val_s1),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks_s1,
            verbose=1
        )
        histories['stage1'] = history_s1
        
        # Train Stage 2 (Detailed Classification)
        print("\n🎯 Training Stage 2: Detailed Classification")
        print("-" * 40)
        
        callbacks_s2 = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=5)
        ]
        
        history_s2 = self.stage2_model.fit(
            X_train, y_train_s2,
            validation_data=(X_val, y_val_s2),
            epochs=epochs,
            batch_size=24,
            callbacks=callbacks_s2,
            verbose=1
        )
        histories['stage2'] = history_s2
        
        # Train Stage 3 (Final Verification) with extra attention to hard examples
        print("\n🎯 Training Stage 3: Final Verification")
        print("-" * 40)
        
        callbacks_s3 = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, min_delta=0.001),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=6, min_lr=1e-6)
        ]
        
        history_s3 = self.stage3_model.fit(
            X_train, y_train_s3,
            validation_data=(X_val, y_val_s3),
            epochs=epochs + 10,  # More epochs for final stage
            batch_size=16,  # Smaller batch for better gradient estimates
            callbacks=callbacks_s3,
            verbose=1
        )
        histories['stage3'] = history_s3
        
        print("\n✅ Multi-Stage Training Complete")
        return histories
    
    def predict_multistage(self, X_sample, return_intermediate=False):
        """Run multi-stage prediction with progressive refinement"""
        
        if len(X_sample.shape) == 3:
            X_sample = np.expand_dims(X_sample, axis=0)
        
        results = {}
        
        # Stage 1: Fast screening
        stage1_start = time.time()
        stage1_pred = self.stage1_model.predict(X_sample, verbose=0)
        stage1_time = time.time() - stage1_start
        
        stage1_prob = stage1_pred[0, 1]  # Probability of being a target
        results['stage1'] = {'probability': stage1_prob, 'time': stage1_time}
        
        # Check Stage 1 threshold
        if stage1_prob < self.stage1_threshold:
            # Classified as background - stop here
            results['final_class'] = 0  # Background
            results['final_confidence'] = 1.0 - stage1_prob
            results['stages_used'] = 1
            results['total_time'] = stage1_time
            
            if return_intermediate:
                return results
            else:
                return results['final_class'], results['final_confidence']
        
        # Stage 2: Detailed classification
        stage2_start = time.time()
        stage2_pred = self.stage2_model.predict(X_sample, verbose=0)
        stage2_time = time.time() - stage2_start
        
        stage2_class = np.argmax(stage2_pred[0])
        stage2_confidence = np.max(stage2_pred[0])
        results['stage2'] = {
            'class': stage2_class, 
            'confidence': stage2_confidence, 
            'probabilities': stage2_pred[0],
            'time': stage2_time
        }\n        \n        # Check Stage 2 threshold\n        if stage2_confidence < self.stage2_threshold or stage2_class == 0:\n            # Either low confidence or classified as background\n            results['final_class'] = stage2_class\n            results['final_confidence'] = stage2_confidence\n            results['stages_used'] = 2\n            results['total_time'] = stage1_time + stage2_time\n            \n            if return_intermediate:\n                return results\n            else:\n                return results['final_class'], results['final_confidence']\n        \n        # Stage 3: Final verification\n        stage3_start = time.time()\n        stage3_pred = self.stage3_model.predict(X_sample, verbose=0)\n        stage3_time = time.time() - stage3_start\n        \n        stage3_class = np.argmax(stage3_pred[0])\n        stage3_confidence = np.max(stage3_pred[0])\n        results['stage3'] = {\n            'class': stage3_class, \n            'confidence': stage3_confidence, \n            'probabilities': stage3_pred[0],\n            'time': stage3_time\n        }\n        \n        # Final decision with high confidence requirement\n        if stage3_confidence >= self.stage3_threshold and stage3_class > 0:\n            # High confidence target detection\n            results['final_class'] = stage3_class\n            results['final_confidence'] = stage3_confidence\n        else:\n            # Uncertain - default to background for safety\n            results['final_class'] = 0\n            results['final_confidence'] = 1.0 - stage3_confidence\n        \n        results['stages_used'] = 3\n        results['total_time'] = stage1_time + stage2_time + stage3_time\n        \n        if return_intermediate:\n            return results\n        else:\n            return results['final_class'], results['final_confidence']\n    \n    def evaluate_multistage_system(self, X_test, y_test):\n        \"\"\"Comprehensive evaluation of multi-stage system\"\"\"\n        print(\"\\n📊 Multi-Stage System Evaluation\")\n        print(\"=\" * 50)\n        \n        predictions = []\n        confidences = []\n        stage_usage = {1: 0, 2: 0, 3: 0}\n        total_time = 0\n        \n        print(f\"Evaluating {len(X_test)} samples...\")\n        \n        for i, sample in enumerate(X_test):\n            result = self.predict_multistage(sample, return_intermediate=True)\n            \n            predictions.append(result['final_class'])\n            confidences.append(result['final_confidence'])\n            stage_usage[result['stages_used']] += 1\n            total_time += result['total_time']\n            \n            if (i + 1) % 50 == 0:\n                print(f\"  Processed {i + 1}/{len(X_test)} samples\")\n        \n        predictions = np.array(predictions)\n        confidences = np.array(confidences)\n        \n        # Calculate metrics\n        accuracy = np.mean(predictions == y_test)\n        \n        # False positive analysis (critical for defense application)\n        background_mask = (y_test == 0)\n        false_positives = np.sum((predictions > 0) & background_mask)\n        total_background = np.sum(background_mask)\n        fp_rate = false_positives / total_background if total_background > 0 else 0\n        \n        # True positive analysis\n        target_mask = (y_test > 0)\n        true_positives = np.sum((predictions > 0) & target_mask)\n        total_targets = np.sum(target_mask)\n        tp_rate = true_positives / total_targets if total_targets > 0 else 0\n        \n        avg_time_ms = (total_time / len(X_test)) * 1000\n        \n        print(f\"\\n🎯 Multi-Stage Results:\")\n        print(f\"  Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\")\n        print(f\"  False Positive Rate: {fp_rate:.3f} ({fp_rate*100:.1f}%)\")\n        print(f\"  True Positive Rate: {tp_rate:.3f} ({tp_rate*100:.1f}%)\")\n        print(f\"  Average Time: {avg_time_ms:.2f} ms\")\n        \n        print(f\"\\n⚡ Stage Usage Distribution:\")\n        total_samples = len(X_test)\n        for stage, count in stage_usage.items():\n            percentage = (count / total_samples) * 100\n            print(f\"  Stage {stage}: {count} samples ({percentage:.1f}%)\")\n        \n        # Performance targets\n        target_accuracy = 0.925  # 92.5%\n        target_fp_rate = 0.05   # 5%\n        \n        accuracy_met = accuracy >= target_accuracy\n        fp_met = fp_rate <= target_fp_rate\n        \n        print(f\"\\n🎯 Target Achievement:\")\n        print(f\"  Accuracy ≥92.5%: {'✅ MET' if accuracy_met else '❌ NOT MET'}\")\n        print(f\"  False Positive ≤5%: {'✅ MET' if fp_met else '❌ NOT MET'}\")\n        print(f\"  Overall: {'✅ SUCCESS' if (accuracy_met and fp_met) else '❌ NEEDS IMPROVEMENT'}\")\n        \n        return {\n            'accuracy': accuracy,\n            'fp_rate': fp_rate,\n            'tp_rate': tp_rate,\n            'avg_time_ms': avg_time_ms,\n            'stage_usage': stage_usage,\n            'targets_met': accuracy_met and fp_met,\n            'predictions': predictions,\n            'confidences': confidences\n        }\n    \n    def save_multistage_models(self, base_path=\"sait01_multistage\"):\n        \"\"\"Save all trained models\"\"\"\n        print(f\"\\n💾 Saving Multi-Stage Models to {base_path}_*\")\n        \n        self.stage1_model.save(f\"{base_path}_stage1.h5\")\n        self.stage2_model.save(f\"{base_path}_stage2.h5\")\n        self.stage3_model.save(f\"{base_path}_stage3.h5\")\n        \n        # Save configuration\n        config = {\n            'stage1_threshold': self.stage1_threshold,\n            'stage2_threshold': self.stage2_threshold,\n            'stage3_threshold': self.stage3_threshold,\n            'background_confidence_min': self.background_confidence_min,\n            'target_confidence_min': self.target_confidence_min\n        }\n        \n        import json\n        with open(f\"{base_path}_config.json\", 'w') as f:\n            json.dump(config, f, indent=2)\n        \n        print(\"✅ All models and configuration saved\")\n        \n        return True\n\ndef main():\n    # Example usage\n    classifier = MultiStageClassifier()\n    classifier.build_multistage_system()\n    print(\"\\n🚀 Multi-Stage Classification System Ready!\")\n    print(\"Ready for training with enhanced dataset...\")\n\nif __name__ == \"__main__\":\n    main()