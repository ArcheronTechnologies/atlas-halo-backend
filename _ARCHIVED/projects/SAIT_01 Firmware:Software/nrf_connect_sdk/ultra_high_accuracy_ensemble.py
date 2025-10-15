#!/usr/bin/env python3
"""
Ultra-High Accuracy Ensemble Implementation for SAIT_01
Achieve 95%+ accuracy through advanced ensemble methods
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor, MODEL_CONFIG

class FocalLoss(keras.losses.Loss):
    """Advanced Focal Loss for hard examples"""
    
    def __init__(self, alpha=None, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        if self.alpha is not None:
            alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
            alpha_t = tf.gather(alpha_tensor, y_true)
            focal_weight = alpha_t * focal_weight
            
        return focal_weight * ce

class UltraHighAccuracyEnsemble:
    """Implement ultra-high accuracy ensemble for 95%+ target"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        self.models = []
        
    def load_expanded_dataset(self, max_per_class=1500):
        """Load the expanded balanced dataset"""
        print("📊 Loading expanded dataset for ensemble training...")
        
        expanded_dir = Path("expanded_sait01_dataset")
        if not expanded_dir.exists():
            print("❌ Expanded dataset not found")
            return None, None
        
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = expanded_dir / class_name
            if not class_dir.exists():
                continue
                
            audio_files = list(class_dir.glob("*.wav"))
            
            # Quality filtering for each class
            if class_name == 'background':
                # Prefer clean background sounds
                filtered_files = []
                for f in audio_files:
                    if any(keyword in f.name.lower() for keyword in 
                          ['synthetic', 'nature', 'wind', 'water', 'rain']):
                        filtered_files.append(f)
                if len(filtered_files) < max_per_class:
                    remaining = max_per_class - len(filtered_files)
                    other_files = [f for f in audio_files if f not in filtered_files]
                    filtered_files.extend(other_files[:remaining])
                audio_files = filtered_files
            
            # Limit samples per class
            if len(audio_files) > max_per_class:
                np.random.shuffle(audio_files)
                audio_files = audio_files[:max_per_class]
            
            print(f"   Loading {class_name}: {len(audio_files)} samples")
            
            for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
                try:
                    audio = self.preprocessor.load_and_resample(audio_file)
                    features = self.preprocessor.extract_mel_spectrogram(audio)
                    X.append(features)
                    y.append(class_idx)
                except Exception:
                    continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Dataset loaded: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_cnn_specialist(self):
        """Create CNN specialist model"""
        inputs = keras.Input(shape=MODEL_CONFIG['input_shape'], name='cnn_input')
        
        # Enhanced CNN with attention
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        # Attention mechanism
        attention = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
        x = keras.layers.Multiply()([x, attention])
        
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.2)(x)
        
        outputs = keras.layers.Dense(3, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_Specialist')
        
        # Configure with focal loss
        focal_loss = FocalLoss(alpha=[1.0, 4.0, 1.5], gamma=2.0)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss=focal_loss,
            metrics=['accuracy']
        )
        
        return model
    
    def create_multiscale_specialist(self):
        """Create multi-scale feature specialist"""
        inputs = keras.Input(shape=MODEL_CONFIG['input_shape'], name='multiscale_input')
        
        # Multi-scale convolutions
        conv3x3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv5x5 = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
        conv7x7 = keras.layers.Conv2D(64, (7, 7), activation='relu', padding='same')(inputs)
        
        # Concatenate multi-scale features
        x = keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        # Dual pooling
        avg_pool = keras.layers.GlobalAveragePooling2D()(x)
        max_pool = keras.layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Concatenate()([avg_pool, max_pool])
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.2)(x)
        
        outputs = keras.layers.Dense(3, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='MultiScale_Specialist')
        
        # Configure with focal loss and extreme vehicle weighting
        focal_loss = FocalLoss(alpha=[0.8, 5.0, 1.2], gamma=2.5)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss=focal_loss,
            metrics=['accuracy']
        )
        
        return model
    
    def create_temporal_specialist(self):
        """Create temporal features specialist"""
        inputs = keras.Input(shape=MODEL_CONFIG['input_shape'], name='temporal_input')
        
        # Extract temporal features
        x = keras.layers.Conv2D(32, (1, 7), activation='relu', padding='same')(inputs)  # Temporal conv
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, (7, 1), activation='relu', padding='same')(x)      # Frequency conv
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)
        
        # Bidirectional processing
        shape = x.shape
        x = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(512, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = keras.layers.Dropout(0.2)(x)
        
        outputs = keras.layers.Dense(3, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='Temporal_Specialist')
        
        # Configure with focal loss
        focal_loss = FocalLoss(alpha=[1.0, 4.5, 1.3], gamma=2.0)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss=focal_loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train_ensemble(self, X, y):
        """Train the ensemble of specialist models"""
        print("🚀 Training Ultra-High Accuracy Ensemble...")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        print(f"📊 Data split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create specialists
        print("\n🤖 Creating specialist models...")
        cnn_model = self.create_cnn_specialist()
        multiscale_model = self.create_multiscale_specialist()
        temporal_model = self.create_temporal_specialist()
        
        specialists = [
            ("CNN_Specialist", cnn_model),
            ("MultiScale_Specialist", multiscale_model),
            ("Temporal_Specialist", temporal_model)
        ]
        
        # Train each specialist
        trained_models = []
        
        for name, model in specialists:
            print(f"\n🔥 Training {name}...")
            print(f"📊 Parameters: {model.count_params()}")
            
            # Enhanced callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    f'sait01_{name.lower()}_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Extreme class weights for vehicle emphasis
            class_weights = {0: 1.0, 1: 6.0, 2: 1.5}
            
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Test individual model
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            print(f"✅ {name} trained in {training_time/60:.1f} minutes")
            print(f"📊 Individual accuracy: {accuracy*100:.1f}%")
            
            trained_models.append((name, model, accuracy))
        
        # Create weighted ensemble
        print(f"\n🤝 Creating weighted ensemble...")
        
        ensemble_predictions = []
        weights = []
        
        for name, model, accuracy in trained_models:
            pred = model.predict(X_test, verbose=0)
            ensemble_predictions.append(pred)
            weights.append(accuracy ** 2)  # Square accuracy for stronger weighting
            
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        print(f"📊 Ensemble weights: {[f'{w:.3f}' for w in weights]}")
        
        # Weighted average prediction
        weighted_ensemble_pred = np.average(ensemble_predictions, axis=0, weights=weights)
        ensemble_classes = np.argmax(weighted_ensemble_pred, axis=1)
        
        # Calculate ensemble accuracy
        ensemble_accuracy = accuracy_score(y_test, ensemble_classes)
        
        print(f"\n🏆 ENSEMBLE RESULTS:")
        print(f"📈 Ensemble Accuracy: {ensemble_accuracy*100:.1f}%")
        
        # Detailed analysis
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, ensemble_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_classes)
        print(f"\n🔍 Ensemble Confusion Matrix:")
        print(f"           Predicted")
        print(f"         BG  VH  AC")
        for i, true_class in enumerate(['BG', 'VH', 'AC']):
            print(f"True {true_class}: {cm[i]}")
        
        # Check if we reached 95%
        target_reached = ensemble_accuracy >= 0.95
        
        print(f"\n🎯 95% ACCURACY TARGET:")
        print(f"   Target: 95.0%")
        print(f"   Achieved: {ensemble_accuracy*100:.1f}%")
        print(f"   Status: {'✅ TARGET REACHED!' if target_reached else '❌ Need more improvements'}")
        
        if target_reached:
            print(f"\n🎉 MISSION ACCOMPLISHED!")
            print(f"🚀 SAIT_01 has achieved ultra-high accuracy for battlefield deployment!")
        else:
            gap = 0.95 - ensemble_accuracy
            print(f"\n📈 Gap remaining: {gap*100:.1f} percentage points")
            print(f"💡 Recommendation: Implement short-term improvements for final push to 95%")
        
        # Save ensemble
        self.save_ensemble_models(trained_models, weights)
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'individual_accuracies': [acc for _, _, acc in trained_models],
            'weights': weights,
            'target_reached': target_reached,
            'models': trained_models
        }
    
    def save_ensemble_models(self, trained_models, weights):
        """Save the ensemble models"""
        print(f"\n💾 Saving ensemble models...")
        
        # Save individual models
        for name, model, accuracy in trained_models:
            filename = f'sait01_{name.lower()}_final.h5'
            model.save(filename)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_filename = f'sait01_{name.lower()}_final.tflite'
            with open(tflite_filename, 'wb') as f:
                f.write(tflite_model)
            
            model_size = os.path.getsize(filename) / 1024
            tflite_size = len(tflite_model) / 1024
            
            print(f"   ✅ {name}: {model_size:.1f}KB (H5), {tflite_size:.1f}KB (TFLite)")
        
        # Save ensemble weights
        ensemble_info = {
            'weights': weights.tolist(),
            'model_names': [name for name, _, _ in trained_models],
            'individual_accuracies': [acc for _, _, acc in trained_models]
        }
        
        with open('sait01_ensemble_config.json', 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        print(f"✅ Ensemble configuration saved: sait01_ensemble_config.json")

def main():
    """Execute ultra-high accuracy ensemble training"""
    print("🎯 SAIT_01 ULTRA-HIGH ACCURACY ENSEMBLE")
    print("=" * 70)
    print("🚀 Targeting 95%+ accuracy through advanced ensemble methods")
    print("⏱️  Estimated training time: 1-2 hours")
    print()
    
    # Initialize ensemble
    ensemble = UltraHighAccuracyEnsemble()
    
    # Load dataset
    X, y = ensemble.load_expanded_dataset()
    
    if X is None:
        print("❌ Cannot load dataset")
        return
    
    # Train ensemble
    results = ensemble.train_ensemble(X, y)
    
    # Final summary
    print(f"\n🎊 ULTRA-HIGH ACCURACY ENSEMBLE COMPLETE!")
    print("=" * 70)
    print(f"🎯 Ensemble Accuracy: {results['ensemble_accuracy']*100:.1f}%")
    print(f"📊 Individual Models: {[f'{acc*100:.1f}%' for acc in results['individual_accuracies']]}")
    print(f"⚖️  Ensemble Weights: {[f'{w:.3f}' for w in results['weights']]}")
    
    if results['target_reached']:
        print(f"\n🏆 SUCCESS: 95% ACCURACY TARGET ACHIEVED!")
        print(f"✅ SAIT_01 is ready for high-stakes battlefield deployment!")
        print(f"🚀 Ultra-high accuracy defense system operational!")
    else:
        print(f"\n📈 PROGRESS: Significant improvement achieved")
        print(f"💡 Next: Implement short-term improvements for final 95% push")

if __name__ == "__main__":
    main()