#!/usr/bin/env python3
"""
Advanced Spectrographic Model Training Script
Trains the state-of-the-art 2024 spectral analysis model
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add current directory for imports
sys.path.append('.')
from advanced_spectrographic_model import AdvancedSpectrogramicModel

def train_advanced_model(dataset_dir="massive_enhanced_dataset", 
                        samples_per_class=800,
                        batch_size=8,
                        epochs=50,
                        learning_rate=0.0002,
                        validation_split=0.2,
                        save_best_only=True,
                        output_dir="models"):
    """Train the advanced spectrographic model"""
    
    print("🚀 TRAINING ADVANCED SPECTROGRAPHIC MODEL 2024")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model builder
    model_builder = AdvancedSpectrogramicModel()
    
    # Load dataset
    print(f"📊 Loading dataset from {dataset_dir}...")
    dataset_path = Path(dataset_dir)
    audio_files = []
    labels = []
    
    # Collect audio files
    for class_idx, class_name in enumerate(model_builder.class_names):
        class_dir = dataset_path / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            np.random.shuffle(files)
            
            selected_files = files[:samples_per_class]
            print(f"   {class_name}: {len(selected_files)} samples")
            
            for audio_file in selected_files:
                audio_files.append(audio_file)
                labels.append(class_idx)
        else:
            print(f"⚠️  Warning: {class_dir} not found")
    
    print(f"Total samples: {len(audio_files)}")
    
    if len(audio_files) == 0:
        print("❌ No audio files found! Check dataset path.")
        return None
    
    # Extract advanced features
    print("\n🔬 Extracting state-of-the-art features...")
    X_features, y = model_builder.prepare_training_data(audio_files, labels, 
                                                        max_samples=len(audio_files))
    
    # Verify we have all required features
    required_features = [
        'gammatone_filterbank', 'wavelet_scaleogram', 'lfcc', 'mfcc', 
        'mfcc_delta', 'mfcc_delta2', 'cqcc', 'baseline_mel',
        'mel_resolution_0', 'mel_resolution_1', 'mel_resolution_2',
        'log_power_spectrogram'
    ]
    
    missing_features = [f for f in required_features if f not in X_features]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return None
    
    # Create advanced model
    print("\n🏗️ Building advanced neural architecture...")
    model = model_builder.create_advanced_neural_architecture()
    
    print(f"✅ Model created successfully!")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Input features: {len(X_features)}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    # Train/validation split
    print(f"\n📊 Splitting data (validation: {validation_split*100:.0f}%)...")
    train_indices, val_indices = train_test_split(
        range(len(y)), 
        test_size=validation_split, 
        random_state=42, 
        stratify=y
    )
    
    X_train = {key: features[train_indices] for key, features in X_features.items()}
    X_val = {key: features[val_indices] for key, features in X_features.items()}
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    print(f"   Training samples: {len(y_train)}")
    print(f"   Validation samples: {len(y_val)}")
    
    # Class distribution
    print("\n📈 Class distribution:")
    for i, class_name in enumerate(model_builder.class_names):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        print(f"   {class_name}: {train_count} train, {val_count} val")
    
    # Setup callbacks
    callbacks = []
    
    # Learning rate reduction
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
        cooldown=3
    ))
    
    # Early stopping
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ))
    
    # Model checkpointing
    checkpoint_path = os.path.join(output_dir, 'advanced_spectrographic_best.h5')
    if save_best_only:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ))
    
    # CSV logging
    csv_path = os.path.join(output_dir, 'advanced_training_log.csv')
    callbacks.append(keras.callbacks.CSVLogger(csv_path, verbose=1))
    
    # Cosine annealing
    callbacks.append(keras.callbacks.CosineRestartSchedule(
        first_restart_step=20,
        t_mul=1.5,
        m_mul=0.8,
        alpha=0.1
    ))
    
    print(f"\n🎯 Starting training...")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    
    # Train model
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'advanced_spectrographic_final.h5')
    model.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    # Final evaluation
    print(f"\n📊 Final evaluation...")
    
    # Load best model if checkpointing was used
    if save_best_only and os.path.exists(checkpoint_path):
        print("Loading best checkpoint for evaluation...")
        best_model = keras.models.load_model(checkpoint_path, compile=False)
        best_model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        best_model = model
    
    # Predict on validation set
    y_pred_proba = best_model.predict(X_val, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    overall_accuracy = accuracy_score(y_val, y_pred)
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Per-class accuracy
    class_accuracies = {}
    print(f"\n📊 Per-class accuracy:")
    for i, class_name in enumerate(model_builder.class_names):
        class_mask = y_val == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_val[class_mask], y_pred[class_mask])
            class_accuracies[class_name] = class_acc
            status = '✅' if class_acc >= 0.95 else '❌'
            print(f"   {status} {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        else:
            class_accuracies[class_name] = 0.0
            print(f"   ⚠️  {class_name}: No validation samples")
    
    # Check if 95% target is met
    meets_target = (overall_accuracy >= 0.95 and 
                   all(acc >= 0.95 for acc in class_accuracies.values()))
    
    print(f"\n🎯 95% ACCURACY TARGET: {'✅ ACHIEVED' if meets_target else '❌ NOT MET'}")
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=model_builder.class_names))
    
    # Save results
    results = {
        'model_name': 'advanced_spectrographic_model_2024',
        'training_date': str(np.datetime64('today')),
        'dataset_path': str(dataset_dir),
        'samples_per_class': samples_per_class,
        'total_samples': len(audio_files),
        'training_samples': len(y_train),
        'validation_samples': len(y_val),
        'model_parameters': int(model.count_params()),
        'hyperparameters': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'validation_split': validation_split
        },
        'results': {
            'overall_accuracy': float(overall_accuracy),
            'class_accuracies': {k: float(v) for k, v in class_accuracies.items()},
            'meets_95_target': meets_target,
            'confusion_matrix': cm.tolist()
        },
        'model_files': {
            'best_checkpoint': checkpoint_path if save_best_only else None,
            'final_model': final_model_path,
            'training_log': csv_path
        },
        'features_used': list(X_features.keys())
    }
    
    results_path = os.path.join(output_dir, 'advanced_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {results_path}")
    
    if meets_target:
        print("\n🎉 CONGRATULATIONS! Advanced model achieved 95%+ accuracy target!")
    else:
        print(f"\n💡 Consider further hyperparameter tuning or more training data.")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train Advanced Spectrographic Model')
    parser.add_argument('--dataset_dir', default='massive_enhanced_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--samples_per_class', type=int, default=800,
                       help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--output_dir', default='models',
                       help='Output directory for models and results')
    
    args = parser.parse_args()
    
    # Check for required dependencies
    required_packages = ['scipy', 'librosa', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    # Train model
    results = train_advanced_model(
        dataset_dir=args.dataset_dir,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        output_dir=args.output_dir
    )
    
    if results:
        print("\n✅ Training pipeline completed successfully!")
        return 0
    else:
        print("\n❌ Training failed!")
        return 1

if __name__ == "__main__":
    # Fix for custom callback
    class CosineRestartSchedule(keras.callbacks.Callback):
        def __init__(self, first_restart_step, t_mul=2.0, m_mul=1.0, alpha=0.0):
            super().__init__()
            self.first_restart_step = first_restart_step
            self.t_mul = t_mul
            self.m_mul = m_mul
            self.alpha = alpha
            self.restart_step = first_restart_step
            self.last_restart = 0
            
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == self.restart_step:
                # Reset learning rate
                lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = lr * self.m_mul
                keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                
                # Schedule next restart
                self.last_restart = epoch
                self.restart_step = epoch + int(self.first_restart_step * self.t_mul)
                print(f"Cosine restart at epoch {epoch}, new LR: {new_lr:.6f}")
    
    # Add the callback to keras
    keras.callbacks.CosineRestartSchedule = CosineRestartSchedule
    
    exit_code = main()
    sys.exit(exit_code)