#!/usr/bin/env python3
"""
Focused Spectrographic Model Training
Streamlined training for the top 4 most promising techniques
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
from sklearn.metrics import classification_report, accuracy_score

sys.path.append('.')
from focused_spectrographic_model import FocusedSpectrogramicModel

def train_focused_model(dataset_dir="massive_enhanced_dataset", 
                       samples_per_class=600,
                       batch_size=16,
                       epochs=40,
                       learning_rate=0.001):
    """Train the focused spectrographic model"""
    
    print("🎯 TRAINING FOCUSED SPECTROGRAPHIC MODEL")
    print("=" * 60)
    
    # Initialize model
    model_builder = FocusedSpectrogramicModel()
    
    # Load dataset
    print(f"📊 Loading dataset from {dataset_dir}...")
    dataset_path = Path(dataset_dir)
    audio_files = []
    labels = []
    
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
    
    print(f"Total samples: {len(audio_files)}")
    
    # Extract features
    X_features, y = model_builder.prepare_focused_data(audio_files, labels)
    
    # Create model
    print("\n🏗️ Building focused model...")
    model = model_builder.create_focused_model()
    print(f"Parameters: {model.count_params():,}")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Split data
    train_indices, val_indices = train_test_split(
        range(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = {key: features[train_indices] for key, features in X_features.items()}
    X_val = {key: features[val_indices] for key, features in X_features.items()}
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    print(f"Training: {len(y_train)}, Validation: {len(y_val)}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('focused_spectrographic_best.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train
    print(f"\n🎯 Training (epochs: {epochs}, batch: {batch_size})...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final evaluation...")
    y_pred = np.argmax(model.predict(X_val), axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    for i, class_name in enumerate(model_builder.class_names):
        class_mask = y_val == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_val[class_mask], y_pred[class_mask])
            status = '✅' if class_acc >= 0.95 else '❌'
            print(f"{status} {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'meets_95_target': accuracy >= 0.95,
        'model_file': 'focused_spectrographic_best.h5',
        'parameters': int(model.count_params())
    }
    
    with open('focused_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 Model saved: focused_spectrographic_best.h5")
    print("💾 Results saved: focused_training_results.json")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='massive_enhanced_dataset')
    parser.add_argument('--samples_per_class', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    results = train_focused_model(
        dataset_dir=args.dataset_dir,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    if results['meets_95_target']:
        print("🎉 SUCCESS: Model achieved 95%+ accuracy!")
        return 0
    else:
        print("💡 Model needs further optimization")
        return 1

if __name__ == "__main__":
    exit(main())