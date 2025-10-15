#!/usr/bin/env python3
"""
Meta-Learning Optimizer - Automatically design optimal training cycles
Uses neural architecture search and hyperparameter optimization
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import random
from pathlib import Path
from itertools import product
import concurrent.futures
from sklearn.model_selection import cross_val_score

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class MetaLearningOptimizer:
    """Automatically optimize training cycles using meta-learning"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.class_names = ['background', 'vehicle', 'aircraft']
        
        # Architecture search space
        self.arch_space = {
            'conv_layers': [2, 3, 4, 5],
            'conv_filters': [32, 64, 128, 256],
            'kernel_sizes': [(3,3), (5,5), (7,7), (3,5), (5,7)],
            'pool_sizes': [(2,2), (3,3), (2,3)],
            'dense_layers': [1, 2, 3, 4],
            'dense_units': [128, 256, 512, 1024],
            'dropout_rates': [0.2, 0.3, 0.4, 0.5, 0.6],
            'activation_functions': ['relu', 'elu', 'swish', 'gelu']
        }
        
        # Training hyperparameter space
        self.training_space = {
            'learning_rates': [0.0001, 0.0003, 0.0005, 0.001, 0.003],
            'batch_sizes': [8, 16, 32, 64],
            'optimizers': ['adam', 'adamw', 'rmsprop', 'sgd'],
            'class_weights': [
                {0: 1.0, 1: 1.0, 2: 1.0},  # Balanced
                {0: 1.0, 1: 1.5, 2: 2.0},  # Cycle 2 weights
                {0: 1.0, 1: 1.3, 2: 1.8},  # Cycle 4 weights
                {0: 1.2, 1: 1.5, 2: 2.2},  # Enhanced aircraft
                {0: 0.8, 1: 1.8, 2: 2.5}   # Strong vehicle/aircraft boost
            ],
            'augmentation_strength': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        # Feature engineering search space
        self.feature_space = {
            'mel_bands': [64, 80, 128],
            'mfcc_coeffs': [13, 20, 26],
            'spectral_features': [True, False],
            'harmonic_features': [True, False],
            'rhythmic_features': [True, False],
            'temporal_pooling': ['mean', 'max', 'attention', 'lstm']
        }
        
    def generate_random_architecture(self):
        """Generate random neural architecture"""
        arch = {}
        
        # Convolutional layers
        num_conv = random.choice(self.arch_space['conv_layers'])
        arch['conv_config'] = []
        
        for i in range(num_conv):
            layer_config = {
                'filters': random.choice(self.arch_space['conv_filters']),
                'kernel_size': random.choice(self.arch_space['kernel_sizes']),
                'activation': random.choice(self.arch_space['activation_functions']),
                'pool_size': random.choice(self.arch_space['pool_sizes']) if i % 2 == 1 else None,
                'dropout': random.choice(self.arch_space['dropout_rates'])
            }
            arch['conv_config'].append(layer_config)
        
        # Dense layers
        num_dense = random.choice(self.arch_space['dense_layers'])
        arch['dense_config'] = []
        
        for i in range(num_dense):
            layer_config = {
                'units': random.choice(self.arch_space['dense_units']),
                'activation': random.choice(self.arch_space['activation_functions']),
                'dropout': random.choice(self.arch_space['dropout_rates'])
            }
            arch['dense_config'].append(layer_config)
        
        return arch
    
    def generate_random_training_config(self):
        """Generate random training configuration"""
        return {
            'learning_rate': random.choice(self.training_space['learning_rates']),
            'batch_size': random.choice(self.training_space['batch_sizes']),
            'optimizer': random.choice(self.training_space['optimizers']),
            'class_weights': random.choice(self.training_space['class_weights']),
            'augmentation_strength': random.choice(self.training_space['augmentation_strength'])
        }
    
    def generate_random_feature_config(self):
        """Generate random feature configuration"""
        return {
            'mel_bands': random.choice(self.feature_space['mel_bands']),
            'mfcc_coeffs': random.choice(self.feature_space['mfcc_coeffs']),
            'spectral_features': random.choice(self.feature_space['spectral_features']),
            'harmonic_features': random.choice(self.feature_space['harmonic_features']),
            'rhythmic_features': random.choice(self.feature_space['rhythmic_features']),
            'temporal_pooling': random.choice(self.feature_space['temporal_pooling'])
        }
    
    def build_model_from_config(self, arch_config, feature_config):
        """Build model from configuration"""
        
        # Input shape based on feature config
        if feature_config['temporal_pooling'] == 'lstm':
            input_shape = (None, feature_config['mel_bands'])  # Variable length sequences
        else:
            input_shape = (feature_config['mel_bands'], 63, 1)  # Fixed size spectrograms
        
        # Build model
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Convolutional layers
        for i, conv_config in enumerate(arch_config['conv_config']):
            if feature_config['temporal_pooling'] == 'lstm' and i == 0:
                # For LSTM, use 1D convolutions
                x = layers.Conv1D(
                    filters=conv_config['filters'],
                    kernel_size=conv_config['kernel_size'][0],
                    activation=conv_config['activation'],
                    padding='same'
                )(x)
            else:
                x = layers.Conv2D(
                    filters=conv_config['filters'],
                    kernel_size=conv_config['kernel_size'],
                    activation=conv_config['activation'],
                    padding='same'
                )(x)
            
            if conv_config['pool_size']:
                if feature_config['temporal_pooling'] == 'lstm':
                    x = layers.MaxPooling1D(pool_size=conv_config['pool_size'][0])(x)
                else:
                    x = layers.MaxPooling2D(pool_size=conv_config['pool_size'])(x)
            
            if conv_config['dropout'] > 0:
                x = layers.Dropout(conv_config['dropout'])(x)
        
        # Temporal pooling
        if feature_config['temporal_pooling'] == 'lstm':
            x = layers.LSTM(128, return_sequences=False)(x)
        elif feature_config['temporal_pooling'] == 'attention':
            # Simple attention mechanism
            attention_weights = layers.Dense(1, activation='softmax')(x)
            x = layers.Multiply()([x, attention_weights])
            x = layers.GlobalAveragePooling2D()(x)
        elif feature_config['temporal_pooling'] == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:  # mean
            x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        for dense_config in arch_config['dense_config']:
            x = layers.Dense(
                units=dense_config['units'],
                activation=dense_config['activation']
            )(x)
            
            if dense_config['dropout'] > 0:
                x = layers.Dropout(dense_config['dropout'])(x)
        
        # Output layer
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def evaluate_configuration(self, arch_config, training_config, feature_config, X_sample, y_sample):
        """Evaluate a specific configuration combination"""
        try:
            # Build model
            model = self.build_model_from_config(arch_config, feature_config)
            
            # Choose optimizer
            if training_config['optimizer'] == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=training_config['learning_rate'])
            elif training_config['optimizer'] == 'adamw':
                optimizer = keras.optimizers.AdamW(learning_rate=training_config['learning_rate'])
            elif training_config['optimizer'] == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=training_config['learning_rate'])
            else:  # sgd
                optimizer = keras.optimizers.SGD(learning_rate=training_config['learning_rate'])
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Quick training (just a few epochs for evaluation)
            history = model.fit(
                X_sample, y_sample,
                batch_size=training_config['batch_size'],
                epochs=5,  # Quick evaluation
                class_weight=training_config['class_weights'],
                verbose=0,
                validation_split=0.2
            )
            
            # Return best validation accuracy
            best_accuracy = max(history.history['val_accuracy'])
            
            return {
                'accuracy': best_accuracy,
                'arch_config': arch_config,
                'training_config': training_config,
                'feature_config': feature_config,
                'model_params': model.count_params()
            }
            
        except Exception as e:
            print(f"Configuration failed: {e}")
            return {
                'accuracy': 0.0,
                'arch_config': arch_config,
                'training_config': training_config,
                'feature_config': feature_config,
                'model_params': 0,
                'error': str(e)
            }
    
    def neural_architecture_search(self, X_sample, y_sample, num_trials=50):
        """Perform neural architecture search"""
        print(f"🔍 Starting Neural Architecture Search ({num_trials} trials)...")
        
        results = []
        
        for trial in range(num_trials):
            print(f"   Trial {trial + 1}/{num_trials}")
            
            # Generate random configuration
            arch_config = self.generate_random_architecture()
            training_config = self.generate_random_training_config()
            feature_config = self.generate_random_feature_config()
            
            # Evaluate configuration
            result = self.evaluate_configuration(
                arch_config, training_config, feature_config, X_sample, y_sample
            )
            
            results.append(result)
            
            print(f"     Accuracy: {result['accuracy']:.4f}, Params: {result['model_params']:,}")
        
        # Sort by accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return results
    
    def evolutionary_optimization(self, X_sample, y_sample, num_generations=10, population_size=20):
        """Use evolutionary algorithm to optimize configurations"""
        print(f"🧬 Starting Evolutionary Optimization...")
        print(f"   Generations: {num_generations}, Population: {population_size}")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {
                'arch': self.generate_random_architecture(),
                'training': self.generate_random_training_config(),
                'features': self.generate_random_feature_config()
            }
            population.append(individual)
        
        best_results = []
        
        for generation in range(num_generations):
            print(f"\n🔄 Generation {generation + 1}/{num_generations}")
            
            # Evaluate population
            generation_results = []
            for i, individual in enumerate(population):
                result = self.evaluate_configuration(
                    individual['arch'], individual['training'], 
                    individual['features'], X_sample, y_sample
                )
                result['individual'] = individual
                generation_results.append(result)
                
                print(f"   Individual {i+1}: {result['accuracy']:.4f}")
            
            # Sort by fitness (accuracy)
            generation_results.sort(key=lambda x: x['accuracy'], reverse=True)
            best_results.append(generation_results[0])
            
            # Selection and mutation for next generation
            if generation < num_generations - 1:
                # Keep top 50% of population
                survivors = generation_results[:population_size//2]
                
                # Create new population through mutation
                new_population = []
                for survivor in survivors:
                    new_population.append(survivor['individual'])  # Keep original
                    
                    # Create mutated version
                    mutated = self.mutate_individual(survivor['individual'])
                    new_population.append(mutated)
                
                population = new_population
        
        return best_results
    
    def mutate_individual(self, individual):
        """Mutate an individual for evolutionary optimization"""
        mutated = {
            'arch': individual['arch'].copy(),
            'training': individual['training'].copy(),
            'features': individual['features'].copy()
        }
        
        # Randomly mutate some parameters
        if random.random() < 0.3:  # 30% chance to mutate architecture
            if 'conv_config' in mutated['arch'] and mutated['arch']['conv_config']:
                layer_idx = random.randint(0, len(mutated['arch']['conv_config']) - 1)
                param = random.choice(['filters', 'kernel_size', 'activation'])
                if param == 'filters':
                    mutated['arch']['conv_config'][layer_idx]['filters'] = random.choice(self.arch_space['conv_filters'])
                elif param == 'kernel_size':
                    mutated['arch']['conv_config'][layer_idx]['kernel_size'] = random.choice(self.arch_space['kernel_sizes'])
                else:
                    mutated['arch']['conv_config'][layer_idx]['activation'] = random.choice(self.arch_space['activation_functions'])
        
        if random.random() < 0.3:  # 30% chance to mutate training
            param = random.choice(['learning_rate', 'batch_size', 'optimizer'])
            if param == 'learning_rate':
                mutated['training']['learning_rate'] = random.choice(self.training_space['learning_rates'])
            elif param == 'batch_size':
                mutated['training']['batch_size'] = random.choice(self.training_space['batch_sizes'])
            else:
                mutated['training']['optimizer'] = random.choice(self.training_space['optimizers'])
        
        if random.random() < 0.3:  # 30% chance to mutate features
            param = random.choice(['mel_bands', 'mfcc_coeffs', 'temporal_pooling'])
            if param == 'mel_bands':
                mutated['features']['mel_bands'] = random.choice(self.feature_space['mel_bands'])
            elif param == 'mfcc_coeffs':
                mutated['features']['mfcc_coeffs'] = random.choice(self.feature_space['mfcc_coeffs'])
            else:
                mutated['features']['temporal_pooling'] = random.choice(self.feature_space['temporal_pooling'])
        
        return mutated
    
    def auto_design_cycle_6(self, X_sample, y_sample):
        """Automatically design optimal Cycle 6"""
        print("🤖 AUTO-DESIGNING CYCLE 6...")
        print("=" * 60)
        
        # Run neural architecture search
        nas_results = self.neural_architecture_search(X_sample, y_sample, num_trials=30)
        
        # Run evolutionary optimization 
        evo_results = self.evolutionary_optimization(X_sample, y_sample, num_generations=5, population_size=10)
        
        # Combine and find best overall configuration
        all_results = nas_results + evo_results
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        best_config = all_results[0]
        
        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"   Validation Accuracy: {best_config['accuracy']:.4f}")
        print(f"   Model Parameters: {best_config['model_params']:,}")
        
        # Save optimal configuration
        with open('cycle_6_optimal_config.json', 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
        
        return best_config

def main():
    print("🧠 META-LEARNING OPTIMIZER")
    print("=" * 50)
    print("🤖 Automatically designing optimal training cycles")
    
    optimizer = MetaLearningOptimizer()
    
    # Load sample data for optimization
    print("\n📊 Loading sample data for meta-learning...")
    dataset_dir = Path("massive_enhanced_dataset")
    
    # Use smaller sample for meta-learning (faster iteration)
    X_sample = []
    y_sample = []
    samples_per_class = 100
    
    for class_idx, class_name in enumerate(optimizer.class_names):
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            audio_files = list(class_dir.glob("*.wav"))[:samples_per_class]
            
            for audio_file in audio_files:
                try:
                    audio = optimizer.preprocessor.load_and_resample(audio_file)
                    features = optimizer.preprocessor.extract_mel_spectrogram(audio)
                    
                    if len(features.shape) == 2:
                        features = np.expand_dims(features, axis=-1)
                    
                    X_sample.append(features)
                    y_sample.append(class_idx)
                except:
                    continue
    
    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)
    
    print(f"Sample data shape: {X_sample.shape}")
    
    # Auto-design optimal Cycle 6
    optimal_config = optimizer.auto_design_cycle_6(X_sample, y_sample)
    
    print("\n🎯 CYCLE 6 OPTIMIZATION COMPLETE!")
    print("💾 Configuration saved to cycle_6_optimal_config.json")

if __name__ == "__main__":
    main()