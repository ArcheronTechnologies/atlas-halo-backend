#!/usr/bin/env python3
"""
SAIT_01 Dataset Expansion Pipeline
Combines multiple public datasets with advanced augmentation for maximum accuracy
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random
import json
import shutil
from collections import defaultdict

class SAIT01DatasetExpander:
    """Comprehensive dataset expansion for SAIT_01 audio classification"""
    
    def __init__(self, target_samples_per_class=2000):
        self.target_samples = target_samples_per_class
        self.sample_rate = 16000
        self.duration = 1.0  # 1 second clips
        
        # Class mapping
        self.sait_classes = {
            'background': 0,
            'vehicle': 1,  # Including drones
            'aircraft': 2   # Including helicopters
        }
        
        print(f"🎯 SAIT_01 Dataset Expansion Pipeline")
        print(f"📊 Target: {target_samples_per_class} samples per class")
        print("=" * 60)
    
    def load_esc50_metadata(self):
        """Load ESC-50 metadata to identify relevant classes"""
        meta_path = Path("expanded_datasets/ESC-50-master/meta/esc50.csv")
        if not meta_path.exists():
            print("❌ ESC-50 metadata not found")
            return None
            
        df = pd.read_csv(meta_path)
        print(f"📋 ESC-50 loaded: {len(df)} samples, {df['category'].nunique()} categories")
        
        # Map ESC-50 classes to SAIT classes
        esc_to_sait = {
            # Background sounds
            'rain': 'background',
            'wind': 'background', 
            'sea_waves': 'background',
            'crackling_fire': 'background',
            'water_drops': 'background',
            'thunderstorm': 'background',
            'chirping_birds': 'background',
            
            # Vehicle sounds
            'car_horn': 'vehicle',
            'engine': 'vehicle',
            'train': 'vehicle',
            'truck': 'vehicle',
            
            # Aircraft sounds  
            'airplane': 'aircraft',
            'helicopter': 'aircraft'
        }
        
        # Filter relevant samples
        relevant_samples = []
        for idx, row in df.iterrows():
            if row['category'] in esc_to_sait:
                relevant_samples.append({
                    'filename': row['filename'],
                    'sait_class': esc_to_sait[row['category']],
                    'esc_class': row['category'],
                    'fold': row['fold']
                })
        
        print(f"📊 ESC-50 relevant samples: {len(relevant_samples)}")
        for sait_class in self.sait_classes.keys():
            count = sum(1 for s in relevant_samples if s['sait_class'] == sait_class)
            print(f"   {sait_class}: {count} samples")
        
        return relevant_samples
    
    def load_existing_dataset(self):
        """Load existing SAIT_01 dataset"""
        existing_data = []
        
        # Check for existing balanced data
        for sait_class in self.sait_classes.keys():
            class_dir = Path(f"quick_balanced_data/{sait_class}")
            if class_dir.exists():
                audio_files = list(class_dir.glob("*.wav"))
                for audio_file in audio_files:
                    existing_data.append({
                        'path': audio_file,
                        'sait_class': sait_class,
                        'source': 'existing'
                    })
        
        print(f"📊 Existing SAIT_01 samples: {len(existing_data)}")
        for sait_class in self.sait_classes.keys():
            count = sum(1 for s in existing_data if s['sait_class'] == sait_class)
            print(f"   {sait_class}: {count} samples")
        
        return existing_data
    
    def advanced_augment_audio(self, audio, augmentation_type):
        """Apply advanced audio augmentation techniques"""
        augmented = audio.copy()
        
        try:
            if augmentation_type == 'time_stretch':
                # Time stretching (80% to 120% speed)
                rate = random.uniform(0.8, 1.2)
                augmented = librosa.effects.time_stretch(augmented, rate=rate)
                
            elif augmentation_type == 'pitch_shift':
                # Pitch shifting (±4 semitones)
                n_steps = random.uniform(-4, 4)
                augmented = librosa.effects.pitch_shift(
                    augmented, sr=self.sample_rate, n_steps=n_steps
                )
                
            elif augmentation_type == 'noise_injection':
                # Add background noise
                noise_factor = random.uniform(0.005, 0.02)
                noise = np.random.randn(len(augmented)) * noise_factor
                augmented = augmented + noise
                
            elif augmentation_type == 'frequency_mask':
                # Spectral masking in frequency domain
                stft = librosa.stft(augmented)
                stft_db = librosa.amplitude_to_db(np.abs(stft))
                
                # Mask random frequency bands
                freq_mask_width = random.randint(5, 15)
                freq_start = random.randint(0, stft_db.shape[0] - freq_mask_width)
                stft_db[freq_start:freq_start + freq_mask_width, :] *= 0.1
                
                # Convert back to audio
                stft_masked = librosa.db_to_amplitude(stft_db) * np.exp(1j * np.angle(stft))
                augmented = librosa.istft(stft_masked)
                
            elif augmentation_type == 'time_mask':
                # Temporal masking
                mask_length = random.randint(int(0.05 * len(augmented)), int(0.15 * len(augmented)))
                mask_start = random.randint(0, len(augmented) - mask_length)
                augmented[mask_start:mask_start + mask_length] *= 0.1
                
            elif augmentation_type == 'dynamic_compression':
                # Dynamic range compression
                compression = random.uniform(0.6, 1.4)
                augmented = np.sign(augmented) * np.power(np.abs(augmented), compression)
                
            elif augmentation_type == 'harmonic_distortion':
                # Add harmonic distortion
                distortion = random.uniform(0.1, 0.3)
                augmented = augmented + distortion * augmented**2
                
            # Normalize to prevent clipping
            if np.max(np.abs(augmented)) > 0:
                augmented = augmented / np.max(np.abs(augmented)) * 0.9
                
        except Exception as e:
            print(f"⚠️  Augmentation failed ({augmentation_type}): {e}")
            return audio
        
        # Ensure correct length
        target_length = int(self.sample_rate * self.duration)
        if len(augmented) > target_length:
            start = random.randint(0, len(augmented) - target_length)
            augmented = augmented[start:start + target_length]
        elif len(augmented) < target_length:
            augmented = np.pad(augmented, (0, target_length - len(augmented)))
        
        return augmented
    
    def generate_synthetic_audio(self, class_type, num_samples=500):
        """Generate synthetic audio samples for each class"""
        print(f"🔄 Generating {num_samples} synthetic {class_type} samples...")
        
        synthetic_samples = []
        
        for i in tqdm(range(num_samples), desc=f"Generating {class_type}"):
            if class_type == 'background':
                # Generate nature sounds
                duration = int(self.sample_rate * self.duration)
                
                # Base noise
                audio = np.random.randn(duration) * 0.1
                
                # Add periodic components for wind/water
                t = np.linspace(0, self.duration, duration)
                
                # Wind-like low frequency component
                wind = 0.2 * np.sin(2 * np.pi * 0.5 * t) * np.random.randn(duration) * 0.3
                
                # Water-like mid frequency
                water = 0.1 * np.sin(2 * np.pi * 8 * t + np.random.randn(duration) * 0.5)
                
                audio = audio + wind + water
                
            elif class_type == 'vehicle':
                # Generate drone/vehicle sounds
                duration = int(self.sample_rate * self.duration)
                t = np.linspace(0, self.duration, duration)
                
                # Base engine frequency
                engine_freq = random.uniform(80, 200)
                audio = 0.3 * np.sin(2 * np.pi * engine_freq * t)
                
                # Add harmonics
                for harmonic in [2, 3, 4]:
                    audio += 0.1 * np.sin(2 * np.pi * engine_freq * harmonic * t)
                
                # Add rotor blade effects for drones
                rotor_freq = random.uniform(20, 40)
                rotor = 0.2 * np.sin(2 * np.pi * rotor_freq * t)
                audio = audio * (1 + rotor)
                
                # Add noise
                audio += np.random.randn(duration) * 0.05
                
            elif class_type == 'aircraft':
                # Generate aircraft/helicopter sounds
                duration = int(self.sample_rate * self.duration)
                t = np.linspace(0, self.duration, duration)
                
                # Turbine/jet engine frequency
                turbine_freq = random.uniform(200, 800)
                audio = 0.4 * np.sin(2 * np.pi * turbine_freq * t)
                
                # Add broadband noise for jet
                audio += np.random.randn(duration) * 0.3
                
                # For helicopters, add rotor blade modulation
                if random.random() < 0.5:
                    rotor_freq = random.uniform(15, 25)
                    rotor = 0.4 * np.sin(2 * np.pi * rotor_freq * t)
                    audio = audio * (1 + rotor)
                
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            synthetic_samples.append({
                'audio': audio,
                'sait_class': class_type,
                'source': 'synthetic',
                'index': i
            })
        
        return synthetic_samples
    
    def create_expanded_dataset(self):
        """Create comprehensive expanded dataset"""
        print("🚀 Creating expanded dataset...")
        
        # Create output directory
        expanded_dir = Path("expanded_sait01_dataset")
        expanded_dir.mkdir(exist_ok=True)
        
        for class_name in self.sait_classes.keys():
            (expanded_dir / class_name).mkdir(exist_ok=True)
        
        # Load existing data
        existing_data = self.load_existing_dataset()
        
        # Load ESC-50 data
        esc50_samples = self.load_esc50_metadata()
        
        # Process each class
        all_samples = defaultdict(list)
        
        # Add existing samples
        for sample in existing_data:
            all_samples[sample['sait_class']].append(sample)
        
        # Add ESC-50 samples
        if esc50_samples:
            esc_audio_dir = Path("expanded_datasets/ESC-50-master/audio")
            for sample in esc50_samples:
                audio_path = esc_audio_dir / sample['filename']
                if audio_path.exists():
                    all_samples[sample['sait_class']].append({
                        'path': audio_path,
                        'sait_class': sample['sait_class'],
                        'source': 'esc50'
                    })
        
        # Generate synthetic samples to fill gaps
        for class_name in self.sait_classes.keys():
            current_count = len(all_samples[class_name])
            needed = max(0, self.target_samples // 3 - current_count)  # Reserve space for augmentation
            
            if needed > 0:
                print(f"📈 Generating {needed} synthetic {class_name} samples...")
                synthetic = self.generate_synthetic_audio(class_name, needed)
                
                for i, sample in enumerate(synthetic):
                    audio_path = expanded_dir / class_name / f"synthetic_{i:04d}.wav"
                    sf.write(audio_path, sample['audio'], self.sample_rate)
                    all_samples[class_name].append({
                        'path': audio_path,
                        'sait_class': class_name,
                        'source': 'synthetic'
                    })
        
        # Copy and augment existing samples
        augmentation_types = [
            'time_stretch', 'pitch_shift', 'noise_injection',
            'frequency_mask', 'time_mask', 'dynamic_compression', 'harmonic_distortion'
        ]
        
        for class_name in self.sait_classes.keys():
            class_dir = expanded_dir / class_name
            samples = all_samples[class_name]
            
            print(f"🔄 Processing {class_name}: {len(samples)} base samples")
            
            # Copy original samples
            for i, sample in enumerate(tqdm(samples, desc=f"Copying {class_name}")):
                if 'path' in sample:
                    try:
                        audio, sr = librosa.load(sample['path'], sr=self.sample_rate)
                        
                        # Ensure correct duration
                        target_length = int(self.sample_rate * self.duration)
                        if len(audio) > target_length:
                            start = random.randint(0, len(audio) - target_length)
                            audio = audio[start:start + target_length]
                        elif len(audio) < target_length:
                            audio = np.pad(audio, (0, target_length - len(audio)))
                        
                        # Save original
                        orig_path = class_dir / f"original_{i:04d}.wav"
                        sf.write(orig_path, audio, self.sample_rate)
                        
                        # Create augmented versions
                        current_count = len(list(class_dir.glob("*.wav")))
                        augmentations_needed = min(len(augmentation_types), 
                                                 self.target_samples - current_count)
                        
                        for aug_idx, aug_type in enumerate(augmentation_types[:augmentations_needed]):
                            aug_audio = self.advanced_augment_audio(audio, aug_type)
                            aug_path = class_dir / f"aug_{aug_type}_{i:04d}_{aug_idx:02d}.wav"
                            sf.write(aug_path, aug_audio, self.sample_rate)
                            
                    except Exception as e:
                        print(f"⚠️  Error processing {sample['path']}: {e}")
                        continue
        
        # Final count
        print(f"\n📊 Expanded Dataset Summary:")
        total_samples = 0
        for class_name in self.sait_classes.keys():
            class_dir = expanded_dir / class_name
            count = len(list(class_dir.glob("*.wav")))
            total_samples += count
            print(f"   {class_name}: {count} samples")
        
        print(f"📈 Total: {total_samples} samples")
        print(f"✅ Expanded dataset saved to: {expanded_dir}")
        
        return expanded_dir
    
    def create_training_metadata(self, dataset_dir):
        """Create metadata file for training"""
        metadata = {
            'dataset_info': {
                'total_samples': 0,
                'classes': self.sait_classes,
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'created': str(pd.Timestamp.now())
            },
            'class_distribution': {},
            'augmentation_types': [
                'time_stretch', 'pitch_shift', 'noise_injection',
                'frequency_mask', 'time_mask', 'dynamic_compression', 'harmonic_distortion'
            ]
        }
        
        total = 0
        for class_name in self.sait_classes.keys():
            class_dir = Path(dataset_dir) / class_name
            count = len(list(class_dir.glob("*.wav")))
            metadata['class_distribution'][class_name] = count
            total += count
        
        metadata['dataset_info']['total_samples'] = total
        
        # Save metadata
        with open(Path(dataset_dir) / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Metadata saved to: {dataset_dir}/metadata.json")
        return metadata

def main():
    """Main execution function"""
    print("🎯 SAIT_01 Comprehensive Dataset Expansion")
    print("=" * 70)
    
    # Initialize expander
    expander = SAIT01DatasetExpander(target_samples_per_class=2500)
    
    # Create expanded dataset
    dataset_dir = expander.create_expanded_dataset()
    
    # Create metadata
    metadata = expander.create_training_metadata(dataset_dir)
    
    print(f"\n🎉 Dataset Expansion Complete!")
    print("=" * 70)
    print(f"📊 Total Samples: {metadata['dataset_info']['total_samples']}")
    print(f"📁 Dataset Location: {dataset_dir}")
    print(f"🚀 Ready for high-accuracy training!")
    
    # Training recommendation
    if metadata['dataset_info']['total_samples'] >= 6000:
        print("\n✅ EXCELLENT: Dataset size optimal for 85%+ accuracy")
    elif metadata['dataset_info']['total_samples'] >= 3000:
        print("\n✅ GOOD: Dataset size sufficient for 70%+ accuracy")
    else:
        print("\n⚠️  MODERATE: Dataset may need further expansion for highest accuracy")

if __name__ == "__main__":
    main()