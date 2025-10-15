#!/usr/bin/env python3
"""
Dataset Balancing Script for SAIT_01
Fixes class imbalance and prepares optimized training data
"""

import os
import shutil
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm
import random

def balance_existing_dataset(data_dir="edth-copenhagen-drone-acoustics/data/raw"):
    """Balance the existing dataset to fix class imbalance"""
    
    print("🔄 Balancing SAIT_01 Dataset")
    print("=" * 40)
    
    data_path = Path(data_dir)
    balanced_path = Path("balanced_sait01_data")
    balanced_path.mkdir(exist_ok=True)
    
    # Create balanced class directories
    classes = {
        'background': 'background',
        'drone': 'drone', 
        'helicopter': 'aircraft'  # Merge helicopter into aircraft for simplicity
    }
    
    for cls in ['background', 'drone', 'aircraft']:
        (balanced_path / cls).mkdir(exist_ok=True)
    
    # Count original samples
    original_counts = {}
    for src_class, dst_class in classes.items():
        train_dir = data_path / "train" / src_class
        val_dir = data_path / "val" / src_class
        
        train_files = list(train_dir.glob("*.wav")) if train_dir.exists() else []
        val_files = list(val_dir.glob("*.wav")) if val_dir.exists() else []
        
        original_counts[src_class] = len(train_files) + len(val_files)
        print(f"📊 {src_class}: {original_counts[src_class]} samples")
    
    # Determine target sample count per class
    target_per_class = 800  # Balanced target
    
    print(f"\n🎯 Target per class: {target_per_class} samples")
    print("🔄 Processing classes...")
    
    # Process each class
    for src_class, dst_class in classes.items():
        print(f"\n📁 Processing {src_class} -> {dst_class}")
        
        # Collect all files for this class
        all_files = []
        for split in ['train', 'val']:
            split_dir = data_path / split / src_class
            if split_dir.exists():
                all_files.extend(list(split_dir.glob("*.wav")))
        
        # Limit background samples, expand others
        if src_class == 'background':
            # Randomly sample background to reduce class imbalance
            if len(all_files) > target_per_class:
                selected_files = random.sample(all_files, target_per_class)
            else:
                selected_files = all_files
        else:
            # For drone/aircraft classes, use all available
            selected_files = all_files
        
        # Copy selected files
        dst_dir = balanced_path / dst_class
        for i, src_file in enumerate(tqdm(selected_files, desc=f"Copying {src_class}")):
            dst_file = dst_dir / f"{src_class}_{i:04d}.wav"
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        
        print(f"✅ {dst_class}: {len(selected_files)} samples")
    
    # Final count
    print(f"\n📈 Balanced Dataset Summary:")
    total_balanced = 0
    for cls in ['background', 'drone', 'aircraft']:
        cls_dir = balanced_path / cls
        count = len(list(cls_dir.glob("*.wav")))
        total_balanced += count
        print(f"   {cls}: {count} samples")
    
    print(f"📊 Total: {total_balanced} samples")
    print(f"✅ Balanced dataset saved to: {balanced_path}")
    
    return balanced_path

def create_augmented_samples(balanced_dir, augment_factor=3):
    """Create augmented samples to expand the dataset"""
    
    print(f"\n🔄 Creating {augment_factor}x augmented samples...")
    
    balanced_path = Path(balanced_dir)
    augmented_path = Path("augmented_sait01_data") 
    augmented_path.mkdir(exist_ok=True)
    
    for cls in ['background', 'drone', 'aircraft']:
        src_dir = balanced_path / cls
        dst_dir = augmented_path / cls
        dst_dir.mkdir(exist_ok=True)
        
        if not src_dir.exists():
            continue
            
        audio_files = list(src_dir.glob("*.wav"))
        print(f"📁 Augmenting {cls}: {len(audio_files)} -> {len(audio_files) * augment_factor}")
        
        for file_idx, audio_file in enumerate(tqdm(audio_files, desc=f"Augmenting {cls}")):
            # Load original audio
            try:
                audio, sr = librosa.load(audio_file, sr=16000)
                
                # Copy original
                shutil.copy2(audio_file, dst_dir / f"{cls}_orig_{file_idx:04d}.wav")
                
                # Create augmented versions
                for aug_idx in range(augment_factor - 1):
                    augmented_audio = augment_audio_sample(audio, sr)
                    
                    # Save augmented sample
                    aug_file = dst_dir / f"{cls}_aug_{file_idx:04d}_{aug_idx:02d}.wav"
                    librosa.output.write_wav(aug_file, augmented_audio, sr)
                    
            except Exception as e:
                print(f"⚠️  Error processing {audio_file}: {e}")
                continue
    
    # Final augmented count
    print(f"\n📈 Augmented Dataset Summary:")
    total_augmented = 0
    for cls in ['background', 'drone', 'aircraft']:
        cls_dir = augmented_path / cls
        count = len(list(cls_dir.glob("*.wav"))) if cls_dir.exists() else 0
        total_augmented += count
        print(f"   {cls}: {count} samples")
    
    print(f"📊 Total: {total_augmented} samples")
    print(f"✅ Augmented dataset saved to: {augmented_path}")
    
    return augmented_path

def augment_audio_sample(audio, sr):
    """Apply random augmentation to audio sample"""
    
    augmented = audio.copy()
    
    # Random time stretching (±20%)
    if random.random() < 0.6:
        rate = random.uniform(0.8, 1.2)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
    
    # Random pitch shifting (±3 semitones)  
    if random.random() < 0.6:
        n_steps = random.uniform(-3, 3)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
    
    # Add random noise
    if random.random() < 0.4:
        noise_factor = random.uniform(0.01, 0.03)
        noise = np.random.randn(len(augmented)) * noise_factor
        augmented = augmented + noise
    
    # Normalize length to 1 second
    target_length = sr  # 16000 samples = 1 second
    if len(augmented) > target_length:
        # Random crop
        start = random.randint(0, len(augmented) - target_length)
        augmented = augmented[start:start + target_length]
    elif len(augmented) < target_length:
        # Pad with zeros
        augmented = np.pad(augmented, (0, target_length - len(augmented)))
    
    # Normalize amplitude
    if np.max(np.abs(augmented)) > 0:
        augmented = augmented / np.max(np.abs(augmented)) * 0.9
    
    return augmented

def analyze_dataset_quality(dataset_dir):
    """Analyze dataset quality and provide recommendations"""
    
    print(f"\n🔍 Analyzing Dataset Quality: {dataset_dir}")
    print("=" * 50)
    
    dataset_path = Path(dataset_dir)
    
    analysis = {
        'total_samples': 0,
        'class_distribution': {},
        'duration_stats': {},
        'quality_issues': []
    }
    
    for cls_dir in dataset_path.glob("*"):
        if cls_dir.is_dir():
            cls_name = cls_dir.name
            audio_files = list(cls_dir.glob("*.wav"))
            
            analysis['class_distribution'][cls_name] = len(audio_files)
            analysis['total_samples'] += len(audio_files)
            
            # Analyze sample quality
            durations = []
            for audio_file in audio_files[:50]:  # Sample first 50 files
                try:
                    audio, sr = librosa.load(audio_file, sr=None)
                    duration = len(audio) / sr
                    durations.append(duration)
                except:
                    analysis['quality_issues'].append(f"Failed to load: {audio_file}")
            
            if durations:
                analysis['duration_stats'][cls_name] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations)
                }
    
    # Print analysis
    print(f"📊 Total Samples: {analysis['total_samples']}")
    print(f"📈 Class Distribution:")
    for cls, count in analysis['class_distribution'].items():
        percentage = (count / analysis['total_samples']) * 100
        print(f"   {cls}: {count} ({percentage:.1f}%)")
    
    print(f"\n⏱️  Duration Statistics:")
    for cls, stats in analysis['duration_stats'].items():
        print(f"   {cls}: {stats['mean']:.2f}s ± {stats['std']:.2f}s")
    
    # Check balance
    counts = list(analysis['class_distribution'].values())
    if len(counts) > 1:
        balance_ratio = min(counts) / max(counts)
        print(f"\n⚖️  Class Balance Ratio: {balance_ratio:.3f}")
        if balance_ratio < 0.3:
            print("⚠️  SEVERE CLASS IMBALANCE - Consider rebalancing")
        elif balance_ratio < 0.5:
            print("⚠️  Moderate class imbalance - Augmentation recommended")
        else:
            print("✅ Well balanced dataset")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    total = analysis['total_samples']
    if total < 1000:
        print("   📈 Dataset too small - Recommend 2000+ samples minimum")
    elif total < 3000:
        print("   📈 Small dataset - Aggressive augmentation recommended")
    else:
        print("   ✅ Good dataset size for training")
    
    if analysis['quality_issues']:
        print(f"   🔧 Fix {len(analysis['quality_issues'])} corrupted files")
    
    return analysis

def main():
    """Main execution function"""
    
    print("🎯 SAIT_01 Dataset Optimization Pipeline")
    print("=" * 50)
    
    # Step 1: Balance existing dataset
    print("\n📊 Step 1: Balance Dataset")
    balanced_dir = balance_existing_dataset()
    
    # Step 2: Analyze balanced dataset
    print("\n🔍 Step 2: Analyze Balanced Dataset")
    analysis = analyze_dataset_quality(balanced_dir)
    
    # Step 3: Create augmented dataset if needed
    if analysis['total_samples'] < 2000:
        print("\n🔄 Step 3: Create Augmented Dataset")
        augmented_dir = create_augmented_samples(balanced_dir, augment_factor=4)
        
        print("\n🔍 Step 4: Analyze Final Dataset")
        final_analysis = analyze_dataset_quality(augmented_dir)
        
        return augmented_dir
    else:
        print("\n✅ Dataset size sufficient - No augmentation needed")
        return balanced_dir

if __name__ == "__main__":
    final_dataset = main()
    print(f"\n🎉 Dataset optimization complete!")
    print(f"📁 Final dataset: {final_dataset}")
    print(f"🚀 Ready for enhanced model training")