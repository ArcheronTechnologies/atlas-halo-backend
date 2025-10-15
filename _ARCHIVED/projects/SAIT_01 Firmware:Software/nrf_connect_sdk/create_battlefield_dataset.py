#!/usr/bin/env python3
"""
Create Battlefield Enhanced Dataset
Generate synthetic combat sounds and create enhanced dataset
"""

import os
import sys
import numpy as np
import shutil
from pathlib import Path
import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class BattlefieldDatasetCreator:
    """Create enhanced dataset with synthetic battlefield audio"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.sample_rate = 22050
        self.duration = 3.0  # 3 seconds
        
    def create_synthetic_combat_sounds(self, output_dir="combat_sounds"):
        """Generate synthetic combat audio samples"""
        print("🔊 Creating synthetic combat sounds...")
        
        combat_dir = Path(output_dir)
        combat_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (combat_dir / "weapons").mkdir(exist_ok=True)
        (combat_dir / "explosions").mkdir(exist_ok=True)
        (combat_dir / "combat_vehicles").mkdir(exist_ok=True)
        (combat_dir / "combat_aircraft").mkdir(exist_ok=True)
        
        sounds_created = 0
        
        # Generate weapon sounds
        print("   Generating weapon sounds...")
        for i in range(50):
            # Create weapon sound (sharp transient + noise)
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            
            # Sharp attack transient
            attack = np.exp(-50 * t) * np.sin(2 * np.pi * 200 * t)
            attack += 0.5 * np.exp(-30 * t) * np.sin(2 * np.pi * 400 * t)
            
            # Add rifle crack
            crack = np.exp(-100 * t) * np.sin(2 * np.pi * 800 * t) 
            
            # Noise component
            noise = 0.3 * np.random.normal(0, 1, len(t))
            noise *= np.exp(-5 * t)  # Decay
            
            # Combine
            weapon_sound = attack + crack + noise
            weapon_sound = weapon_sound / np.max(np.abs(weapon_sound)) * 0.8
            
            filename = combat_dir / "weapons" / f"weapon_{i:03d}.wav"
            sf.write(str(filename), weapon_sound, self.sample_rate)
            sounds_created += 1
        
        # Generate explosion sounds
        print("   Generating explosion sounds...")
        for i in range(80):
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            
            # Low frequency boom
            boom = np.exp(-3 * t) * np.sin(2 * np.pi * 40 * t)
            boom += 0.7 * np.exp(-2 * t) * np.sin(2 * np.pi * 80 * t)
            
            # High frequency crack
            crack = np.exp(-20 * t) * np.sin(2 * np.pi * 1000 * t)
            
            # Rumble/debris
            rumble = 0.5 * np.random.normal(0, 1, len(t))
            rumble *= np.exp(-1 * t)
            
            explosion_sound = boom + crack + rumble
            explosion_sound = explosion_sound / np.max(np.abs(explosion_sound)) * 0.9
            
            filename = combat_dir / "explosions" / f"explosion_{i:03d}.wav"
            sf.write(str(filename), explosion_sound, self.sample_rate)
            sounds_created += 1
        
        # Generate combat vehicle sounds
        print("   Generating combat vehicle sounds...")
        for i in range(60):
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            
            # Engine rumble
            engine = 0.6 * np.sin(2 * np.pi * 60 * t)
            engine += 0.4 * np.sin(2 * np.pi * 120 * t)
            engine += 0.3 * np.sin(2 * np.pi * 180 * t)
            
            # Add weapon fire
            weapon_times = np.random.choice(len(t), size=3, replace=False)
            for wt in weapon_times:
                if wt < len(t) - 1000:
                    weapon_burst = np.exp(-50 * np.arange(1000) / self.sample_rate)
                    weapon_burst *= np.sin(2 * np.pi * 400 * np.arange(1000) / self.sample_rate)
                    engine[wt:wt+1000] += weapon_burst
            
            # Mechanical noise
            mech_noise = 0.2 * np.random.normal(0, 1, len(t))
            
            vehicle_sound = engine + mech_noise
            vehicle_sound = vehicle_sound / np.max(np.abs(vehicle_sound)) * 0.8
            
            filename = combat_dir / "combat_vehicles" / f"combat_vehicle_{i:03d}.wav"
            sf.write(str(filename), vehicle_sound, self.sample_rate)
            sounds_created += 1
        
        # Generate combat aircraft sounds
        print("   Generating combat aircraft sounds...")
        for i in range(40):
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            
            # Jet engine (high frequency)
            jet = 0.8 * np.sin(2 * np.pi * 400 * t)
            jet += 0.5 * np.sin(2 * np.pi * 800 * t)
            jet += 0.3 * np.sin(2 * np.pi * 1200 * t)
            
            # Doppler effect simulation
            freq_mod = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
            jet = jet * freq_mod
            
            # Add missile/gun sounds
            if np.random.random() > 0.5:
                weapon_time = np.random.randint(1000, len(t)-1000)
                weapon_burst = np.exp(-30 * np.arange(500) / self.sample_rate)
                weapon_burst *= np.sin(2 * np.pi * 600 * np.arange(500) / self.sample_rate)
                jet[weapon_time:weapon_time+500] += weapon_burst
            
            # Wind noise
            wind = 0.3 * np.random.normal(0, 1, len(t))
            
            aircraft_sound = jet + wind
            aircraft_sound = aircraft_sound / np.max(np.abs(aircraft_sound)) * 0.8
            
            filename = combat_dir / "combat_aircraft" / f"combat_aircraft_{i:03d}.wav"
            sf.write(str(filename), aircraft_sound, self.sample_rate)
            sounds_created += 1
        
        # Generate additional weapon variations
        for i in range(50):
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            
            # Machine gun burst
            burst_pattern = np.zeros(len(t))
            burst_start = np.random.randint(0, len(t)//2)
            burst_length = np.random.randint(500, 2000)
            
            for shot in range(0, burst_length, 100):
                if burst_start + shot < len(t) - 50:
                    shot_sound = np.exp(-80 * np.arange(50) / self.sample_rate)
                    shot_sound *= np.sin(2 * np.pi * 300 * np.arange(50) / self.sample_rate)
                    burst_pattern[burst_start + shot:burst_start + shot + 50] = shot_sound
            
            # Add echo
            echo_delay = int(0.1 * self.sample_rate)
            if echo_delay < len(burst_pattern):
                echo = np.zeros(len(burst_pattern))
                echo[echo_delay:] = 0.3 * burst_pattern[:-echo_delay]
                burst_pattern += echo
            
            burst_pattern = burst_pattern / (np.max(np.abs(burst_pattern)) + 1e-8) * 0.7
            
            filename = combat_dir / "weapons" / f"burst_{i:03d}.wav"
            sf.write(str(filename), burst_pattern, self.sample_rate)
            sounds_created += 1
        
        print(f"✅ Generated {sounds_created} synthetic combat sound files")
        return combat_dir
    
    def integrate_with_existing_dataset(self, combat_dir):
        """Integrate combat sounds with existing dataset"""
        print("🔗 Integrating combat sounds with existing dataset...")
        
        # Source and destination directories
        source_dir = Path("expanded_sait01_dataset")
        dest_dir = Path("enhanced_sait01_dataset")
        
        if not source_dir.exists():
            print("❌ Source dataset not found: expanded_sait01_dataset")
            return False
        
        # Create enhanced dataset directory
        dest_dir.mkdir(exist_ok=True)
        
        # Copy existing dataset
        for class_name in ['background', 'vehicle', 'aircraft']:
            source_class_dir = source_dir / class_name
            dest_class_dir = dest_dir / class_name
            dest_class_dir.mkdir(exist_ok=True)
            
            if source_class_dir.exists():
                # Copy existing files
                for audio_file in source_class_dir.glob("*.wav"):
                    shutil.copy2(audio_file, dest_class_dir)
        
        # Add combat sounds to appropriate classes
        combat_sounds = {
            'vehicle': [
                (combat_dir / "combat_vehicles", 60),
                (combat_dir / "weapons", 50)  # Vehicle-mounted weapons
            ],
            'aircraft': [
                (combat_dir / "combat_aircraft", 40)
            ],
            'background': [
                (combat_dir / "explosions", 80),  # Distant explosions
                (combat_dir / "weapons", 50)     # Distant gunfire
            ]
        }
        
        for class_name, sound_sources in combat_sounds.items():
            dest_class_dir = dest_dir / class_name
            
            for sound_dir, max_files in sound_sources:
                if sound_dir.exists():
                    sound_files = list(sound_dir.glob("*.wav"))[:max_files]
                    print(f"   Adding {len(sound_files)} {sound_dir.name} sounds to {class_name} class...")
                    
                    for i, sound_file in enumerate(sound_files):
                        dest_file = dest_class_dir / f"{class_name}_combat_{sound_dir.name}_{i:03d}.wav"
                        shutil.copy2(sound_file, dest_file)
        
        # Count final dataset
        total_counts = {}
        for class_name in ['background', 'vehicle', 'aircraft']:
            class_dir = dest_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.wav")))
                total_counts[class_name] = count
                print(f"   {class_name}: {count} samples")
        
        print("✅ Enhanced dataset created:")
        for class_name, count in total_counts.items():
            print(f"   {class_name}: {count} samples")
        
        return True

def main():
    """Create battlefield-enhanced dataset"""
    print("⚔️  SAIT_01 BATTLEFIELD DATASET CREATION")
    print("=" * 70)
    print("🎯 Objective: Create combat-enhanced dataset for 95% accuracy")
    print("🚀 Generating synthetic battlefield audio")
    print("=" * 70)
    
    creator = BattlefieldDatasetCreator()
    
    # Generate combat sounds
    combat_dir = creator.create_synthetic_combat_sounds()
    
    # Integrate with existing dataset
    success = creator.integrate_with_existing_dataset(combat_dir)
    
    if success:
        print(f"\n✅ BATTLEFIELD DATASET CREATION COMPLETE")
        print("=" * 70)
        print("🚀 Enhanced dataset ready at: enhanced_sait01_dataset/")
        print("⚡ Next step: Run python3 train_battlefield_model.py")
    else:
        print("❌ Dataset creation failed")

if __name__ == "__main__":
    main()