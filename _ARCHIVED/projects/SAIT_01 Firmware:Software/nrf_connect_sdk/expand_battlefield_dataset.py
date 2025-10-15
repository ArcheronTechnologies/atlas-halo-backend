#!/usr/bin/env python3
"""
Expand Battlefield Dataset
Create a much larger dataset with diverse combat sounds and variations
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
import random

# Add current directory for imports
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class MassiveBattlefieldDatasetCreator:
    """Create massive enhanced dataset with extensive combat variations"""
    
    def __init__(self):
        self.preprocessor = SaitAudioPreprocessor()
        self.sample_rate = 22050
        self.duration = 3.0  # 3 seconds
        
    def create_massive_combat_sounds(self, output_dir="massive_combat_sounds"):
        """Generate massive variety of synthetic combat audio"""
        print("🎯 Creating MASSIVE combat sound library...")
        
        combat_dir = Path(output_dir)
        combat_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (combat_dir / "weapons").mkdir(exist_ok=True)
        (combat_dir / "explosions").mkdir(exist_ok=True)
        (combat_dir / "combat_vehicles").mkdir(exist_ok=True)
        (combat_dir / "combat_aircraft").mkdir(exist_ok=True)
        (combat_dir / "mixed_combat").mkdir(exist_ok=True)
        
        sounds_created = 0
        
        # MASSIVE weapon sound generation (500 samples)
        print("   Generating 500+ weapon sound variations...")
        weapon_types = [
            # Rifle variants
            {'name': 'rifle', 'freq': [200, 400], 'decay': [30, 80], 'crack_freq': [600, 1000]},
            {'name': 'assault_rifle', 'freq': [150, 350], 'decay': [40, 90], 'crack_freq': [500, 900]},
            {'name': 'sniper_rifle', 'freq': [100, 250], 'decay': [20, 60], 'crack_freq': [800, 1200]},
            # Machine guns
            {'name': 'machine_gun', 'freq': [180, 320], 'decay': [50, 100], 'crack_freq': [400, 800]},
            {'name': 'heavy_mg', 'freq': [120, 280], 'decay': [60, 120], 'crack_freq': [300, 700]},
            # Pistols
            {'name': 'pistol', 'freq': [250, 450], 'decay': [40, 70], 'crack_freq': [700, 1100]},
            # Automatic weapons
            {'name': 'submachine_gun', 'freq': [200, 380], 'decay': [35, 75], 'crack_freq': [500, 900]},
        ]
        
        for weapon in weapon_types:
            for i in range(75):  # 75 variants per weapon type
                t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
                
                # Randomize parameters
                base_freq = random.uniform(weapon['freq'][0], weapon['freq'][1])
                decay_rate = random.uniform(weapon['decay'][0], weapon['decay'][1])
                crack_freq = random.uniform(weapon['crack_freq'][0], weapon['crack_freq'][1])
                
                # Generate weapon sound
                attack = np.exp(-decay_rate * t) * np.sin(2 * np.pi * base_freq * t)
                attack += 0.5 * np.exp(-decay_rate * 0.8 * t) * np.sin(2 * np.pi * base_freq * 2 * t)
                
                # Add muzzle blast/crack
                crack = np.exp(-100 * t) * np.sin(2 * np.pi * crack_freq * t)
                
                # Add realistic noise
                noise = 0.3 * np.random.normal(0, 1, len(t))
                noise *= np.exp(-random.uniform(3, 8) * t)
                
                # Combine with random variations
                weapon_sound = attack + crack + noise
                
                # Add random echo/reverberation
                if random.random() > 0.5:
                    echo_delay = random.randint(int(0.05 * self.sample_rate), int(0.2 * self.sample_rate))
                    if echo_delay < len(weapon_sound):
                        echo = np.zeros(len(weapon_sound))
                        echo[echo_delay:] = 0.3 * weapon_sound[:-echo_delay]
                        weapon_sound += echo
                
                weapon_sound = weapon_sound / (np.max(np.abs(weapon_sound)) + 1e-8) * 0.8
                
                filename = combat_dir / "weapons" / f"{weapon['name']}_{i:03d}.wav"
                sf.write(str(filename), weapon_sound, self.sample_rate)
                sounds_created += 1
        
        # MASSIVE explosion generation (400 samples)
        print("   Generating 400+ explosion variations...")
        explosion_types = [
            {'name': 'grenade', 'boom_freq': [30, 60], 'duration_mult': 1.0},
            {'name': 'artillery', 'boom_freq': [20, 40], 'duration_mult': 1.5},
            {'name': 'mortar', 'boom_freq': [25, 50], 'duration_mult': 1.2},
            {'name': 'ied', 'boom_freq': [35, 70], 'duration_mult': 0.8},
            {'name': 'rpg', 'boom_freq': [40, 80], 'duration_mult': 1.1},
            {'name': 'tank_shell', 'boom_freq': [15, 35], 'duration_mult': 1.8},
            {'name': 'mine', 'boom_freq': [30, 65], 'duration_mult': 0.9},
            {'name': 'cluster', 'boom_freq': [40, 85], 'duration_mult': 0.7},
        ]
        
        for explosion in explosion_types:
            for i in range(50):  # 50 variants per explosion type
                t = np.linspace(0, self.duration * explosion['duration_mult'], 
                              int(self.sample_rate * self.duration * explosion['duration_mult']))
                
                # Randomize parameters
                boom_freq = random.uniform(explosion['boom_freq'][0], explosion['boom_freq'][1])
                decay_rate = random.uniform(1, 4)
                
                # Low frequency boom
                boom = np.exp(-decay_rate * t) * np.sin(2 * np.pi * boom_freq * t)
                boom += 0.7 * np.exp(-decay_rate * 0.8 * t) * np.sin(2 * np.pi * boom_freq * 2 * t)
                
                # High frequency crack/shrapnel
                crack_freq = random.uniform(800, 1500)
                crack = np.exp(-20 * t) * np.sin(2 * np.pi * crack_freq * t)
                
                # Debris/rumble
                rumble = 0.6 * np.random.normal(0, 1, len(t))
                rumble *= np.exp(-random.uniform(0.5, 2) * t)
                
                # Multiple explosion effect for cluster munitions
                if explosion['name'] == 'cluster' and random.random() > 0.3:
                    for burst in range(random.randint(2, 5)):
                        burst_start = random.randint(0, len(t)//3)
                        burst_length = min(int(0.3 * self.sample_rate), len(t) - burst_start)
                        mini_boom = np.exp(-10 * np.arange(burst_length) / self.sample_rate)
                        mini_boom *= np.sin(2 * np.pi * boom_freq * 1.5 * np.arange(burst_length) / self.sample_rate)
                        boom[burst_start:burst_start + burst_length] += 0.5 * mini_boom
                
                explosion_sound = boom + crack + rumble
                
                # Pad/trim to standard duration
                if len(explosion_sound) > int(self.sample_rate * self.duration):
                    explosion_sound = explosion_sound[:int(self.sample_rate * self.duration)]
                else:
                    explosion_sound = np.pad(explosion_sound, (0, int(self.sample_rate * self.duration) - len(explosion_sound)))
                
                explosion_sound = explosion_sound / (np.max(np.abs(explosion_sound)) + 1e-8) * 0.9
                
                filename = combat_dir / "explosions" / f"{explosion['name']}_{i:03d}.wav"
                sf.write(str(filename), explosion_sound, self.sample_rate)
                sounds_created += 1
        
        # MASSIVE combat vehicle generation (300 samples)
        print("   Generating 300+ combat vehicle variations...")
        vehicle_types = [
            {'name': 'tank', 'engine_freq': [40, 80], 'weapon_prob': 0.7},
            {'name': 'apc', 'engine_freq': [50, 100], 'weapon_prob': 0.6},
            {'name': 'ifv', 'engine_freq': [45, 90], 'weapon_prob': 0.8},
            {'name': 'humvee', 'engine_freq': [60, 120], 'weapon_prob': 0.5},
            {'name': 'bradley', 'engine_freq': [35, 75], 'weapon_prob': 0.9},
            {'name': 'truck', 'engine_freq': [55, 110], 'weapon_prob': 0.3},
        ]
        
        for vehicle in vehicle_types:
            for i in range(50):  # 50 variants per vehicle
                t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
                
                # Engine characteristics
                base_freq = random.uniform(vehicle['engine_freq'][0], vehicle['engine_freq'][1])
                engine = 0.6 * np.sin(2 * np.pi * base_freq * t)
                engine += 0.4 * np.sin(2 * np.pi * base_freq * 2 * t)
                engine += 0.3 * np.sin(2 * np.pi * base_freq * 3 * t)
                
                # Add engine modulation (RPM changes)
                rpm_variation = 1 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
                engine = engine * rpm_variation
                
                # Track/wheel noise
                if vehicle['name'] in ['tank', 'apc', 'ifv', 'bradley']:
                    track_noise = 0.3 * np.random.normal(0, 1, len(t))
                    track_noise *= np.sin(2 * np.pi * 20 * t)  # Track frequency
                else:
                    track_noise = 0.2 * np.random.normal(0, 1, len(t))  # Road noise
                
                # Add weapon fire if combat
                if random.random() < vehicle['weapon_prob']:
                    num_bursts = random.randint(1, 4)
                    for burst in range(num_bursts):
                        burst_start = random.randint(int(0.2 * self.sample_rate), int(2.5 * self.sample_rate))
                        burst_length = random.randint(200, 1000)
                        
                        if burst_start + burst_length < len(engine):
                            weapon_burst = np.exp(-30 * np.arange(burst_length) / self.sample_rate)
                            weapon_freq = random.uniform(300, 600)
                            weapon_burst *= np.sin(2 * np.pi * weapon_freq * np.arange(burst_length) / self.sample_rate)
                            engine[burst_start:burst_start + burst_length] += 0.8 * weapon_burst
                
                vehicle_sound = engine + track_noise
                vehicle_sound = vehicle_sound / (np.max(np.abs(vehicle_sound)) + 1e-8) * 0.8
                
                filename = combat_dir / "combat_vehicles" / f"{vehicle['name']}_{i:03d}.wav"
                sf.write(str(filename), vehicle_sound, self.sample_rate)
                sounds_created += 1
        
        # MASSIVE combat aircraft generation (200 samples)
        print("   Generating 200+ combat aircraft variations...")
        aircraft_types = [
            {'name': 'fighter_jet', 'engine_freq': [300, 600], 'weapon_prob': 0.8},
            {'name': 'attack_helo', 'engine_freq': [100, 200], 'weapon_prob': 0.9},
            {'name': 'transport_helo', 'engine_freq': [80, 160], 'weapon_prob': 0.3},
            {'name': 'drone', 'engine_freq': [200, 400], 'weapon_prob': 0.7},
            {'name': 'bomber', 'engine_freq': [150, 300], 'weapon_prob': 0.6},
        ]
        
        for aircraft in aircraft_types:
            for i in range(40):  # 40 variants per aircraft
                t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
                
                # Engine characteristics
                base_freq = random.uniform(aircraft['engine_freq'][0], aircraft['engine_freq'][1])
                engine = 0.8 * np.sin(2 * np.pi * base_freq * t)
                engine += 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
                engine += 0.3 * np.sin(2 * np.pi * base_freq * 1.5 * t)
                
                # Doppler effect (flyby)
                doppler_factor = 1 + 0.15 * np.sin(2 * np.pi * 0.3 * t)
                engine = engine * doppler_factor
                
                # Rotor blade effects for helicopters
                if 'helo' in aircraft['name']:
                    rotor_freq = random.uniform(8, 15)  # Blade passage frequency
                    rotor_sound = 0.4 * np.sin(2 * np.pi * rotor_freq * t)
                    engine += rotor_sound
                
                # Wind/aerodynamic noise
                wind_noise = 0.3 * np.random.normal(0, 1, len(t))
                wind_filter = np.exp(-0.5 * t)  # High frequency content fades
                wind_noise *= wind_filter
                
                # Add weapon systems
                if random.random() < aircraft['weapon_prob']:
                    weapon_types = ['cannon', 'missile', 'rockets']
                    weapon_type = random.choice(weapon_types)
                    
                    if weapon_type == 'cannon':
                        # Rapid fire cannon
                        burst_start = random.randint(int(0.5 * self.sample_rate), int(2 * self.sample_rate))
                        burst_duration = random.randint(500, 1500)
                        for shot in range(0, burst_duration, 50):  # 20 rounds/sec
                            if burst_start + shot < len(engine) - 25:
                                shot_sound = np.exp(-100 * np.arange(25) / self.sample_rate)
                                shot_sound *= np.sin(2 * np.pi * 800 * np.arange(25) / self.sample_rate)
                                engine[burst_start + shot:burst_start + shot + 25] += 0.6 * shot_sound
                    
                    elif weapon_type == 'missile':
                        # Missile launch
                        launch_time = random.randint(int(0.3 * self.sample_rate), int(2 * self.sample_rate))
                        launch_duration = int(0.5 * self.sample_rate)
                        if launch_time + launch_duration < len(engine):
                            missile_sound = np.exp(-2 * np.arange(launch_duration) / self.sample_rate)
                            missile_sound *= np.sin(2 * np.pi * 200 * np.arange(launch_duration) / self.sample_rate)
                            engine[launch_time:launch_time + launch_duration] += 0.7 * missile_sound
                
                aircraft_sound = engine + wind_noise
                aircraft_sound = aircraft_sound / (np.max(np.abs(aircraft_sound)) + 1e-8) * 0.8
                
                filename = combat_dir / "combat_aircraft" / f"{aircraft['name']}_{i:03d}.wav"
                sf.write(str(filename), aircraft_sound, self.sample_rate)
                sounds_created += 1
        
        # Mixed combat scenarios (200 samples)
        print("   Generating 200+ mixed combat scenarios...")
        for i in range(200):
            t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
            
            # Base background
            background = 0.1 * np.random.normal(0, 1, len(t))
            
            # Add random combat elements
            num_elements = random.randint(2, 4)
            for element in range(num_elements):
                element_type = random.choice(['weapon', 'explosion', 'vehicle', 'aircraft'])
                element_start = random.randint(0, int(2 * self.sample_rate))
                
                if element_type == 'weapon':
                    duration = random.randint(100, 500)
                    if element_start + duration < len(background):
                        weapon = np.exp(-50 * np.arange(duration) / self.sample_rate)
                        weapon *= np.sin(2 * np.pi * random.uniform(200, 800) * np.arange(duration) / self.sample_rate)
                        background[element_start:element_start + duration] += 0.5 * weapon
                
                elif element_type == 'explosion':
                    duration = random.randint(500, 1500)
                    if element_start + duration < len(background):
                        explosion = np.exp(-3 * np.arange(duration) / self.sample_rate)
                        explosion *= np.sin(2 * np.pi * random.uniform(30, 80) * np.arange(duration) / self.sample_rate)
                        background[element_start:element_start + duration] += 0.7 * explosion
            
            mixed_sound = background / (np.max(np.abs(background)) + 1e-8) * 0.8
            
            filename = combat_dir / "mixed_combat" / f"mixed_combat_{i:03d}.wav"
            sf.write(str(filename), mixed_sound, self.sample_rate)
            sounds_created += 1
        
        print(f"✅ Generated {sounds_created} massive combat sound variations!")
        return combat_dir
    
    def create_massive_enhanced_dataset(self, combat_dir):
        """Create massive enhanced dataset"""
        print("🚀 Creating MASSIVE enhanced dataset...")
        
        # Source and destination
        source_dir = Path("expanded_sait01_dataset")
        dest_dir = Path("massive_enhanced_dataset")
        
        if not source_dir.exists():
            print("❌ Source dataset not found")
            return False
        
        # Create destination
        dest_dir.mkdir(exist_ok=True)
        
        # Copy all existing data first
        for class_name in ['background', 'vehicle', 'aircraft']:
            source_class_dir = source_dir / class_name
            dest_class_dir = dest_dir / class_name
            dest_class_dir.mkdir(exist_ok=True)
            
            if source_class_dir.exists():
                for audio_file in source_class_dir.glob("*.wav"):
                    shutil.copy2(audio_file, dest_class_dir)
        
        # Add MASSIVE combat sound integration
        combat_distribution = {
            'background': [
                (combat_dir / "explosions", 400),      # All explosions
                (combat_dir / "weapons", 300),         # Distant weapons
                (combat_dir / "mixed_combat", 200),    # Mixed scenarios
            ],
            'vehicle': [
                (combat_dir / "combat_vehicles", 300), # All combat vehicles
                (combat_dir / "weapons", 200),         # Vehicle weapons
                (combat_dir / "mixed_combat", 100),    # Vehicle in combat
            ],
            'aircraft': [
                (combat_dir / "combat_aircraft", 200), # All combat aircraft
                (combat_dir / "weapons", 100),         # Aircraft weapons
                (combat_dir / "mixed_combat", 100),    # Aircraft in combat
            ]
        }
        
        for class_name, sound_sources in combat_distribution.items():
            dest_class_dir = dest_dir / class_name
            
            for sound_dir, max_files in sound_sources:
                if sound_dir.exists():
                    sound_files = list(sound_dir.glob("*.wav"))[:max_files]
                    print(f"   Adding {len(sound_files)} {sound_dir.name} to {class_name}...")
                    
                    for i, sound_file in enumerate(sound_files):
                        dest_file = dest_class_dir / f"massive_{class_name}_{sound_dir.name}_{i:04d}.wav"
                        shutil.copy2(sound_file, dest_file)
        
        # Count final massive dataset
        total_counts = {}
        total_samples = 0
        for class_name in ['background', 'vehicle', 'aircraft']:
            class_dir = dest_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.wav")))
                total_counts[class_name] = count
                total_samples += count
        
        print(f"✅ MASSIVE enhanced dataset created:")
        for class_name, count in total_counts.items():
            print(f"   {class_name}: {count:,} samples")
        print(f"📊 Total samples: {total_samples:,}")
        
        return True

def main():
    print("⚔️  MASSIVE BATTLEFIELD DATASET EXPANSION")
    print("=" * 70)
    print("🎯 Creating massive dataset for 95%+ accuracy")
    print("🚀 Target: 10,000+ combat-enhanced samples")
    print("=" * 70)
    
    creator = MassiveBattlefieldDatasetCreator()
    
    # Generate massive combat sound library
    combat_dir = creator.create_massive_combat_sounds()
    
    # Create massive enhanced dataset
    success = creator.create_massive_enhanced_dataset(combat_dir)
    
    if success:
        print(f"\n🏆 MASSIVE DATASET CREATION COMPLETE!")
        print("=" * 70)
        print("📊 Dataset ready: massive_enhanced_dataset/")
        print("🎯 Ready for 95%+ accuracy training!")
        print("⚡ Next: Update training script to use massive dataset")
    else:
        print("❌ Massive dataset creation failed")

if __name__ == "__main__":
    main()