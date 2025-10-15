#!/usr/bin/env python3
"""
Realistic Audio Processing Pipeline for SAIT_01
Generate and process battlefield audio for QADT-R training
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.signal import butter, filtfilt, spectrogram
import json
import os
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

class BattlefieldAudioGenerator:
    """Generate realistic battlefield audio based on threat taxonomy"""
    
    def __init__(self, sample_rate: int = 16000, duration: float = 2.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
        # Load threat taxonomy
        with open('27_class_threat_taxonomy.json', 'r') as f:
            self.taxonomy = json.load(f)
        
        print(f"🎵 Audio Generator: {sample_rate}Hz, {duration}s duration")
    
    def generate_engine_sound(self, rpm: float = 1500, complexity: float = 1.0) -> np.ndarray:
        """Generate realistic engine sound"""
        t = np.linspace(0, self.duration, self.num_samples)
        
        # Base engine frequency
        base_freq = rpm / 60.0  # Convert RPM to Hz
        
        # Engine harmonics
        sound = np.zeros_like(t)
        for harmonic in range(1, 8):
            amplitude = 1.0 / (harmonic ** 0.7) * complexity
            freq = base_freq * harmonic
            phase = np.random.uniform(0, 2*np.pi)
            sound += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Engine irregularities (combustion cycles)
        for i in range(int(base_freq * self.duration * 4)):  # 4 cycles per revolution
            pulse_time = i / (base_freq * 4)
            if pulse_time < self.duration:
                pulse_idx = int(pulse_time * self.sample_rate)
                if pulse_idx < len(sound):
                    sound[pulse_idx] += 0.3 * complexity
        
        # Add engine vibration (low frequency modulation)
        vibration_freq = base_freq / 10
        vibration = 1 + 0.1 * np.sin(2 * np.pi * vibration_freq * t)
        sound *= vibration
        
        return sound
    
    def generate_explosion_sound(self, intensity: float = 1.0) -> np.ndarray:
        """Generate realistic explosion sound"""
        t = np.linspace(0, self.duration, self.num_samples)
        
        # Initial blast (high energy, broad spectrum)
        blast_duration = 0.1
        blast_samples = int(blast_duration * self.sample_rate)
        
        sound = np.zeros_like(t)
        
        # Sharp initial crack
        crack = np.random.normal(0, intensity, blast_samples)
        crack *= np.exp(-t[:blast_samples] * 50)  # Rapid decay
        sound[:blast_samples] += crack
        
        # Low frequency rumble
        rumble_freq = np.random.uniform(20, 80)
        rumble = intensity * 0.5 * np.sin(2 * np.pi * rumble_freq * t)
        rumble *= np.exp(-t * 2)  # Exponential decay
        sound += rumble
        
        # High frequency debris
        debris_freqs = np.random.uniform(1000, 8000, 20)
        for freq in debris_freqs:
            debris_amp = intensity * np.random.uniform(0.1, 0.3)
            debris_decay = np.random.uniform(5, 15)
            debris = debris_amp * np.sin(2 * np.pi * freq * t)
            debris *= np.exp(-t * debris_decay)
            sound += debris
        
        return sound
    
    def generate_gunfire_sound(self, weapon_type: str = "rifle") -> np.ndarray:
        """Generate realistic gunfire sound"""
        t = np.linspace(0, self.duration, self.num_samples)
        
        if weapon_type == "rifle":
            # Sharp crack followed by echo
            crack_duration = 0.01
            crack_samples = int(crack_duration * self.sample_rate)
            
            sound = np.zeros_like(t)
            
            # Muzzle blast (sharp, high frequency)
            crack = np.random.normal(0, 1.0, crack_samples)
            crack *= np.exp(-t[:crack_samples] * 100)
            sound[:crack_samples] += crack
            
            # Sonic boom (supersonic bullet)
            boom_freq = np.random.uniform(200, 500)
            boom = 0.7 * np.sin(2 * np.pi * boom_freq * t)
            boom *= np.exp(-t * 20)
            sound += boom
            
        elif weapon_type == "machine_gun":
            # Rapid succession of shots
            shots_per_second = 10
            shot_interval = 1.0 / shots_per_second
            
            sound = np.zeros_like(t)
            for shot in range(int(self.duration * shots_per_second)):
                shot_start = shot * shot_interval
                if shot_start < self.duration:
                    shot_start_idx = int(shot_start * self.sample_rate)
                    shot_end_idx = min(shot_start_idx + int(0.05 * self.sample_rate), len(sound))
                    
                    # Individual shot
                    shot_t = t[shot_start_idx:shot_end_idx] - shot_start
                    shot_sound = np.random.normal(0, 0.8, len(shot_t))
                    shot_sound *= np.exp(-shot_t * 50)
                    sound[shot_start_idx:shot_end_idx] += shot_sound
        
        return sound
    
    def generate_aircraft_sound(self, aircraft_type: str = "helicopter") -> np.ndarray:
        """Generate realistic aircraft sound"""
        t = np.linspace(0, self.duration, self.num_samples)
        
        if aircraft_type == "helicopter":
            # Rotor blade frequency
            rotor_freq = np.random.uniform(15, 25)  # Main rotor
            tail_rotor_freq = rotor_freq * 5.5  # Tail rotor
            
            sound = np.zeros_like(t)
            
            # Main rotor (low frequency, periodic)
            main_rotor = 0.8 * np.sin(2 * np.pi * rotor_freq * t)
            # Add blade slap
            for blade in range(4):  # 4 blades
                blade_freq = rotor_freq * 4
                blade_phase = blade * np.pi / 2
                blade_sound = 0.3 * np.sin(2 * np.pi * blade_freq * t + blade_phase)
                main_rotor += blade_sound
            
            sound += main_rotor
            
            # Tail rotor (high frequency)
            tail_rotor = 0.4 * np.sin(2 * np.pi * tail_rotor_freq * t)
            sound += tail_rotor
            
            # Engine whine
            engine_freq = np.random.uniform(800, 1200)
            engine = 0.3 * np.sin(2 * np.pi * engine_freq * t)
            sound += engine
            
        elif aircraft_type == "jet":
            # Turbine whine
            turbine_freq = np.random.uniform(1000, 3000)
            turbine = 0.9 * np.sin(2 * np.pi * turbine_freq * t)
            
            # Jet exhaust (broadband noise)
            exhaust = np.random.normal(0, 0.5, len(t))
            
            # Doppler effect (frequency shift)
            doppler_shift = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Approaching/receding
            turbine *= doppler_shift
            
            sound = turbine + exhaust
        
        return sound
    
    def generate_environmental_sound(self, environment: str = "wind") -> np.ndarray:
        """Generate environmental background sounds"""
        t = np.linspace(0, self.duration, self.num_samples)
        
        if environment == "wind":
            # Low frequency wind noise
            wind = np.random.normal(0, 0.2, len(t))
            # Apply low-pass filter
            b, a = butter(3, 200, btype='low', fs=self.sample_rate)
            wind = filtfilt(b, a, wind)
            
            # Wind gusts
            gust_freq = 0.3
            gust_modulation = 1 + 0.5 * np.sin(2 * np.pi * gust_freq * t)
            wind *= gust_modulation
            
            return wind
            
        elif environment == "rain":
            # High frequency random noise (raindrops)
            rain = np.random.normal(0, 0.1, len(t))
            # Apply high-pass filter
            b, a = butter(3, 1000, btype='high', fs=self.sample_rate)
            rain = filtfilt(b, a, rain)
            
            return rain
        
        return np.zeros_like(t)
    
    def generate_audio_for_class(self, class_name: str, threat_info: Dict) -> np.ndarray:
        """Generate audio for specific threat class"""
        
        # Get class category
        category = threat_info.get('category', 'UNKNOWN')
        
        if class_name == 'BACKGROUND_QUIET':
            # Very low amplitude ambient noise
            return np.random.normal(0, 0.01, self.num_samples)
            
        elif class_name == 'ENVIRONMENTAL':
            # Mix wind and rain
            wind = self.generate_environmental_sound('wind')
            rain = self.generate_environmental_sound('rain')
            return 0.7 * wind + 0.3 * rain
            
        elif class_name in ['CIVILIAN_TRAFFIC', 'FRIENDLY_VEHICLE', 'TRUCK_MILITARY']:
            # Vehicle engines
            rpm = np.random.uniform(1000, 3000)
            complexity = 0.8 if 'MILITARY' in class_name else 0.6
            return self.generate_engine_sound(rpm, complexity)
            
        elif class_name in ['TANK_TRACKED', 'IFV_APC']:
            # Heavy tracked vehicles
            rpm = np.random.uniform(800, 1500)
            engine = self.generate_engine_sound(rpm, 1.2)
            # Add track noise
            track_freq = rpm / 60 * 2  # Track speed
            track_noise = 0.3 * np.random.normal(0, 1, self.num_samples)
            return engine + track_noise
            
        elif class_name in ['ATTACK_HELICOPTER', 'TRANSPORT_HELICOPTER']:
            return self.generate_aircraft_sound('helicopter')
            
        elif class_name in ['JET_FIGHTER', 'TRANSPORT_AIRCRAFT', 'RECON_AIRCRAFT']:
            return self.generate_aircraft_sound('jet')
            
        elif class_name in ['INCOMING_ARTILLERY', 'INCOMING_MISSILE', 'EXPLOSION_NEAR']:
            return self.generate_explosion_sound(1.0)
            
        elif class_name in ['SNIPER_FIRE', 'RPG_LAUNCH', 'MORTAR_LAUNCH']:
            weapon_type = 'rifle' if 'SNIPER' in class_name else 'rifle'
            return self.generate_gunfire_sound(weapon_type)
            
        elif class_name == 'ATTACK_DRONE':
            # Small propeller drone
            prop_freq = np.random.uniform(50, 150)
            t = np.linspace(0, self.duration, self.num_samples)
            drone_sound = 0.6 * np.sin(2 * np.pi * prop_freq * t)
            # Add motor whine
            motor_freq = prop_freq * 10
            motor_sound = 0.3 * np.sin(2 * np.pi * motor_freq * t)
            return drone_sound + motor_sound
            
        elif class_name == 'SURVEILLANCE_DRONE':
            # Quieter drone
            prop_freq = np.random.uniform(30, 80)
            t = np.linspace(0, self.duration, self.num_samples)
            return 0.4 * np.sin(2 * np.pi * prop_freq * t)
            
        else:
            # Default: mixed environmental + mechanical
            env = self.generate_environmental_sound('wind')
            mech = self.generate_engine_sound(1500, 0.5)
            return 0.5 * env + 0.3 * mech

class AudioSpectrogramProcessor:
    """Convert audio to spectrograms for neural network training"""
    
    def __init__(self, 
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 64,
                 sample_rate: int = 16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        
        print(f"🔊 Spectrogram Processor: {n_mels} mel bins, {n_fft} FFT")
    
    def audio_to_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mel-spectrogram"""
        
        # Compute STFT
        f, t, Zxx = spectrogram(audio, 
                               fs=self.sample_rate,
                               nperseg=self.n_fft,
                               noverlap=self.n_fft - self.hop_length)
        
        # Convert to power spectrogram
        power_spec = np.abs(Zxx) ** 2
        
        # Convert to mel scale
        mel_spec = self.linear_to_mel(power_spec, f)
        
        # Convert to log scale (dB)
        log_mel_spec = 10 * np.log10(mel_spec + 1e-10)
        
        # Normalize
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
        
        return log_mel_spec
    
    def linear_to_mel(self, power_spec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Convert linear frequency spectrogram to mel scale"""
        
        # Create mel filter bank
        mel_filters = self.create_mel_filter_bank(freqs)
        
        # Apply mel filters
        mel_spec = np.dot(mel_filters, power_spec)
        
        return mel_spec
    
    def create_mel_filter_bank(self, freqs: np.ndarray) -> np.ndarray:
        """Create mel filter bank"""
        
        # Mel scale conversion functions
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel scale points
        mel_min = hz_to_mel(freqs[0])
        mel_max = hz_to_mel(freqs[-1])
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Create filter bank
        filter_bank = np.zeros((self.n_mels, len(freqs)))
        
        for i in range(self.n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            for j, freq in enumerate(freqs):
                if left <= freq <= center:
                    filter_bank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filter_bank[i, j] = (right - freq) / (right - center)
        
        return filter_bank
    
    def pad_or_crop_spectrogram(self, spectrogram: np.ndarray, target_width: int = 64) -> np.ndarray:
        """Pad or crop spectrogram to target width"""
        
        current_width = spectrogram.shape[1]
        
        if current_width > target_width:
            # Crop from center
            start = (current_width - target_width) // 2
            return spectrogram[:, start:start + target_width]
        elif current_width < target_width:
            # Pad with zeros
            pad_width = target_width - current_width
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            return np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode='constant')
        else:
            return spectrogram

def create_realistic_audio_dataset(num_samples_per_class: int = 100):
    """Create realistic audio dataset for all 27 threat classes"""
    
    print("🎯 Creating Realistic Battlefield Audio Dataset")
    print("=" * 60)
    
    # Initialize components
    audio_gen = BattlefieldAudioGenerator(sample_rate=16000, duration=2.0)
    spec_processor = AudioSpectrogramProcessor(n_mels=64)
    
    # Load taxonomy
    with open('27_class_threat_taxonomy.json', 'r') as f:
        taxonomy = json.load(f)
    
    threat_classes = taxonomy['threat_classes']
    
    dataset = {
        'spectrograms': [],
        'labels': [],
        'class_names': [],
        'metadata': {
            'num_classes': len(threat_classes),
            'samples_per_class': num_samples_per_class,
            'total_samples': len(threat_classes) * num_samples_per_class,
            'spectrogram_shape': (64, 64),
            'sample_rate': 16000,
            'duration': 2.0
        }
    }
    
    class_id = 0
    for class_name, threat_info in threat_classes.items():
        print(f"🔊 Generating {class_name}: {num_samples_per_class} samples")
        
        class_spectrograms = []
        
        for sample_idx in range(num_samples_per_class):
            # Generate audio
            audio = audio_gen.generate_audio_for_class(class_name, threat_info)
            
            # Add realistic noise
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, len(audio))
            audio_noisy = audio + noise
            
            # Normalize
            audio_noisy = audio_noisy / (np.max(np.abs(audio_noisy)) + 1e-8)
            
            # Convert to spectrogram
            spectrogram = spec_processor.audio_to_spectrogram(audio_noisy)
            spectrogram = spec_processor.pad_or_crop_spectrogram(spectrogram, 64)
            
            class_spectrograms.append(spectrogram)
            dataset['labels'].append(class_id)
        
        dataset['spectrograms'].extend(class_spectrograms)
        dataset['class_names'].append(class_name)
        class_id += 1
    
    # Convert to tensors
    spectrograms_tensor = torch.stack([torch.FloatTensor(spec).unsqueeze(0) 
                                     for spec in dataset['spectrograms']])
    labels_tensor = torch.LongTensor(dataset['labels'])
    
    print(f"\n✅ Dataset created:")
    print(f"   📊 Shape: {spectrograms_tensor.shape}")
    print(f"   🏷️  Classes: {len(dataset['class_names'])}")
    print(f"   📈 Total samples: {len(spectrograms_tensor)}")
    
    # Save dataset
    torch.save({
        'spectrograms': spectrograms_tensor,
        'labels': labels_tensor,
        'class_names': dataset['class_names'],
        'metadata': dataset['metadata']
    }, 'realistic_battlefield_audio_dataset.pth')
    
    print(f"💾 Saved: realistic_battlefield_audio_dataset.pth")
    
    return spectrograms_tensor, labels_tensor, dataset['class_names']

if __name__ == "__main__":
    # Create the realistic audio dataset
    spectrograms, labels, class_names = create_realistic_audio_dataset(num_samples_per_class=150)
    
    print("\n🎉 Realistic Audio Processing Pipeline Complete!")
    print("📋 Next steps:")
    print("   1. Train QADT-R on realistic audio data")
    print("   2. Test robustness on actual battlefield-like spectrograms")
    print("   3. Validate performance meets real-world requirements")