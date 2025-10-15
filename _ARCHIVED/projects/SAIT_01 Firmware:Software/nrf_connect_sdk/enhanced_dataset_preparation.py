#!/usr/bin/env python3
"""
Enhanced Dataset Preparation for False Positive Rejection
Creates comprehensive negative samples to prevent natural noise triggering
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
import time

# Add current directory to path
sys.path.append('.')
from sait01_model_architecture import SaitAudioPreprocessor

class EnhancedDatasetPreparer:
    """Prepare enhanced dataset with comprehensive negative examples"""
    
    def __init__(self, data_dir="edth-copenhagen-drone-acoustics/data/raw"):
        self.data_dir = data_dir
        self.preprocessor = SaitAudioPreprocessor()
        self.sample_rate = 16000
        self.duration = 1.0  # 1 second samples
        
        print("🛡️  Enhanced Dataset Preparation")
        print("Target: Robust false positive rejection")
        
    def generate_natural_noise_samples(self, num_samples=300):
        """Generate diverse natural noise samples that should NOT trigger detection"""
        
        output_dir = Path(self.data_dir) / 'train' / 'background'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🌿 Generating {num_samples} natural noise samples...")
        
        # Natural noise types that should be rejected
        noise_types = [
            ('wind', self._generate_wind_noise),
            ('rain', self._generate_rain_noise),
            ('traffic_distant', self._generate_distant_traffic),
            ('birds', self._generate_bird_sounds),
            ('leaves', self._generate_rustling_leaves),
            ('water', self._generate_water_sounds),
            ('urban_ambient', self._generate_urban_ambient),
            ('mechanical_hum', self._generate_mechanical_hum),
            ('voices_distant', self._generate_distant_voices),
            ('construction_distant', self._generate_distant_construction)
        ]
        
        samples_per_type = num_samples // len(noise_types)
        generated_count = 0
        
        for noise_name, generator_func in noise_types:
            print(f"  Generating {samples_per_type} {noise_name} samples...")
            
            for i in range(samples_per_type):
                try:
                    # Generate audio sample
                    audio = generator_func()
                    
                    # Apply random environmental effects
                    audio = self._apply_environmental_effects(audio)
                    
                    # Normalize
                    audio = librosa.util.normalize(audio)
                    
                    # Save
                    filename = f"generated_{noise_name}_{i:03d}.wav"
                    filepath = output_dir / filename
                    sf.write(filepath, audio, self.sample_rate)
                    
                    generated_count += 1
                    
                except Exception as e:
                    print(f"    Error generating {noise_name}_{i}: {e}")
                    continue
        
        print(f"✅ Generated {generated_count} natural noise samples")
        return generated_count
    
    def _generate_wind_noise(self):
        """Generate wind noise that should NOT trigger aircraft detection"""
        duration = self.duration
        samples = int(duration * self.sample_rate)
        
        # Base wind noise (pink noise filtered)
        noise = np.random.randn(samples)
        
        # Apply low-pass filter for wind characteristics
        from scipy import signal
        b, a = signal.butter(4, 800, fs=self.sample_rate, btype='low')
        wind_base = signal.filtfilt(b, a, noise)
        
        # Add gusty variations
        gust_freq = random.uniform(0.1, 0.5)
        t = np.linspace(0, duration, samples)
        gust_envelope = 1 + 0.3 * np.sin(2 * np.pi * gust_freq * t)
        
        wind_noise = wind_base * gust_envelope
        
        # Scale to reasonable amplitude
        return wind_noise * random.uniform(0.1, 0.4)
    
    def _generate_rain_noise(self):
        """Generate rain noise"""
        samples = int(self.duration * self.sample_rate)
        
        # High frequency white noise for raindrops
        rain_base = np.random.randn(samples) * 0.1
        
        # Apply high-pass filter
        from scipy import signal
        b, a = signal.butter(2, 2000, fs=self.sample_rate, btype='high')
        rain_noise = signal.filtfilt(b, a, rain_base)
        
        # Add random intensity variations
        intensity = random.uniform(0.05, 0.3)
        return rain_noise * intensity
    
    def _generate_distant_traffic(self):
        """Generate distant traffic noise that might confuse with aircraft"""
        samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, samples)
        
        # Multiple vehicle components
        traffic = np.zeros(samples)
        
        # Low frequency rumble (trucks)
        rumble_freq = random.uniform(40, 80)
        traffic += 0.3 * np.sin(2 * np.pi * rumble_freq * t)
        
        # Mid frequency engine noise
        engine_freq = random.uniform(100, 200)
        traffic += 0.2 * np.sin(2 * np.pi * engine_freq * t)
        
        # Add some high frequency tire noise
        tire_noise = np.random.randn(samples) * 0.05
        from scipy import signal
        b, a = signal.butter(2, [1000, 4000], fs=self.sample_rate, btype='band')
        tire_filtered = signal.filtfilt(b, a, tire_noise)
        traffic += tire_filtered
        
        # Distance attenuation (low-pass filter)
        b, a = signal.butter(3, 1500, fs=self.sample_rate, btype='low')
        distant_traffic = signal.filtfilt(b, a, traffic)
        
        return distant_traffic * random.uniform(0.1, 0.25)
    
    def _generate_bird_sounds(self):
        """Generate bird sounds that should be rejected"""
        samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, samples)
        
        bird_audio = np.zeros(samples)
        
        # Generate multiple bird calls
        num_birds = random.randint(1, 4)
        
        for _ in range(num_birds):
            # Random bird call frequency
            bird_freq = random.uniform(800, 4000)
            
            # Call duration and timing
            call_duration = random.uniform(0.1, 0.3)
            call_start = random.uniform(0, self.duration - call_duration)
            
            call_samples = int(call_duration * self.sample_rate)
            start_idx = int(call_start * self.sample_rate)
            end_idx = start_idx + call_samples
            
            if end_idx < samples:
                # Generate chirp-like sound
                call_t = np.linspace(0, call_duration, call_samples)
                freq_sweep = bird_freq + random.uniform(-200, 200) * call_t
                
                call = np.sin(2 * np.pi * freq_sweep * call_t)
                
                # Apply envelope
                envelope = np.exp(-3 * call_t)
                call *= envelope
                
                bird_audio[start_idx:end_idx] += call * random.uniform(0.1, 0.3)
        
        return bird_audio
    
    def _generate_rustling_leaves(self):
        """Generate leaf rustling sounds"""
        samples = int(self.duration * self.sample_rate)
        
        # High frequency noise filtered to simulate leaf movement
        noise = np.random.randn(samples)
        
        from scipy import signal
        # Band-pass filter for leaf rustling characteristics
        b, a = signal.butter(2, [500, 8000], fs=self.sample_rate, btype='band')
        rustling = signal.filtfilt(b, a, noise)
        
        # Add amplitude modulation for movement
        t = np.linspace(0, self.duration, samples)
        modulation = 1 + 0.5 * np.sin(2 * np.pi * random.uniform(2, 8) * t)
        
        return rustling * modulation * random.uniform(0.05, 0.2)
    
    def _generate_water_sounds(self):
        """Generate water sounds (streams, fountains)"""
        samples = int(self.duration * self.sample_rate)
        
        # Pink noise base for water
        noise = np.random.randn(samples)
        
        # Apply filter to simulate water characteristics
        from scipy import signal
        b, a = signal.butter(3, [200, 6000], fs=self.sample_rate, btype='band')
        water_base = signal.filtfilt(b, a, noise)
        
        # Add bubbling effect
        t = np.linspace(0, self.duration, samples)
        bubble_freq = random.uniform(5, 15)
        bubbling = 1 + 0.2 * np.sin(2 * np.pi * bubble_freq * t)
        
        return water_base * bubbling * random.uniform(0.1, 0.3)
    
    def _generate_urban_ambient(self):
        """Generate general urban ambient that should be ignored"""
        samples = int(self.duration * self.sample_rate)
        
        # Combination of various urban sounds
        ambient = np.zeros(samples)
        
        # Air conditioning hum
        ac_freq = random.uniform(50, 120)
        t = np.linspace(0, self.duration, samples)
        ambient += 0.1 * np.sin(2 * np.pi * ac_freq * t)
        
        # General low frequency urban noise
        urban_noise = np.random.randn(samples) * 0.05
        from scipy import signal
        b, a = signal.butter(2, 500, fs=self.sample_rate, btype='low')
        ambient += signal.filtfilt(b, a, urban_noise)
        
        return ambient * random.uniform(0.1, 0.25)
    
    def _generate_mechanical_hum(self):
        """Generate mechanical hum that might be confused with aircraft"""
        samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, samples)
        
        # Main hum frequency
        hum_freq = random.uniform(100, 300)
        hum = np.sin(2 * np.pi * hum_freq * t)
        
        # Add harmonics
        for harmonic in [2, 3, 4]:
            hum += 0.3 * np.sin(2 * np.pi * hum_freq * harmonic * t)
        
        # Add slight frequency variation
        freq_variation = 1 + 0.02 * np.sin(2 * np.pi * 0.5 * t)
        hum *= freq_variation
        
        return hum * random.uniform(0.1, 0.3)
    
    def _generate_distant_voices(self):
        """Generate distant human voices that should be rejected"""
        samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, samples)
        
        voices = np.zeros(samples)
        
        # Simulate distant conversation
        num_speakers = random.randint(1, 3)
        
        for speaker in range(num_speakers):
            # Voice fundamental frequency
            voice_freq = random.uniform(80, 250)  # Human voice range
            
            # Speaking segments
            num_segments = random.randint(1, 4)
            
            for segment in range(num_segments):
                segment_duration = random.uniform(0.2, 0.5)
                segment_start = random.uniform(0, self.duration - segment_duration)
                
                seg_samples = int(segment_duration * self.sample_rate)
                start_idx = int(segment_start * self.sample_rate)
                end_idx = start_idx + seg_samples
                
                if end_idx < samples:
                    seg_t = np.linspace(0, segment_duration, seg_samples)
                    
                    # Voice with formants
                    voice_seg = np.sin(2 * np.pi * voice_freq * seg_t)
                    voice_seg += 0.3 * np.sin(2 * np.pi * voice_freq * 2 * seg_t)
                    voice_seg += 0.2 * np.sin(2 * np.pi * voice_freq * 3 * seg_t)
                    
                    # Distance filtering (low-pass)
                    from scipy import signal
                    b, a = signal.butter(2, 800, fs=self.sample_rate, btype='low')
                    voice_seg = signal.filtfilt(b, a, voice_seg)
                    
                    voices[start_idx:end_idx] += voice_seg * random.uniform(0.05, 0.15)
        
        return voices
    
    def _generate_distant_construction(self):
        """Generate distant construction noise"""
        samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, samples)
        
        construction = np.zeros(samples)
        
        # Intermittent impacts (hammering, etc.)
        num_impacts = random.randint(2, 8)
        
        for impact in range(num_impacts):
            impact_time = random.uniform(0, self.duration)
            impact_idx = int(impact_time * self.sample_rate)
            
            # Impact duration
            impact_duration = random.uniform(0.05, 0.2)
            impact_samples = int(impact_duration * self.sample_rate)
            
            if impact_idx + impact_samples < samples:
                # Generate impact sound
                impact_t = np.linspace(0, impact_duration, impact_samples)
                
                # Multiple frequency components
                impact_sound = np.zeros(impact_samples)
                for freq in [100, 200, 400, 800]:
                    impact_sound += np.sin(2 * np.pi * freq * impact_t) * np.exp(-10 * impact_t)
                
                construction[impact_idx:impact_idx + impact_samples] += impact_sound * random.uniform(0.1, 0.2)
        
        # Distance attenuation
        from scipy import signal
        b, a = signal.butter(3, 1000, fs=self.sample_rate, btype='low')
        construction = signal.filtfilt(b, a, construction)
        
        return construction
    
    def _apply_environmental_effects(self, audio):
        """Apply random environmental effects to make samples more realistic"""
        
        # Random amplitude scaling
        amplitude_factor = random.uniform(0.3, 1.0)
        audio *= amplitude_factor
        
        # Random background noise
        if random.random() < 0.7:
            noise_level = random.uniform(0.01, 0.05)
            background_noise = np.random.randn(len(audio)) * noise_level
            audio += background_noise
        
        # Random filtering (simulate distance/obstruction)
        if random.random() < 0.5:
            from scipy import signal
            cutoff = random.uniform(2000, 8000)
            b, a = signal.butter(2, cutoff, fs=self.sample_rate, btype='low')
            audio = signal.filtfilt(b, a, audio)
        
        # Random compression (simulate microphone characteristics)
        if random.random() < 0.3:
            compression_ratio = random.uniform(0.7, 1.3)
            audio = np.sign(audio) * np.power(np.abs(audio), compression_ratio)
        
        return audio
    
    def augment_existing_negatives(self, multiplier=3):
        """Augment existing background samples with variations"""
        
        background_dir = Path(self.data_dir) / 'train' / 'background'
        if not background_dir.exists():
            print("❌ Background directory not found")
            return 0
        
        existing_files = list(background_dir.glob("*.wav"))
        print(f"🔄 Augmenting {len(existing_files)} existing background samples...")
        
        augmented_count = 0
        
        for audio_file in existing_files:
            try:
                # Load original audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                
                # Create multiple augmented versions
                for i in range(multiplier):
                    augmented_audio = audio.copy()
                    
                    # Apply various augmentations
                    augmented_audio = self._apply_environmental_effects(augmented_audio)
                    
                    # Additional aggressive augmentations for negatives
                    augmented_audio = self._apply_negative_augmentations(augmented_audio)
                    
                    # Save augmented version
                    base_name = audio_file.stem
                    aug_filename = f"{base_name}_aug_{i:02d}.wav"
                    aug_filepath = background_dir / aug_filename
                    
                    sf.write(aug_filepath, augmented_audio, self.sample_rate)
                    augmented_count += 1
                    
            except Exception as e:
                print(f"  Error augmenting {audio_file.name}: {e}")
                continue
        
        print(f"✅ Created {augmented_count} augmented background samples")
        return augmented_count
    
    def _apply_negative_augmentations(self, audio):
        """Apply aggressive augmentations specifically for negative samples"""
        
        # Mix with various noise types
        if random.random() < 0.4:
            noise_type = random.choice(['wind', 'rain', 'traffic', 'mechanical'])
            
            if noise_type == 'wind':
                noise = self._generate_wind_noise()
            elif noise_type == 'rain':
                noise = self._generate_rain_noise()
            elif noise_type == 'traffic':
                noise = self._generate_distant_traffic()
            else:
                noise = self._generate_mechanical_hum()
            
            # Ensure same length
            if len(noise) != len(audio):
                noise = noise[:len(audio)]
            
            mix_ratio = random.uniform(0.1, 0.4)
            audio = (1 - mix_ratio) * audio + mix_ratio * noise
        
        # Dynamic range compression
        if random.random() < 0.3:
            threshold = random.uniform(0.1, 0.3)
            ratio = random.uniform(2, 6)
            
            # Simple compression
            compressed = np.where(np.abs(audio) > threshold,
                                threshold + (np.abs(audio) - threshold) / ratio,
                                np.abs(audio)) * np.sign(audio)
            audio = compressed
        
        # Random EQ (frequency response modification)
        if random.random() < 0.5:
            from scipy import signal
            
            # Random boost/cut in different frequency bands
            for _ in range(random.randint(1, 3)):
                center_freq = random.uniform(100, 4000)
                q_factor = random.uniform(0.5, 3.0)
                gain_db = random.uniform(-6, 6)
                
                # Create parametric EQ filter
                gain_linear = 10 ** (gain_db / 20)
                
                # Simple peak filter approximation
                b, a = signal.iirpeak(center_freq, q_factor, fs=self.sample_rate)
                
                if gain_db > 0:
                    # Boost
                    audio = signal.filtfilt(b * gain_linear, a, audio)
                else:
                    # Cut (inverse filter approximation)
                    audio = signal.filtfilt(b / gain_linear, a, audio)
        
        return audio
    
    def prepare_enhanced_dataset(self):
        """Complete enhanced dataset preparation"""
        print("\n🛡️  ENHANCED DATASET PREPARATION")
        print("=" * 50)
        print("Objective: Create comprehensive negative samples")
        print("Goal: <5% false positive rate on natural sounds")
        print("=" * 50)
        
        # Generate natural noise samples
        generated_negatives = self.generate_natural_noise_samples(500)
        
        # Augment existing background samples
        augmented_negatives = self.augment_existing_negatives(multiplier=4)
        
        # Summary
        total_negatives = generated_negatives + augmented_negatives
        
        print(f"\n✅ ENHANCED DATASET PREPARATION COMPLETE")
        print(f"📊 Summary:")
        print(f"  Generated natural noise samples: {generated_negatives}")
        print(f"  Augmented existing samples: {augmented_negatives}")
        print(f"  Total new negative samples: {total_negatives}")
        
        # Validate dataset balance
        self._validate_dataset_balance()
        
        return total_negatives
    
    def _validate_dataset_balance(self):
        """Validate the dataset balance after enhancement"""
        
        data_path = Path(self.data_dir) / 'train'
        
        class_counts = {}
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                wav_files = list(class_dir.glob("*.wav"))
                class_counts[class_dir.name] = len(wav_files)
        
        print(f"\n📊 Enhanced Dataset Balance:")
        total_samples = sum(class_counts.values())
        
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Check if background class is well represented
        if 'background' in class_counts:
            bg_ratio = class_counts['background'] / total_samples
            if bg_ratio >= 0.4:  # At least 40% background
                print("✅ Good negative sample representation")
            else:
                print("⚠️  Consider adding more negative samples")
        
        return class_counts

def main():
    preparer = EnhancedDatasetPreparer()
    preparer.prepare_enhanced_dataset()
    print("\n🚀 Ready for advanced training with enhanced false positive rejection!")

if __name__ == "__main__":
    main()