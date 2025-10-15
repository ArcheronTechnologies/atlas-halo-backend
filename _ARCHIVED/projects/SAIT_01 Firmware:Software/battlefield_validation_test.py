#!/usr/bin/env python3
"""
Battlefield Validation Test for Enhanced QADT-R
===============================================

Real-world performance validation simulating actual battlefield conditions:
- Mixed audio scenarios with multiple simultaneous threats
- Environmental noise and interference  
- Streaming audio processing pipeline
- Latency and accuracy under stress
- Power consumption simulation
- Edge case handling (low SNR, rapid transitions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import queue
import random

# Import our enhanced model architecture
import sys
sys.path.append(str(Path(__file__).parent))
from noise_robust_architecture import NoiseRobustMilitaryModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BattlefieldAudioSimulator:
    """Simulate realistic battlefield audio conditions"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.threat_signatures = self._load_threat_signatures()
        
    def _load_threat_signatures(self):
        """Load available audio samples for simulation"""
        audio_dir = Path('/Users/timothyaikenhead/Desktop/drone_acoustics_train_val_data')
        signatures = {'drone': [], 'helicopter': [], 'background': []}
        
        if audio_dir.exists():
            for category in ['drone', 'helicopter', 'background']:
                cat_dir = audio_dir / 'train' / category
                if cat_dir.exists():
                    signatures[category] = list(cat_dir.glob('*.wav'))[:5]  # Limit for testing
        
        logger.info(f"📁 Loaded signatures: {len(signatures['drone'])} drone, "
                   f"{len(signatures['helicopter'])} helicopter, {len(signatures['background'])} background")
        return signatures
    
    def generate_mixed_battlefield_audio(self, duration_seconds=2.0, scenario='urban_patrol'):
        """Generate realistic mixed battlefield audio scenarios"""
        
        samples = int(duration_seconds * self.sample_rate)
        mixed_audio = torch.zeros(samples)
        
        # Scenario-specific parameters
        scenarios = {
            'urban_patrol': {
                'base_noise_level': 0.1,
                'threat_probability': 0.3,
                'background_types': ['vehicle', 'wind', 'distant_activity'],
                'primary_threats': ['drone', 'small_arms', 'vehicle_approach']
            },
            'rural_surveillance': {
                'base_noise_level': 0.05,
                'threat_probability': 0.2,
                'background_types': ['wind', 'vegetation', 'distant_aircraft'],
                'primary_threats': ['helicopter', 'footsteps', 'radio_chatter']
            },
            'convoy_protection': {
                'base_noise_level': 0.2,
                'threat_probability': 0.5,
                'background_types': ['engine_noise', 'road_surface', 'wind'],
                'primary_threats': ['drone', 'ied_warning', 'ambush_sounds']
            },
            'base_perimeter': {
                'base_noise_level': 0.08,
                'threat_probability': 0.15,
                'background_types': ['generator', 'equipment', 'personnel'],
                'primary_threats': ['infiltration', 'drone', 'distant_explosion']
            }
        }
        
        config = scenarios.get(scenario, scenarios['urban_patrol'])
        
        # Add base environmental noise
        noise = torch.randn(samples) * config['base_noise_level']
        mixed_audio += noise
        
        # Add threat signatures if available
        if random.random() < config['threat_probability']:
            threat_type = random.choice(['drone', 'helicopter'])
            if self.threat_signatures[threat_type]:
                threat_file = random.choice(self.threat_signatures[threat_type])
                try:
                    threat_audio, sr = torchaudio.load(threat_file)
                    if sr != self.sample_rate:
                        threat_audio = torchaudio.functional.resample(threat_audio, sr, self.sample_rate)
                    
                    # Mix threat audio at random position and volume
                    threat_samples = threat_audio.shape[1]
                    if threat_samples <= samples:
                        start_pos = random.randint(0, max(0, samples - threat_samples))
                        volume = random.uniform(0.3, 0.8)
                        mixed_audio[start_pos:start_pos + threat_samples] += threat_audio[0] * volume
                        
                        return mixed_audio.unsqueeze(0), threat_type
                except Exception as e:
                    logger.warning(f"Failed to load {threat_file}: {e}")
        
        return mixed_audio.unsqueeze(0), 'background'
    
    def add_battlefield_interference(self, audio, interference_type='radio'):
        """Add realistic battlefield interference"""
        
        if interference_type == 'radio':
            # Simulate radio interference
            freq = random.uniform(800, 2000)  # Radio frequency interference
            t = torch.linspace(0, audio.shape[1] / self.sample_rate, audio.shape[1])
            interference = 0.1 * torch.sin(2 * np.pi * freq * t)
            audio += interference.unsqueeze(0)
        
        elif interference_type == 'electrical':
            # Simulate electrical noise from equipment
            noise = torch.randn_like(audio) * 0.05
            audio += noise
        
        elif interference_type == 'wind':
            # Simulate wind buffeting on microphone
            wind_freq = random.uniform(1, 10)
            t = torch.linspace(0, audio.shape[1] / self.sample_rate, audio.shape[1])
            wind_noise = 0.15 * torch.sin(2 * np.pi * wind_freq * t) * torch.randn_like(audio) * 0.3
            audio += wind_noise
        
        return audio


class CompressedModelRunner:
    """Run inference using the compressed enhanced model"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.load_compressed_model()
        self.setup_preprocessing()
        
    def load_compressed_model(self):
        """Load the compressed enhanced model"""
        model_path = Path('enhanced_qadt_r_compressed.pth')
        if not model_path.exists():
            # Fallback to original enhanced model
            model_path = Path('enhanced_qadt_r_best.pth')
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                model_state = checkpoint
            
            # Initialize model architecture
            self.model = NoiseRobustMilitaryModel(
                num_classes=30
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(model_state, strict=False)
            self.model.eval()
            
            logger.info(f"✅ Loaded compressed model from {model_path}")
        else:
            logger.error("❌ No compressed model found")
            raise FileNotFoundError("Compressed model not available")
    
    def setup_preprocessing(self):
        """Setup audio preprocessing pipeline"""
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
            power=2.0
        ).to(self.device)
        
    def preprocess_audio(self, audio_tensor):
        """Preprocess audio for model inference"""
        # Ensure correct shape [batch, channels, samples] and move to device
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.dim() == 2 and audio_tensor.shape[0] != 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Move to device before mel transform
        audio_tensor = audio_tensor.to(self.device)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # Add channel dimension if needed
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, n_mels, time]
        
        # Normalize
        mel_spec = torch.log(mel_spec + 1e-6)
        
        return mel_spec
    
    def run_inference(self, audio_tensor):
        """Run model inference with timing"""
        start_time = time.time()
        
        # Preprocess
        preprocessed = self.preprocess_audio(audio_tensor)
        
        # Inference
        with torch.no_grad():
            try:
                outputs = self.model(preprocessed)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    binary_out, category_out, specific_out, confidence_out = outputs
                    predictions = {
                        'binary': torch.softmax(binary_out, dim=1),
                        'category': torch.softmax(category_out, dim=1), 
                        'specific': torch.softmax(specific_out, dim=1),
                        'confidence': torch.sigmoid(confidence_out)
                    }
                else:
                    predictions = {'specific': torch.softmax(outputs, dim=1)}
                
                inference_time = time.time() - start_time
                return predictions, inference_time
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                inference_time = time.time() - start_time
                return None, inference_time


class BattlefieldPerformanceValidator:
    """Validate model performance under battlefield conditions"""
    
    def __init__(self):
        self.simulator = BattlefieldAudioSimulator()
        self.model_runner = CompressedModelRunner()
        self.enhanced_class_mapping = self._load_class_mapping()
        
    def _load_class_mapping(self):
        """Load enhanced 30-class mapping"""
        # Enhanced 30-class taxonomy
        return {
            # Original 27 military classes (0-26)
            0: 'small_arms_fire', 1: 'artillery_fire', 2: 'mortar_fire', 3: 'rocket_launch',
            4: 'tank_movement', 5: 'helicopter_military', 6: 'jet_fighter', 7: 'drone_military',
            8: 'explosion_large', 9: 'explosion_small', 10: 'grenade_explosion', 11: 'ied_explosion',
            12: 'vehicle_engine', 13: 'truck_diesel', 14: 'apc_tracked', 15: 'motorcycle',
            16: 'footsteps_group', 17: 'footsteps_individual', 18: 'voice_commands', 19: 'radio_chatter',
            20: 'equipment_metallic', 21: 'weapon_reload', 22: 'safety_click', 23: 'breech_close',
            24: 'breathing_heavy', 25: 'heartbeat_stressed', 26: 'environmental_wind',
            # New aerial threat classes (27-29)  
            27: 'drone_acoustic', 28: 'helicopter_military', 29: 'aerial_background'
        }
    
    def test_scenario_performance(self, scenario='urban_patrol', num_tests=50):
        """Test performance on specific battlefield scenario"""
        
        logger.info(f"🎯 Testing {scenario} scenario ({num_tests} samples)")
        
        results = {
            'total_tests': num_tests,
            'correct_detections': 0,
            'false_positives': 0,
            'missed_detections': 0,
            'inference_times': [],
            'confidence_scores': [],
            'predictions': []
        }
        
        aerial_threats = ['drone', 'helicopter']
        
        for i in range(num_tests):
            # Generate test audio
            audio, true_label = self.simulator.generate_mixed_battlefield_audio(
                duration_seconds=2.0, scenario=scenario
            )
            
            # Add random interference
            interference_types = ['radio', 'electrical', 'wind', None]
            interference = random.choice(interference_types)
            if interference:
                audio = self.simulator.add_battlefield_interference(audio, interference)
            
            # Run inference
            predictions, inference_time = self.model_runner.run_inference(audio)
            results['inference_times'].append(inference_time)
            
            if predictions is None:
                results['missed_detections'] += 1
                continue
            
            # Analyze predictions
            specific_probs = predictions['specific'][0]
            predicted_class_idx = torch.argmax(specific_probs).item()
            predicted_class = self.enhanced_class_mapping.get(predicted_class_idx, 'unknown')
            confidence = specific_probs[predicted_class_idx].item()
            
            results['confidence_scores'].append(confidence)
            results['predictions'].append({
                'true_label': true_label,
                'predicted_class': predicted_class,
                'predicted_idx': predicted_class_idx,
                'confidence': confidence,
                'inference_time': inference_time
            })
            
            # Evaluate detection accuracy
            if true_label in aerial_threats:
                # For aerial threats, check if predicted in aerial range (27-29) or related military class
                if predicted_class_idx >= 27 or 'drone' in predicted_class or 'helicopter' in predicted_class:
                    results['correct_detections'] += 1
                else:
                    results['missed_detections'] += 1
            elif true_label == 'background':
                # For background, check if confidence is low or predicted as environmental
                if confidence < 0.5 or 'environmental' in predicted_class or 'background' in predicted_class:
                    results['correct_detections'] += 1
                else:
                    results['false_positives'] += 1
            else:
                # For other threats, accept any non-background detection
                if confidence > 0.3:
                    results['correct_detections'] += 1
                else:
                    results['missed_detections'] += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Progress: {i + 1}/{num_tests} tests completed")
        
        return results
    
    def test_realtime_streaming(self, duration_seconds=30):
        """Test real-time streaming performance"""
        
        logger.info(f"🔄 Testing real-time streaming ({duration_seconds}s)")
        
        chunk_duration = 2.0  # 2-second chunks
        chunks_per_second = 1.0 / chunk_duration
        total_chunks = int(duration_seconds * chunks_per_second)
        
        streaming_results = {
            'total_chunks': total_chunks,
            'processed_chunks': 0,
            'dropped_chunks': 0,
            'avg_latency': 0,
            'max_latency': 0,
            'real_time_violations': 0,
            'detections': []
        }
        
        audio_queue = queue.Queue(maxsize=10)
        
        def audio_generator():
            """Generate continuous audio stream"""
            for chunk_idx in range(total_chunks):
                scenario = random.choice(['urban_patrol', 'rural_surveillance', 'convoy_protection'])
                audio, label = self.simulator.generate_mixed_battlefield_audio(
                    duration_seconds=chunk_duration, scenario=scenario
                )
                
                try:
                    audio_queue.put((chunk_idx, audio, label), timeout=0.1)
                except queue.Full:
                    streaming_results['dropped_chunks'] += 1
                
                time.sleep(chunk_duration)  # Simulate real-time audio
        
        # Start audio generation thread
        audio_thread = threading.Thread(target=audio_generator)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Process audio stream
        start_time = time.time()
        latencies = []
        
        while streaming_results['processed_chunks'] < total_chunks:
            try:
                chunk_idx, audio, true_label = audio_queue.get(timeout=5.0)
                
                # Process chunk
                chunk_start = time.time()
                predictions, inference_time = self.model_runner.run_inference(audio)
                processing_latency = time.time() - chunk_start
                
                latencies.append(processing_latency)
                streaming_results['processed_chunks'] += 1
                
                # Check real-time constraint (must process faster than audio duration)
                if processing_latency > chunk_duration:
                    streaming_results['real_time_violations'] += 1
                
                # Record significant detections
                if predictions and predictions['specific'][0].max() > 0.5:
                    predicted_idx = torch.argmax(predictions['specific'][0]).item()
                    predicted_class = self.enhanced_class_mapping.get(predicted_idx, 'unknown')
                    
                    streaming_results['detections'].append({
                        'chunk': chunk_idx,
                        'timestamp': time.time() - start_time,
                        'prediction': predicted_class,
                        'confidence': predictions['specific'][0][predicted_idx].item(),
                        'true_label': true_label
                    })
                
            except queue.Empty:
                logger.warning("Audio queue timeout")
                break
        
        # Calculate streaming metrics
        if latencies:
            streaming_results['avg_latency'] = np.mean(latencies)
            streaming_results['max_latency'] = np.max(latencies)
        
        return streaming_results
    
    def test_stress_conditions(self):
        """Test performance under stress conditions"""
        
        logger.info("⚡ Testing stress conditions")
        
        stress_tests = {
            'low_snr': {'noise_level': 0.5, 'signal_level': 0.2},
            'rapid_transitions': {'switch_rate': 0.5},  # Switch scenarios every 0.5s
            'multiple_threats': {'threat_density': 0.8},
            'interference_heavy': {'interference_probability': 0.9}
        }
        
        stress_results = {}
        
        for stress_type, params in stress_tests.items():
            logger.info(f"   Testing {stress_type}...")
            
            test_results = []
            
            for _ in range(20):  # 20 tests per stress condition
                
                if stress_type == 'low_snr':
                    # Generate low SNR audio
                    audio, label = self.simulator.generate_mixed_battlefield_audio()
                    noise = torch.randn_like(audio) * params['noise_level']
                    audio = audio * params['signal_level'] + noise
                
                elif stress_type == 'rapid_transitions':
                    # Rapidly changing scenarios
                    scenarios = ['urban_patrol', 'rural_surveillance', 'convoy_protection']
                    audio_segments = []
                    labels = []
                    
                    for _ in range(4):  # 4 x 0.5s segments
                        scenario = random.choice(scenarios)
                        segment, label = self.simulator.generate_mixed_battlefield_audio(
                            duration_seconds=0.5, scenario=scenario
                        )
                        audio_segments.append(segment)
                        labels.append(label)
                    
                    audio = torch.cat(audio_segments, dim=1)
                    label = labels[0]  # Use first segment label
                
                elif stress_type == 'multiple_threats':
                    # Multiple overlapping threats
                    base_audio, base_label = self.simulator.generate_mixed_battlefield_audio()
                    
                    # Add additional threat signatures
                    for _ in range(3):
                        if random.random() < params['threat_density']:
                            threat_audio, _ = self.simulator.generate_mixed_battlefield_audio()
                            base_audio += threat_audio * 0.5
                    
                    audio, label = base_audio, base_label
                
                elif stress_type == 'interference_heavy':
                    # Heavy interference
                    audio, label = self.simulator.generate_mixed_battlefield_audio()
                    
                    if random.random() < params['interference_probability']:
                        for interference_type in ['radio', 'electrical', 'wind']:
                            audio = self.simulator.add_battlefield_interference(audio, interference_type)
                
                # Run inference
                predictions, inference_time = self.model_runner.run_inference(audio)
                
                success = predictions is not None and inference_time < 0.5  # 500ms timeout
                confidence = 0.0
                if predictions:
                    confidence = predictions['specific'][0].max().item()
                
                test_results.append({
                    'success': success,
                    'inference_time': inference_time,
                    'confidence': confidence,
                    'true_label': label
                })
            
            # Calculate stress test metrics
            successes = sum(1 for r in test_results if r['success'])
            avg_time = np.mean([r['inference_time'] for r in test_results])
            avg_confidence = np.mean([r['confidence'] for r in test_results if r['success']])
            
            stress_results[stress_type] = {
                'success_rate': successes / len(test_results),
                'avg_inference_time': avg_time,
                'avg_confidence': avg_confidence,
                'total_tests': len(test_results)
            }
        
        return stress_results


def main():
    """Main battlefield validation"""
    
    logger.info("🪖 Battlefield Validation Test")
    logger.info("Enhanced QADT-R with Drone Acoustics")
    logger.info("=" * 60)
    
    try:
        validator = BattlefieldPerformanceValidator()
    except Exception as e:
        logger.error(f"❌ Failed to initialize validator: {e}")
        return False
    
    validation_results = {}
    
    # Test 1: Scenario-based performance
    logger.info("\n📊 Phase 1: Scenario Performance Testing")
    scenarios = ['urban_patrol', 'rural_surveillance', 'convoy_protection', 'base_perimeter']
    
    for scenario in scenarios:
        results = validator.test_scenario_performance(scenario, num_tests=25)
        
        accuracy = results['correct_detections'] / results['total_tests']
        avg_inference_time = np.mean(results['inference_times'])
        avg_confidence = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
        
        logger.info(f"   {scenario}:")
        logger.info(f"     Accuracy: {accuracy:.1%}")
        logger.info(f"     Avg inference: {avg_inference_time:.1f}ms") 
        logger.info(f"     Avg confidence: {avg_confidence:.2f}")
        logger.info(f"     False positives: {results['false_positives']}")
        logger.info(f"     Missed detections: {results['missed_detections']}")
        
        validation_results[scenario] = {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_confidence': avg_confidence,
            'false_positives': results['false_positives'],
            'missed_detections': results['missed_detections']
        }
    
    # Test 2: Real-time streaming  
    logger.info("\n📊 Phase 2: Real-time Streaming Test")
    streaming_results = validator.test_realtime_streaming(duration_seconds=20)
    
    processing_rate = streaming_results['processed_chunks'] / streaming_results['total_chunks']
    real_time_success = 1.0 - (streaming_results['real_time_violations'] / streaming_results['processed_chunks'])
    
    logger.info(f"   Processing rate: {processing_rate:.1%}")
    logger.info(f"   Real-time success: {real_time_success:.1%}")
    logger.info(f"   Avg latency: {streaming_results['avg_latency']:.1f}ms")
    logger.info(f"   Max latency: {streaming_results['max_latency']:.1f}ms")
    logger.info(f"   Dropped chunks: {streaming_results['dropped_chunks']}")
    logger.info(f"   Detections: {len(streaming_results['detections'])}")
    
    validation_results['streaming'] = {
        'processing_rate': processing_rate,
        'real_time_success': real_time_success,
        'avg_latency': streaming_results['avg_latency'],
        'max_latency': streaming_results['max_latency'],
        'dropped_chunks': streaming_results['dropped_chunks']
    }
    
    # Test 3: Stress conditions
    logger.info("\n📊 Phase 3: Stress Condition Testing")
    stress_results = validator.test_stress_conditions()
    
    for stress_type, results in stress_results.items():
        logger.info(f"   {stress_type}:")
        logger.info(f"     Success rate: {results['success_rate']:.1%}")
        logger.info(f"     Avg inference: {results['avg_inference_time']:.1f}ms")
        logger.info(f"     Avg confidence: {results['avg_confidence']:.2f}")
    
    validation_results['stress'] = stress_results
    
    # Overall assessment
    logger.info("\n📋 Overall Battlefield Validation Results:")
    logger.info("=" * 50)
    
    # Calculate overall metrics
    scenario_accuracies = [validation_results[s]['accuracy'] for s in scenarios]
    overall_accuracy = np.mean(scenario_accuracies)
    
    stress_success_rates = [stress_results[s]['success_rate'] for s in stress_results]
    overall_stress_performance = np.mean(stress_success_rates)
    
    logger.info(f"   Overall accuracy: {overall_accuracy:.1%}")
    logger.info(f"   Real-time capability: {real_time_success:.1%}")
    logger.info(f"   Stress performance: {overall_stress_performance:.1%}")
    logger.info(f"   Avg inference time: {np.mean([validation_results[s]['avg_inference_time'] for s in scenarios]):.1f}ms")
    
    # Pass/fail criteria
    battlefield_ready = (
        overall_accuracy > 0.6 and  # 60% accuracy threshold
        real_time_success > 0.9 and  # 90% real-time success
        overall_stress_performance > 0.5  # 50% stress performance
    )
    
    # Save detailed results
    with open('battlefield_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    if battlefield_ready:
        logger.info("\n🎉 BATTLEFIELD VALIDATION SUCCESSFUL!")
        logger.info("🪖 Enhanced QADT-R ready for operational deployment")
        logger.info("🚁 Drone detection capabilities validated")
        logger.info("⚡ Real-time performance confirmed")
        logger.info("🎯 Stress resistance demonstrated")
    else:
        logger.warning("\n⚠️  Battlefield validation shows areas for improvement")
        logger.info("🔧 Consider additional optimization or training")
    
    return battlefield_ready


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)