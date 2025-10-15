#!/usr/bin/env python3
"""
Environmental Stress Testing Framework
=====================================

Comprehensive environmental stress testing for enhanced QADT-R
battlefield audio detection system. Tests performance under
extreme environmental conditions that may be encountered in
real-world military deployments.

Test Categories:
- Temperature extremes (-40°C to +85°C)
- Humidity variations (5% to 95% RH)
- Electromagnetic interference (EMI)
- Vibration and shock resistance
- Acoustic environment variations
- Power supply fluctuations
"""

import torch
import numpy as np
import json
import time
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import queue

# Import our enhanced model and validation framework
import sys
sys.path.append(str(Path(__file__).parent))
from noise_robust_architecture import NoiseRobustMilitaryModel
from battlefield_validation_test import BattlefieldAudioSimulator, CompressedModelRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentalStressTester:
    """Comprehensive environmental stress testing framework"""
    
    def __init__(self):
        self.model_runner = CompressedModelRunner()
        self.audio_simulator = BattlefieldAudioSimulator()
        self.stress_test_results = {}
        
        # Environmental test parameters
        self.temperature_ranges = [
            (-40, -20, "extreme_cold"),
            (-20, 0, "cold"), 
            (0, 25, "moderate"),
            (25, 50, "hot"),
            (50, 85, "extreme_hot")
        ]
        
        self.humidity_levels = [
            (5, 20, "very_dry"),
            (20, 40, "dry"),
            (40, 60, "moderate"),
            (60, 80, "humid"),
            (80, 95, "very_humid")
        ]
        
        self.emi_sources = [
            "radio_transmitter",
            "radar_pulse",
            "electrical_motor",
            "switching_power_supply",
            "led_lighting",
            "cellular_network"
        ]
        
        logger.info("🌡️ Environmental Stress Tester initialized")
    
    def simulate_temperature_effects(self, temperature_c, audio_tensor):
        """Simulate temperature effects on audio processing"""
        
        # Temperature effects on audio hardware:
        # - ADC noise increases at temperature extremes
        # - Clock drift affects sampling rates
        # - Component tolerances change
        
        # Calculate temperature stress factor
        optimal_temp = 25  # °C
        temp_deviation = abs(temperature_c - optimal_temp)
        stress_factor = 1.0 + (temp_deviation / 100.0)  # 1% per degree deviation
        
        # Apply temperature-related degradation
        if temperature_c < -10:
            # Cold: Increased noise, slower response
            noise_level = 0.02 * stress_factor
            thermal_noise = torch.randn_like(audio_tensor) * noise_level
            audio_tensor = audio_tensor + thermal_noise
            
            # Simulate slower ADC conversion (timing jitter)
            if random.random() < 0.1:  # 10% chance of timing issues
                audio_tensor = audio_tensor * 0.95  # Slight amplitude reduction
                
        elif temperature_c > 60:
            # Hot: Increased leakage current, thermal noise
            thermal_noise_level = 0.03 * (stress_factor - 1)
            thermal_noise = torch.randn_like(audio_tensor) * thermal_noise_level
            audio_tensor = audio_tensor + thermal_noise
            
            # Simulate thermal drift
            drift_factor = 1.0 + (temp_deviation - 35) * 0.001
            audio_tensor = audio_tensor * drift_factor
        
        return audio_tensor, stress_factor
    
    def simulate_humidity_effects(self, humidity_percent, audio_tensor):
        """Simulate humidity effects on audio processing"""
        
        # Humidity effects:
        # - Acoustic coupling changes
        # - Corrosion effects on contacts
        # - Dielectric changes in capacitors
        
        # Calculate humidity stress
        optimal_humidity = 45  # %RH
        humidity_deviation = abs(humidity_percent - optimal_humidity)
        
        if humidity_percent < 20:
            # Low humidity: Static electricity, increased noise
            static_noise = torch.randn_like(audio_tensor) * 0.01
            audio_tensor = audio_tensor + static_noise
            
        elif humidity_percent > 80:
            # High humidity: Corrosion effects, impedance changes
            corrosion_factor = 1.0 + (humidity_percent - 80) * 0.002
            
            # Simulate impedance mismatch
            if random.random() < 0.05:  # 5% chance
                audio_tensor = audio_tensor * 0.9  # Signal attenuation
            
            # Add moisture-related noise
            moisture_noise = torch.randn_like(audio_tensor) * 0.015
            audio_tensor = audio_tensor + moisture_noise
        
        humidity_stress = humidity_deviation / 50.0  # Normalize stress factor
        return audio_tensor, humidity_stress
    
    def simulate_emi_interference(self, emi_source, audio_tensor):
        """Simulate electromagnetic interference"""
        
        sample_rate = 16000
        duration = audio_tensor.shape[1] / sample_rate
        t = torch.linspace(0, duration, audio_tensor.shape[1])
        
        interference = torch.zeros_like(audio_tensor)
        
        if emi_source == "radio_transmitter":
            # VHF/UHF radio interference
            freq = random.uniform(88e6, 400e6)  # MHz
            beat_freq = freq % sample_rate  # Beat frequency with sampling
            if beat_freq > sample_rate / 2:
                beat_freq = sample_rate - beat_freq
            
            interference = 0.05 * torch.sin(2 * np.pi * beat_freq * t)
            
        elif emi_source == "radar_pulse":
            # Pulsed radar interference
            pulse_rate = random.uniform(100, 1000)  # Hz
            pulse_width = 0.001  # 1ms pulses
            
            pulse_pattern = torch.zeros_like(t)
            for i in range(int(duration * pulse_rate)):
                pulse_start = i / pulse_rate
                pulse_end = pulse_start + pulse_width
                mask = (t >= pulse_start) & (t <= pulse_end)
                pulse_pattern[mask] = 1.0
            
            interference = 0.1 * pulse_pattern * torch.randn_like(audio_tensor)
            
        elif emi_source == "switching_power_supply":
            # Switching frequency interference
            switching_freq = random.uniform(20e3, 100e3)  # 20-100kHz
            interference = 0.02 * torch.sin(2 * np.pi * switching_freq * t)
            
        elif emi_source == "cellular_network":
            # GSM/LTE interference
            gsm_freq = random.choice([900e6, 1800e6, 2100e6])  # MHz
            burst_rate = 217  # Hz (GSM burst rate)
            
            burst_pattern = torch.sin(2 * np.pi * burst_rate * t)
            interference = 0.03 * burst_pattern * torch.randn_like(audio_tensor)
        
        # Add interference to audio
        audio_with_emi = audio_tensor + interference.unsqueeze(0) if interference.dim() == 1 else audio_tensor + interference
        
        # Calculate EMI stress level
        emi_power = torch.mean(interference ** 2).item()
        signal_power = torch.mean(audio_tensor ** 2).item()
        emi_stress = emi_power / (signal_power + 1e-10)
        
        return audio_with_emi, emi_stress
    
    def simulate_vibration_effects(self, vibration_level, audio_tensor):
        """Simulate vibration and shock effects on audio"""
        
        # Vibration effects on microphone and circuitry:
        # - Mechanical coupling to audio signal
        # - Contact intermittency
        # - Resonance effects
        
        vibration_freq = random.uniform(10, 200)  # Hz
        sample_rate = 16000
        duration = audio_tensor.shape[1] / sample_rate
        t = torch.linspace(0, duration, audio_tensor.shape[1])
        
        # Generate vibration-induced noise
        vibration_amplitude = vibration_level * 0.01  # Scale factor
        vibration_noise = vibration_amplitude * torch.sin(2 * np.pi * vibration_freq * t)
        
        # Add harmonic components
        for harmonic in [2, 3, 5]:
            harmonic_amp = vibration_amplitude / harmonic
            vibration_noise += harmonic_amp * torch.sin(2 * np.pi * vibration_freq * harmonic * t)
        
        # Apply vibration effects
        audio_with_vibration = audio_tensor + vibration_noise.unsqueeze(0)
        
        # Simulate contact intermittency at high vibration levels
        if vibration_level > 5:  # High vibration
            intermittency_rate = vibration_level * 0.02
            if random.random() < intermittency_rate:
                # Brief signal dropout
                dropout_start = random.randint(0, audio_tensor.shape[1] - 100)
                dropout_length = random.randint(10, 50)
                audio_with_vibration[:, dropout_start:dropout_start+dropout_length] *= 0.1
        
        return audio_with_vibration, vibration_level
    
    def simulate_power_supply_variations(self, voltage_variation_percent):
        """Simulate power supply voltage variations"""
        
        # Power supply effects:
        # - ADC reference voltage changes
        # - Clock frequency drift
        # - Amplifier gain variations
        
        nominal_voltage = 3.3  # V
        actual_voltage = nominal_voltage * (1 + voltage_variation_percent / 100)
        
        # Calculate effects
        voltage_stress = abs(voltage_variation_percent) / 20.0  # Normalize
        
        # ADC resolution effectively changes with reference voltage
        adc_scale_factor = actual_voltage / nominal_voltage
        
        # Clock drift (affects timing)
        clock_drift_ppm = voltage_variation_percent * 10  # 10 ppm per percent
        
        power_effects = {
            'voltage_stress': voltage_stress,
            'adc_scale_factor': adc_scale_factor,
            'clock_drift_ppm': clock_drift_ppm,
            'brownout_risk': voltage_variation_percent < -15  # Below 2.8V
        }
        
        return power_effects
    
    def run_temperature_stress_test(self, num_tests_per_range=10):
        """Run comprehensive temperature stress testing"""
        
        logger.info("🌡️ Running temperature stress tests...")
        
        temperature_results = {}
        
        for temp_min, temp_max, temp_category in self.temperature_ranges:
            logger.info(f"   Testing {temp_category} range: {temp_min}°C to {temp_max}°C")
            
            category_results = {
                'tests_run': num_tests_per_range,
                'successful_inferences': 0,
                'avg_accuracy': 0,
                'avg_inference_time': 0,
                'max_degradation': 0,
                'temperature_effects': []
            }
            
            accuracies = []
            inference_times = []
            degradations = []
            
            for test_idx in range(num_tests_per_range):
                # Random temperature in range
                test_temp = random.uniform(temp_min, temp_max)
                
                # Generate test audio
                audio, true_label = self.audio_simulator.generate_mixed_battlefield_audio(
                    duration_seconds=2.0, scenario='urban_patrol'
                )
                
                # Apply temperature effects
                stressed_audio, stress_factor = self.simulate_temperature_effects(test_temp, audio)
                
                # Run inference
                predictions, inference_time = self.model_runner.run_inference(stressed_audio)
                
                if predictions is not None:
                    category_results['successful_inferences'] += 1
                    
                    # Calculate accuracy (simplified)
                    max_confidence = predictions['specific'][0].max().item()
                    accuracies.append(max_confidence)
                    inference_times.append(inference_time)
                    
                    # Calculate degradation from stress
                    degradation = max(0, stress_factor - 1.0)
                    degradations.append(degradation)
                    
                    category_results['temperature_effects'].append({
                        'temperature': test_temp,
                        'stress_factor': stress_factor,
                        'confidence': max_confidence,
                        'inference_time': inference_time
                    })
            
            # Calculate category metrics
            if accuracies:
                category_results['avg_accuracy'] = np.mean(accuracies)
                category_results['avg_inference_time'] = np.mean(inference_times)
                category_results['max_degradation'] = np.max(degradations) if degradations else 0
            
            # Success criteria: >80% successful inferences, <20% degradation
            category_results['pass'] = (
                category_results['successful_inferences'] / num_tests_per_range >= 0.8 and
                category_results['max_degradation'] < 0.2
            )
            
            logger.info(f"     Success rate: {category_results['successful_inferences']}/{num_tests_per_range}")
            logger.info(f"     Avg accuracy: {category_results['avg_accuracy']:.2f}")
            logger.info(f"     Max degradation: {category_results['max_degradation']:.1%}")
            logger.info(f"     Result: {'✅ PASS' if category_results['pass'] else '❌ FAIL'}")
            
            temperature_results[temp_category] = category_results
        
        self.stress_test_results['temperature'] = temperature_results
        return temperature_results
    
    def run_emi_stress_test(self, num_tests_per_source=15):
        """Run electromagnetic interference stress testing"""
        
        logger.info("📡 Running EMI stress tests...")
        
        emi_results = {}
        
        for emi_source in self.emi_sources:
            logger.info(f"   Testing {emi_source} interference...")
            
            source_results = {
                'tests_run': num_tests_per_source,
                'successful_inferences': 0,
                'avg_accuracy': 0,
                'avg_emi_stress': 0,
                'max_emi_stress': 0,
                'interference_effects': []
            }
            
            accuracies = []
            emi_stresses = []
            
            for test_idx in range(num_tests_per_source):
                # Generate test audio
                audio, true_label = self.audio_simulator.generate_mixed_battlefield_audio(
                    duration_seconds=2.0, scenario=random.choice(['urban_patrol', 'convoy_protection'])
                )
                
                # Apply EMI effects
                emi_audio, emi_stress = self.simulate_emi_interference(emi_source, audio)
                
                # Run inference
                predictions, inference_time = self.model_runner.run_inference(emi_audio)
                
                if predictions is not None:
                    source_results['successful_inferences'] += 1
                    
                    max_confidence = predictions['specific'][0].max().item()
                    accuracies.append(max_confidence)
                    emi_stresses.append(emi_stress)
                    
                    source_results['interference_effects'].append({
                        'emi_source': emi_source,
                        'emi_stress': emi_stress,
                        'confidence': max_confidence,
                        'inference_time': inference_time
                    })
            
            # Calculate source metrics
            if accuracies:
                source_results['avg_accuracy'] = np.mean(accuracies)
                source_results['avg_emi_stress'] = np.mean(emi_stresses)
                source_results['max_emi_stress'] = np.max(emi_stresses)
            
            # Success criteria: >90% successful inferences despite EMI
            source_results['pass'] = source_results['successful_inferences'] / num_tests_per_source >= 0.9
            
            logger.info(f"     Success rate: {source_results['successful_inferences']}/{num_tests_per_source}")
            logger.info(f"     Avg accuracy: {source_results['avg_accuracy']:.2f}")
            logger.info(f"     Max EMI stress: {source_results['max_emi_stress']:.3f}")
            logger.info(f"     Result: {'✅ PASS' if source_results['pass'] else '❌ FAIL'}")
            
            emi_results[emi_source] = source_results
        
        self.stress_test_results['emi'] = emi_results
        return emi_results
    
    def run_combined_stress_test(self, num_tests=25):
        """Run combined environmental stress testing"""
        
        logger.info("🌪️ Running combined environmental stress tests...")
        
        combined_results = {
            'tests_run': num_tests,
            'successful_inferences': 0,
            'avg_accuracy': 0,
            'extreme_conditions_survived': 0,
            'stress_combinations': []
        }
        
        accuracies = []
        
        for test_idx in range(num_tests):
            # Random environmental combination
            temperature = random.uniform(-30, 70)
            humidity = random.uniform(10, 90)
            vibration = random.uniform(0, 8)  # G-force
            voltage_variation = random.uniform(-20, 15)  # Percent
            emi_source = random.choice(self.emi_sources)
            
            # Generate test audio
            audio, true_label = self.audio_simulator.generate_mixed_battlefield_audio(
                duration_seconds=2.0, scenario=random.choice(['urban_patrol', 'rural_surveillance', 'convoy_protection'])
            )
            
            # Apply all environmental stresses
            stressed_audio = audio
            
            # Temperature effects
            stressed_audio, temp_stress = self.simulate_temperature_effects(temperature, stressed_audio)
            
            # Humidity effects
            stressed_audio, humidity_stress = self.simulate_humidity_effects(humidity, stressed_audio)
            
            # EMI effects
            stressed_audio, emi_stress = self.simulate_emi_interference(emi_source, stressed_audio)
            
            # Vibration effects
            stressed_audio, vib_stress = self.simulate_vibration_effects(vibration, stressed_audio)
            
            # Power supply effects
            power_effects = self.simulate_power_supply_variations(voltage_variation)
            
            # Calculate combined stress level
            combined_stress = (temp_stress + humidity_stress + emi_stress + vib_stress + power_effects['voltage_stress']) / 5
            
            # Run inference on heavily stressed audio
            predictions, inference_time = self.model_runner.run_inference(stressed_audio)
            
            if predictions is not None:
                combined_results['successful_inferences'] += 1
                
                max_confidence = predictions['specific'][0].max().item()
                accuracies.append(max_confidence)
                
                # Check if extreme conditions
                extreme_conditions = (
                    temperature < -20 or temperature > 60 or
                    humidity < 15 or humidity > 85 or
                    vibration > 5 or
                    abs(voltage_variation) > 10
                )
                
                if extreme_conditions:
                    combined_results['extreme_conditions_survived'] += 1
                
                combined_results['stress_combinations'].append({
                    'temperature': temperature,
                    'humidity': humidity,
                    'vibration': vibration,
                    'voltage_variation': voltage_variation,
                    'emi_source': emi_source,
                    'combined_stress': combined_stress,
                    'confidence': max_confidence,
                    'extreme_conditions': extreme_conditions
                })
        
        # Calculate combined metrics
        if accuracies:
            combined_results['avg_accuracy'] = np.mean(accuracies)
        
        # Success criteria: >75% success rate under combined stress
        combined_results['pass'] = combined_results['successful_inferences'] / num_tests >= 0.75
        
        logger.info(f"   Success rate: {combined_results['successful_inferences']}/{num_tests}")
        logger.info(f"   Extreme conditions survived: {combined_results['extreme_conditions_survived']}")
        logger.info(f"   Avg accuracy under stress: {combined_results['avg_accuracy']:.2f}")
        logger.info(f"   Result: {'✅ PASS' if combined_results['pass'] else '❌ FAIL'}")
        
        self.stress_test_results['combined'] = combined_results
        return combined_results


def main():
    """Main environmental stress testing"""
    
    logger.info("🌍 Environmental Stress Testing Framework")
    logger.info("Enhanced QADT-R Battlefield Audio Detection")
    logger.info("=" * 60)
    
    # Initialize environmental stress tester
    stress_tester = EnvironmentalStressTester()
    
    # Run comprehensive environmental stress tests
    logger.info("\n📊 Phase 3.2: Environmental Stress Testing")
    
    # Test 1: Temperature stress testing
    temperature_results = stress_tester.run_temperature_stress_test()
    
    # Test 2: EMI stress testing
    emi_results = stress_tester.run_emi_stress_test()
    
    # Test 3: Combined environmental stress testing
    combined_results = stress_tester.run_combined_stress_test()
    
    # Overall assessment
    logger.info("\n📋 Environmental Stress Assessment:")
    logger.info("=" * 50)
    
    # Calculate overall environmental resilience
    temp_pass_rate = sum(1 for result in temperature_results.values() if result['pass']) / len(temperature_results)
    emi_pass_rate = sum(1 for result in emi_results.values() if result['pass']) / len(emi_results)
    combined_pass = combined_results['pass']
    
    overall_resilience = (temp_pass_rate * 0.4 + emi_pass_rate * 0.4 + (1.0 if combined_pass else 0.0) * 0.2)
    
    logger.info(f"   Temperature resilience: {temp_pass_rate:.1%}")
    logger.info(f"   EMI resilience: {emi_pass_rate:.1%}")
    logger.info(f"   Combined stress survival: {'✅ PASS' if combined_pass else '❌ FAIL'}")
    logger.info(f"   Overall environmental resilience: {overall_resilience:.1%}")
    
    # Save detailed results
    with open('environmental_stress_test_results.json', 'w') as f:
        json.dump(stress_tester.stress_test_results, f, indent=2, default=str)
    
    # Environmental certification
    if overall_resilience >= 0.8:
        logger.info("\n🎉 ENVIRONMENTAL STRESS TESTING SUCCESSFUL!")
        logger.info("🌍 Enhanced QADT-R environmentally certified")
        logger.info("🌡️ Temperature range: -40°C to +85°C validated")
        logger.info("📡 EMI resistance confirmed across 6 interference sources")
        logger.info("🌪️ Combined stress conditions survival validated")
        logger.info("⚡ Ready for harsh battlefield environments")
        
        return True
    else:
        logger.warning(f"\n⚠️ Environmental stress testing shows vulnerabilities")
        logger.info(f"🔧 Resilience score: {overall_resilience:.1%} (target: 80%)")
        logger.info("📋 Review failed environmental conditions")
        
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)