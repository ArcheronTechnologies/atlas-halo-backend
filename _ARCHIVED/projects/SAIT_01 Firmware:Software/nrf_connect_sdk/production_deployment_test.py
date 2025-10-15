#!/usr/bin/env python3
"""
🛡️ SAIT_01 Production Deployment Test
=====================================
Quick validation test for production deployment readiness

Tests core functionality with trained models and consensus system
"""

import numpy as np
import time
import os
from mesh_threat_detection import MeshThreatDetectionSystem
import librosa

def quick_deployment_test():
    """Quick test to validate production deployment readiness"""
    print("🛡️ SAIT_01 Production Deployment Test")
    print("=" * 50)
    
    # Check for trained models
    model_files = [
        "final_best_model.h5",
        "sait01_final_high_accuracy.tflite",
        "sait01_quantized_model.tflite",
        "sait01_simple_model.tflite"
    ]
    
    available_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            available_models.append(model_file)
            print(f"✅ Found model: {model_file}")
        else:
            print(f"⚠️  Missing model: {model_file}")
    
    if not available_models:
        print("❌ No trained models found - run training first")
        return False
    
    # Test with TFLite model if available
    tflite_model = None
    for model in available_models:
        if "tflite" in model:
            tflite_model = model
            break
    
    print(f"\n🧪 Testing with model: {tflite_model or 'baseline'}")
    
    # Initialize detection system
    detection_system = MeshThreatDetectionSystem("SAIT01_Production_Test")
    
    if tflite_model:
        detection_system.load_model(tflite_model)
    
    detection_system.start_system(location=(40.7128, -74.0060))
    
    # Quick functional tests
    test_scenarios = [
        ("Background noise", np.random.randn(16000) * 0.1),
        ("Synthetic drone", generate_drone_audio()),
        ("Synthetic helicopter", generate_helicopter_audio())
    ]
    
    results = []
    
    for scenario_name, audio_data in test_scenarios:
        print(f"\n📊 Testing: {scenario_name}")
        
        start_time = time.time()
        alerts = detection_system.process_audio_stream(audio_data)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"   Processing time: {processing_time:.1f}ms")
        print(f"   Alerts generated: {len(alerts)}")
        
        if alerts:
            max_confidence = max(alert.confidence for alert in alerts)
            threat_levels = [alert.threat_level.name for alert in alerts]
            print(f"   Max confidence: {max_confidence:.3f}")
            print(f"   Threat levels: {threat_levels}")
        
        results.append({
            'scenario': scenario_name,
            'processing_time': processing_time,
            'alerts': len(alerts),
            'functional': processing_time < 200  # 200ms tolerance for testing
        })
    
    # System status
    status = detection_system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Local model loaded: {status['local_model_loaded']}")
    print(f"   Mesh nodes: {status['mesh_status']['active_neighbors'] + 1}")
    print(f"   Total detections: {status['detection_stats']['total_detections']}")
    
    # Cleanup
    detection_system.stop_system()
    
    # Assessment
    all_functional = all(r['functional'] for r in results)
    avg_processing = np.mean([r['processing_time'] for r in results])
    
    print(f"\n🎯 Deployment Assessment:")
    print(f"   Functional tests: {'✅ PASS' if all_functional else '❌ FAIL'}")
    print(f"   Avg processing: {avg_processing:.1f}ms")
    print(f"   Real-time capable: {'✅ YES' if avg_processing < 100 else '⚠️  MARGINAL' if avg_processing < 200 else '❌ NO'}")
    
    if all_functional and avg_processing < 200:
        print(f"   Status: 🚀 READY FOR DEPLOYMENT")
        return True
    else:
        print(f"   Status: ⚠️  NEEDS OPTIMIZATION")
        return False

def generate_drone_audio(duration=1.0, sample_rate=16000):
    """Generate synthetic drone audio for testing"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Drone characteristics
    freq1 = 1800 + 200 * np.sin(2 * np.pi * 0.5 * t)
    freq2 = 2200 + 150 * np.sin(2 * np.pi * 0.8 * t)
    
    audio = 0.3 * np.sin(2 * np.pi * freq1 * t)
    audio += 0.2 * np.sin(2 * np.pi * freq2 * t)
    audio += 0.05 * np.random.randn(len(t))
    
    return audio

def generate_helicopter_audio(duration=1.0, sample_rate=16000):
    """Generate synthetic helicopter audio for testing"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Helicopter rotor characteristics
    rotor_freq = 12 + 2 * np.sin(2 * np.pi * 0.3 * t)
    
    audio = 0.5 * np.sin(2 * np.pi * rotor_freq * t)
    audio += 0.3 * np.sin(2 * np.pi * rotor_freq * 2 * t)
    audio += 0.1 * np.sin(2 * np.pi * rotor_freq * 3 * t)
    audio += 0.05 * np.random.randn(len(t))
    
    return audio

if __name__ == "__main__":
    success = quick_deployment_test()
    exit(0 if success else 1)