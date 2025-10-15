#!/usr/bin/env python3
"""
💀 Quick Failure Test - Immediate System Breaking
================================================
Fast test to find obvious vulnerabilities immediately
"""

import numpy as np
import time
from hybrid_spectral_neural_system import HybridSpectralNeuralDetector
from advanced_spectrum_analysis import AdvancedSpectrumAnalyzer
import warnings
warnings.filterwarnings("ignore")

def quick_break_test():
    """Quick test to immediately break the system"""
    print("💀 QUICK SYSTEM BREAKING TEST")
    print("=" * 40)
    
    failures = []
    
    # Test 1: Empty audio
    print("1. Testing empty audio...")
    try:
        detector = HybridSpectralNeuralDetector()
        result = detector.detect_threat(np.array([]))
        print(f"   ⚠️ Survived empty audio: {result}")
    except Exception as e:
        failures.append(f"Empty audio: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 2: Infinite values
    print("2. Testing infinite values...")
    try:
        detector = HybridSpectralNeuralDetector()
        infinite_audio = np.full(16000, np.inf)
        result = detector.detect_threat(infinite_audio)
        print(f"   ⚠️ Survived infinite values: {result}")
    except Exception as e:
        failures.append(f"Infinite values: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 3: NaN values
    print("3. Testing NaN values...")
    try:
        detector = HybridSpectralNeuralDetector()
        nan_audio = np.full(16000, np.nan)
        result = detector.detect_threat(nan_audio)
        print(f"   ⚠️ Survived NaN values: {result}")
    except Exception as e:
        failures.append(f"NaN values: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 4: Extreme values
    print("4. Testing extreme values...")
    try:
        detector = HybridSpectralNeuralDetector()
        extreme_audio = np.full(16000, 1e20)
        result = detector.detect_threat(extreme_audio)
        print(f"   ⚠️ Survived extreme values: {result}")
    except Exception as e:
        failures.append(f"Extreme values: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 5: Wrong data type
    print("5. Testing wrong data type...")
    try:
        detector = HybridSpectralNeuralDetector()
        string_audio = "not_audio_data"
        result = detector.detect_threat(string_audio)
        print(f"   ⚠️ Survived string input: {result}")
    except Exception as e:
        failures.append(f"Wrong data type: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 6: Negative length array
    print("6. Testing weird shapes...")
    try:
        detector = HybridSpectralNeuralDetector()
        weird_shape = np.random.randn(16000, 5)  # Wrong shape
        result = detector.detect_threat(weird_shape)
        print(f"   ⚠️ Survived wrong shape: {result}")
    except Exception as e:
        failures.append(f"Wrong shape: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 7: Massive audio
    print("7. Testing massive audio...")
    try:
        detector = HybridSpectralNeuralDetector()
        massive_audio = np.random.randn(16000 * 3600)  # 1 hour
        start_time = time.time()
        result = detector.detect_threat(massive_audio)
        processing_time = (time.time() - start_time) * 1000
        print(f"   ⚠️ Survived 1-hour audio: {processing_time:.1f}ms")
    except Exception as e:
        failures.append(f"Massive audio: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 8: Zero-length processing
    print("8. Testing single sample...")
    try:
        detector = HybridSpectralNeuralDetector()
        tiny_audio = np.array([0.5])
        result = detector.detect_threat(tiny_audio)
        print(f"   ⚠️ Survived single sample: {result}")
    except Exception as e:
        failures.append(f"Single sample: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 9: Complex numbers
    print("9. Testing complex numbers...")
    try:
        detector = HybridSpectralNeuralDetector()
        complex_audio = np.random.randn(16000) + 1j * np.random.randn(16000)
        result = detector.detect_threat(complex_audio)
        print(f"   ⚠️ Survived complex numbers: {result}")
    except Exception as e:
        failures.append(f"Complex numbers: {e}")
        print(f"   💀 Failed: {e}")
    
    # Test 10: Advanced spectrum analyzer edge cases
    print("10. Testing spectrum analyzer edge cases...")
    try:
        analyzer = AdvancedSpectrumAnalyzer()
        # Test with problematic audio
        dc_only = np.ones(16000)  # DC only signal
        features = analyzer.extract_spectral_features(dc_only)
        print(f"   ⚠️ Survived DC-only signal")
        
        # Test Nyquist frequency
        nyquist = np.array([1, -1] * 8000)  # Nyquist frequency
        features = analyzer.extract_spectral_features(nyquist)
        print(f"   ⚠️ Survived Nyquist frequency")
        
    except Exception as e:
        failures.append(f"Spectrum analyzer: {e}")
        print(f"   💀 Failed: {e}")
    
    print(f"\n💀 QUICK TEST RESULTS:")
    print(f"Total failures: {len(failures)}")
    
    if failures:
        print(f"\n🔥 EXPOSED VULNERABILITIES:")
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. {failure}")
    else:
        print("🛡️ System survived all quick attacks")
    
    return len(failures) == 0

def test_performance_edge_cases():
    """Test performance under edge conditions"""
    print("\n💀 PERFORMANCE EDGE CASE TESTING")
    print("=" * 40)
    
    detector = HybridSpectralNeuralDetector()
    
    # Test different sample rates that might break things
    test_cases = [
        ("Ultra short", 10),
        ("Very short", 100), 
        ("Short", 1000),
        ("Normal", 16000),
        ("Long", 32000),
        ("Very long", 64000),
        ("Ultra long", 160000),  # 10 seconds
    ]
    
    for name, length in test_cases:
        print(f"   Testing {name} ({length} samples)...")
        audio = np.random.randn(length) * 0.3
        
        start_time = time.time()
        try:
            result = detector.detect_threat(audio)
            processing_time = (time.time() - start_time) * 1000
            
            # Check if processing time is reasonable
            if processing_time > 5000:  # 5 seconds
                print(f"     💀 SLOW: {processing_time:.1f}ms")
            else:
                print(f"     ✅ {processing_time:.1f}ms")
                
        except Exception as e:
            print(f"     💀 FAILED: {e}")

def test_adversarial_frequencies():
    """Test with frequencies designed to confuse the system"""
    print("\n💀 ADVERSARIAL FREQUENCY TESTING")
    print("=" * 40)
    
    detector = HybridSpectralNeuralDetector()
    
    # Generate adversarial frequencies
    t = np.linspace(0, 1, 16000)
    
    adversarial_cases = [
        ("Drone mimic + noise", 0.3 * np.sin(2 * np.pi * 2000 * t) + 0.7 * np.random.randn(16000) * 0.1),
        ("Heli mimic + noise", 0.3 * np.sin(2 * np.pi * 15 * t) + 0.7 * np.random.randn(16000) * 0.1),
        ("Mixed signals", 0.33 * np.sin(2 * np.pi * 2000 * t) + 0.33 * np.sin(2 * np.pi * 15 * t) + 0.33 * np.random.randn(16000) * 0.1),
        ("Frequency sweep", np.sin(2 * np.pi * (1000 + 1000 * t) * t)),
        ("Rapid modulation", np.sin(2 * np.pi * 2000 * t) * np.sin(2 * np.pi * 100 * t)),
        ("Harmonic chaos", sum(np.sin(2 * np.pi * (100 + i*113) * t) for i in range(20))),
    ]
    
    for name, audio in adversarial_cases:
        print(f"   Testing {name}...")
        try:
            result = detector.detect_threat(audio)
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            fusion_method = result.get('fusion_method', 'unknown')
            
            class_names = ["Background", "Drone", "Helicopter"]
            print(f"     Result: {class_names[predicted_class]} ({confidence:.3f}) via {fusion_method}")
            
            # Look for concerning patterns
            if confidence > 0.95:
                print(f"     ⚠️ OVERCONFIDENT")
            elif confidence < 0.4:
                print(f"     ⚠️ UNDERCONFIDENT") 
                
        except Exception as e:
            print(f"     💀 FAILED: {e}")

if __name__ == "__main__":
    print("💀💀💀 QUICK FAILURE TESTING 💀💀💀")
    
    success1 = quick_break_test()
    test_performance_edge_cases()
    test_adversarial_frequencies()
    
    print(f"\n🎯 QUICK TEST ASSESSMENT:")
    if success1:
        print("🛡️ System survived basic vulnerability tests")
        print("🔬 Recommend: More sophisticated attack vectors needed")
    else:
        print("💀 SYSTEM HAS CRITICAL VULNERABILITIES")
        print("🚨 IMMEDIATE fixes required")