#!/usr/bin/env python3
"""
💀 Simple Vulnerability Test - Find the Hanging Issue
====================================================
"""

import numpy as np
import time
from advanced_spectrum_analysis import AdvancedSpectrumAnalyzer
import warnings
warnings.filterwarnings("ignore")

def test_individual_components():
    """Test individual components to isolate the hanging issue"""
    print("💀 INDIVIDUAL COMPONENT TESTING")
    print("=" * 40)
    
    # Test 1: Basic spectrum analyzer with empty array
    print("1. Testing AdvancedSpectrumAnalyzer with empty array...")
    try:
        analyzer = AdvancedSpectrumAnalyzer()
        start_time = time.time()
        features = analyzer.extract_spectral_features(np.array([]))
        elapsed = (time.time() - start_time) * 1000
        print(f"   ✅ Completed in {elapsed:.1f}ms")
    except Exception as e:
        print(f"   💀 Failed: {e}")
    
    # Test 2: Basic spectrum analyzer with single sample
    print("2. Testing AdvancedSpectrumAnalyzer with single sample...")
    try:
        analyzer = AdvancedSpectrumAnalyzer()
        start_time = time.time()
        features = analyzer.extract_spectral_features(np.array([0.5]))
        elapsed = (time.time() - start_time) * 1000
        print(f"   ✅ Completed in {elapsed:.1f}ms")
    except Exception as e:
        print(f"   💀 Failed: {e}")
    
    # Test 3: Basic spectrum analyzer with normal audio
    print("3. Testing AdvancedSpectrumAnalyzer with normal audio...")
    try:
        analyzer = AdvancedSpectrumAnalyzer()
        start_time = time.time()
        normal_audio = np.random.randn(16000) * 0.3
        features = analyzer.extract_spectral_features(normal_audio)
        elapsed = (time.time() - start_time) * 1000
        print(f"   ✅ Completed in {elapsed:.1f}ms")
    except Exception as e:
        print(f"   💀 Failed: {e}")
    
    # Test 4: Test harmonic analysis specifically
    print("4. Testing harmonic analysis with edge cases...")
    try:
        analyzer = AdvancedSpectrumAnalyzer()
        
        # Test with DC signal
        start_time = time.time()
        dc_signal = np.ones(16000)
        features = analyzer._compute_harmonic_features(dc_signal)
        elapsed = (time.time() - start_time) * 1000
        print(f"   DC signal: {elapsed:.1f}ms")
        
        # Test with constant signal
        start_time = time.time()
        constant_signal = np.zeros(16000)
        features = analyzer._compute_harmonic_features(constant_signal)
        elapsed = (time.time() - start_time) * 1000
        print(f"   Zero signal: {elapsed:.1f}ms")
        
    except Exception as e:
        print(f"   💀 Harmonic analysis failed: {e}")
    
    # Test 5: Test FFT features
    print("5. Testing FFT features with edge cases...")
    try:
        analyzer = AdvancedSpectrumAnalyzer()
        
        # Test with very short signal
        start_time = time.time()
        short_signal = np.array([1.0, -1.0])
        features = analyzer._compute_fft_features(short_signal)
        elapsed = (time.time() - start_time) * 1000
        print(f"   Short signal FFT: {elapsed:.1f}ms")
        
    except Exception as e:
        print(f"   💀 FFT analysis failed: {e}")
    
    # Test 6: Test autocorrelation function
    print("6. Testing fundamental frequency estimation...")
    try:
        analyzer = AdvancedSpectrumAnalyzer()
        
        # Test with problematic signals
        test_signals = [
            ("Empty", np.array([])),
            ("Single", np.array([0.5])),
            ("Two samples", np.array([1.0, -1.0])),
            ("All zeros", np.zeros(100)),
            ("All ones", np.ones(100)),
        ]
        
        for name, signal in test_signals:
            start_time = time.time()
            freq = analyzer._estimate_fundamental_frequency(signal)
            elapsed = (time.time() - start_time) * 1000
            print(f"   {name}: {elapsed:.1f}ms (freq: {freq})")
            
    except Exception as e:
        print(f"   💀 Fundamental freq estimation failed: {e}")

def test_specific_hanging_cases():
    """Test specific cases that might cause hanging"""
    print("\n💀 TESTING SPECIFIC HANGING CASES")
    print("=" * 40)
    
    analyzer = AdvancedSpectrumAnalyzer()
    
    # Test cases that might cause infinite loops or excessive computation
    hanging_cases = [
        ("Infinite values", np.full(1000, np.inf)),
        ("NaN values", np.full(1000, np.nan)),
        ("Mixed inf/nan", np.array([np.inf, np.nan, 0] * 333 + [0])),
        ("Very large values", np.full(1000, 1e20)),
        ("Very small values", np.full(1000, 1e-20)),
    ]
    
    for name, signal in hanging_cases:
        print(f"   Testing {name}...")
        start_time = time.time()
        
        try:
            # Set a timeout for each test
            features = analyzer.extract_spectral_features(signal)
            elapsed = (time.time() - start_time) * 1000
            
            if elapsed > 1000:  # More than 1 second
                print(f"     💀 SLOW: {elapsed:.1f}ms")
            else:
                print(f"     ✅ {elapsed:.1f}ms")
                
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            print(f"     💀 Failed after {elapsed:.1f}ms: {e}")

if __name__ == "__main__":
    print("💀💀💀 SIMPLE VULNERABILITY TEST 💀💀💀")
    
    test_individual_components()
    test_specific_hanging_cases()
    
    print(f"\n🎯 SIMPLE TEST COMPLETE")
    print("Check for any components that took >1000ms")