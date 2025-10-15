#!/usr/bin/env python3
"""
💀 Isolated Test - Find the exact hanging component
==================================================
"""

import numpy as np
import time
import signal
import sys

def timeout_handler(signum, frame):
    print("💀 TIMEOUT: Test exceeded time limit!")
    sys.exit(1)

def test_with_timeout(test_func, timeout_sec=5):
    """Run test with timeout"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    try:
        result = test_func()
        signal.alarm(0)  # Cancel alarm
        return result
    except Exception as e:
        signal.alarm(0)
        raise e

def test_basic_imports():
    """Test if imports cause hanging"""
    print("1. Testing imports...")
    
    def import_test():
        from advanced_spectrum_analysis import AdvancedSpectrumAnalyzer
        return "Import successful"
    
    result = test_with_timeout(import_test, 10)
    print(f"   ✅ {result}")

def test_empty_array():
    """Test empty array processing"""
    print("2. Testing empty array...")
    
    def empty_test():
        from advanced_spectrum_analysis import AdvancedSpectrumAnalyzer
        analyzer = AdvancedSpectrumAnalyzer()
        features = analyzer.extract_spectral_features(np.array([]))
        return f"Empty array processed"
    
    try:
        result = test_with_timeout(empty_test, 5)
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   💀 Failed: {e}")

def test_single_sample():
    """Test single sample"""
    print("3. Testing single sample...")
    
    def single_test():
        from advanced_spectrum_analysis import AdvancedSpectrumAnalyzer
        analyzer = AdvancedSpectrumAnalyzer()
        features = analyzer.extract_spectral_features(np.array([0.5]))
        return f"Single sample processed"
    
    try:
        result = test_with_timeout(single_test, 5)
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   💀 Failed: {e}")

def test_librosa_hanging():
    """Test if librosa causes hanging"""
    print("4. Testing librosa operations...")
    
    def librosa_test():
        import librosa
        # Test mel-spectrogram on edge case
        audio = np.array([0.5, -0.5])  # Minimal audio
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_mels=63,
            n_fft=1024,
            hop_length=256
        )
        return f"Librosa processed {mel_spec.shape}"
    
    try:
        result = test_with_timeout(librosa_test, 10)
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   💀 Failed: {e}")

if __name__ == "__main__":
    print("💀 ISOLATED COMPONENT TESTING")
    print("=" * 40)
    
    try:
        test_basic_imports()
        test_empty_array()
        test_single_sample() 
        test_librosa_hanging()
        print("\n✅ All isolated tests completed")
    except KeyboardInterrupt:
        print("\n💀 Test interrupted")
    except Exception as e:
        print(f"\n💀 Test failed: {e}")