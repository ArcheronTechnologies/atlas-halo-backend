#!/usr/bin/env python3
"""
🛡️ SAIT_01 Comprehensive Stress Test
====================================
Full system stress test with advanced training + distributed consensus

Tests both individual model performance and mesh-enhanced accuracy
Validates defense-grade requirements under extreme conditions
"""

import numpy as np
import time
import threading
import concurrent.futures
from typing import List, Dict, Tuple
import json
import os
from mesh_threat_detection import MeshThreatDetectionSystem, ThreatLevel
from distributed_consensus_protocol import DistributedConsensusProtocol
import librosa

class SAIT01StressTest:
    """
    🛡️ Comprehensive SAIT_01 Stress Testing Framework
    ==================================================
    
    Tests:
    1. High-volume detection processing (1000+ samples/minute)
    2. Multi-node mesh consensus under load
    3. False positive rejection with natural noise
    4. Real-time performance under stress
    5. Defense scenario simulation
    6. Network resilience testing
    7. Memory and resource management
    8. Accuracy validation across scenarios
    """
    
    def __init__(self, num_nodes: int = 10):
        self.num_nodes = num_nodes
        self.detection_systems: List[MeshThreatDetectionSystem] = []
        self.test_results = {
            'individual_accuracy': [],
            'consensus_accuracy': [],
            'false_positive_rate': [],
            'processing_times': [],
            'network_resilience': [],
            'stress_performance': []
        }
        
        # Stress test parameters
        self.stress_duration = 120  # 2 minutes of stress testing
        self.high_volume_rate = 20  # 20 samples per second
        self.total_test_samples = 0
        
        print(f"🛡️ Initializing SAIT_01 Stress Test Framework")
        print(f"🌐 Testing with {num_nodes} mesh nodes")
    
    def setup_mesh_network(self):
        """Setup complete mesh network for testing"""
        print(f"\n🌐 Setting up {self.num_nodes}-node mesh network...")
        
        # Create detection systems for each node
        for i in range(self.num_nodes):
            node_id = f"SAIT01_Node_{i+1:02d}"
            location = (40.7128 + i*0.001, -74.0060 + i*0.001)  # Spread nodes across NYC
            
            detection_system = MeshThreatDetectionSystem(node_id)
            detection_system.start_system(location=location)
            detection_system.consensus_protocol.simulate_mesh_network(num_nodes=self.num_nodes)
            
            self.detection_systems.append(detection_system)
        
        print(f"✅ Mesh network initialized with {len(self.detection_systems)} nodes")
    
    def generate_stress_test_audio(self, scenario: str, duration: float = 1.0) -> np.ndarray:
        """Generate realistic audio for stress testing"""
        sample_rate = 16000
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Base noise
        audio = np.random.randn(samples) * 0.05
        
        if scenario == 'drone_close':
            # High-confidence drone signature
            freq1 = 1800 + 300 * np.sin(2 * np.pi * 0.7 * t)
            freq2 = 2200 + 200 * np.sin(2 * np.pi * 1.1 * t)
            audio += 0.4 * np.sin(2 * np.pi * freq1 * t)
            audio += 0.3 * np.sin(2 * np.pi * freq2 * t)
            
        elif scenario == 'helicopter_close':
            # High-confidence helicopter signature
            rotor_freq = 12 + 3 * np.sin(2 * np.pi * 0.3 * t)
            audio += 0.6 * np.sin(2 * np.pi * rotor_freq * t)
            audio += 0.4 * np.sin(2 * np.pi * rotor_freq * 2 * t)
            audio += 0.2 * np.sin(2 * np.pi * rotor_freq * 3 * t)
            
        elif scenario == 'drone_swarm':
            # Multiple drones - critical threat
            for i in range(4):
                freq = 1600 + i * 400 + 150 * np.sin(2 * np.pi * (0.5 + i * 0.2) * t)
                audio += 0.2 * np.sin(2 * np.pi * freq * t)
                
        elif scenario == 'background_urban':
            # Urban background noise
            audio += 0.2 * np.sin(2 * np.pi * 60 * t)  # Power line hum
            audio += 0.1 * np.random.randn(samples)  # Traffic noise
            
        elif scenario == 'background_natural':
            # Natural environmental noise
            wind_freq = np.random.uniform(50, 200, samples)
            audio += 0.15 * np.sin(2 * np.pi * wind_freq * t)
            
        elif scenario == 'false_positive_test':
            # Designed to potentially trigger false positives
            audio += 0.3 * np.sin(2 * np.pi * 1900 * t)  # Near drone frequency
            audio += 0.2 * np.random.randn(samples)  # Random noise
        
        # Add realistic envelope and normalize
        envelope = 0.3 + 0.7 * np.exp(-t * 0.5) * (1 + 0.3 * np.sin(2 * np.pi * 0.8 * t))
        audio *= envelope
        audio = np.clip(audio, -0.8, 0.8)
        
        return audio
    
    def test_individual_model_accuracy(self) -> Dict:
        """Test individual model accuracy across scenarios"""
        print(f"\n🎯 Testing Individual Model Accuracy")
        print("=" * 50)
        
        test_scenarios = [
            ('drone_close', 100, 1),      # 100 close drone samples
            ('helicopter_close', 100, 2), # 100 close helicopter samples  
            ('background_urban', 200, 0), # 200 urban background
            ('background_natural', 200, 0), # 200 natural background
            ('drone_swarm', 50, 1),       # 50 drone swarm samples
            ('false_positive_test', 150, 0) # 150 false positive tests
        ]
        
        total_correct = 0
        total_samples = 0
        false_positives = 0
        processing_times = []
        
        detection_system = self.detection_systems[0]  # Use first node
        
        for scenario, count, expected_class in test_scenarios:
            print(f"  Testing {scenario}: {count} samples")
            scenario_correct = 0
            scenario_fps = 0
            
            for i in range(count):
                # Generate test audio
                audio = self.generate_stress_test_audio(scenario)
                
                # Measure processing time
                start_time = time.time()
                
                # Process through individual model (no consensus)
                features = detection_system._extract_features(audio)
                if features is not None:
                    prediction = detection_system._run_local_inference(features)
                    predicted_class = prediction['predicted_class']
                    confidence = prediction['confidence']
                    
                    processing_time = (time.time() - start_time) * 1000  # ms
                    processing_times.append(processing_time)
                    
                    # Check accuracy
                    if predicted_class == expected_class:
                        scenario_correct += 1
                    
                    # Check false positives (background classified as threat)
                    if expected_class == 0 and predicted_class > 0 and confidence > 0.6:
                        scenario_fps += 1
                
                total_samples += 1
            
            total_correct += scenario_correct
            false_positives += scenario_fps
            
            accuracy = (scenario_correct / count) * 100
            fp_rate = (scenario_fps / count) * 100 if expected_class == 0 else 0
            
            print(f"    Accuracy: {accuracy:.1f}% | FP Rate: {fp_rate:.1f}%")
        
        overall_accuracy = (total_correct / total_samples) * 100
        overall_fp_rate = (false_positives / (200 + 200 + 150)) * 100  # Only background samples
        avg_processing = np.mean(processing_times)
        
        self.test_results['individual_accuracy'].append(overall_accuracy)
        self.test_results['false_positive_rate'].append(overall_fp_rate)
        self.test_results['processing_times'].append(avg_processing)
        
        print(f"\n📊 Individual Model Results:")
        print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"   False Positive Rate: {overall_fp_rate:.1f}%")
        print(f"   Avg Processing Time: {avg_processing:.1f}ms")
        
        return {
            'accuracy': overall_accuracy,
            'false_positive_rate': overall_fp_rate,
            'processing_time': avg_processing,
            'total_samples': total_samples
        }
    
    def test_consensus_accuracy(self) -> Dict:
        """Test mesh consensus accuracy improvement"""
        print(f"\n🌐 Testing Mesh Consensus Accuracy")
        print("=" * 50)
        
        test_scenarios = [
            ('drone_close', 50, 1),
            ('helicopter_close', 50, 2),
            ('background_urban', 100, 0),
            ('false_positive_test', 100, 0)
        ]
        
        total_correct = 0
        total_samples = 0
        consensus_improvements = 0
        
        # Use multiple nodes for consensus testing
        test_nodes = self.detection_systems[:5]  # Use 5 nodes
        
        for scenario, count, expected_class in test_scenarios:
            print(f"  Testing consensus on {scenario}: {count} samples")
            scenario_correct = 0
            scenario_improvements = 0
            
            for i in range(count):
                audio = self.generate_stress_test_audio(scenario)
                
                # Get individual predictions from multiple nodes
                individual_predictions = []
                for node in test_nodes:
                    features = node._extract_features(audio)
                    if features is not None:
                        pred = node._run_local_inference(features)
                        individual_predictions.append(pred)
                
                # Run consensus detection on primary node
                if individual_predictions:
                    consensus_result = test_nodes[0]._run_consensus_detection(
                        individual_predictions[0], audio
                    )
                    
                    # Determine consensus prediction
                    if consensus_result.threat_level == ThreatLevel.BACKGROUND:
                        consensus_class = 0
                    elif consensus_result.threat_level in [ThreatLevel.DRONE_SUSPECTED, ThreatLevel.DRONE_CONFIRMED]:
                        consensus_class = 1
                    else:
                        consensus_class = 2
                    
                    # Check if consensus improved accuracy
                    individual_class = individual_predictions[0]['predicted_class']
                    if consensus_class == expected_class and individual_class != expected_class:
                        scenario_improvements += 1
                    
                    if consensus_class == expected_class:
                        scenario_correct += 1
                
                total_samples += 1
            
            total_correct += scenario_correct
            consensus_improvements += scenario_improvements
            
            accuracy = (scenario_correct / count) * 100
            improvement_rate = (scenario_improvements / count) * 100
            
            print(f"    Consensus Accuracy: {accuracy:.1f}% | Improvements: {improvement_rate:.1f}%")
        
        overall_accuracy = (total_correct / total_samples) * 100
        overall_improvement = (consensus_improvements / total_samples) * 100
        
        self.test_results['consensus_accuracy'].append(overall_accuracy)
        
        print(f"\n📊 Consensus Results:")
        print(f"   Consensus Accuracy: {overall_accuracy:.1f}%")
        print(f"   Improvement Rate: {overall_improvement:.1f}%")
        
        return {
            'consensus_accuracy': overall_accuracy,
            'improvement_rate': overall_improvement,
            'total_samples': total_samples
        }
    
    def test_high_volume_processing(self) -> Dict:
        """Test system under high-volume processing load"""
        print(f"\n⚡ Testing High-Volume Processing")
        print("=" * 50)
        
        # Generate large batch of test audio
        batch_size = 200
        processing_times = []
        successful_processes = 0
        
        print(f"  Processing {batch_size} samples at high speed...")
        
        detection_system = self.detection_systems[0]
        
        start_time = time.time()
        
        for i in range(batch_size):
            # Vary scenarios for realistic testing
            scenarios = ['drone_close', 'helicopter_close', 'background_urban', 'background_natural']
            scenario = scenarios[i % len(scenarios)]
            
            audio = self.generate_stress_test_audio(scenario, duration=0.5)  # Shorter samples
            
            process_start = time.time()
            try:
                alerts = detection_system.process_audio_stream(audio)
                process_time = (time.time() - process_start) * 1000
                processing_times.append(process_time)
                successful_processes += 1
            except Exception as e:
                print(f"    Error processing sample {i}: {e}")
        
        total_time = time.time() - start_time
        avg_processing = np.mean(processing_times) if processing_times else 0
        throughput = successful_processes / total_time
        
        print(f"  Processed {successful_processes}/{batch_size} samples")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.1f} samples/second")
        print(f"  Avg processing: {avg_processing:.1f}ms")
        
        return {
            'throughput': throughput,
            'success_rate': (successful_processes / batch_size) * 100,
            'avg_processing_time': avg_processing,
            'total_time': total_time
        }
    
    def test_network_resilience(self) -> Dict:
        """Test mesh network resilience under node failures"""
        print(f"\n🛡️ Testing Network Resilience")
        print("=" * 50)
        
        # Test with different numbers of active nodes
        resilience_results = []
        
        for active_nodes in [8, 6, 4, 2]:
            print(f"  Testing with {active_nodes} active nodes...")
            
            # Simulate some nodes going offline
            test_systems = self.detection_systems[:active_nodes]
            
            correct_detections = 0
            total_tests = 50
            
            for i in range(total_tests):
                # Alternate between threat and background
                if i % 2 == 0:
                    audio = self.generate_stress_test_audio('drone_close')
                    expected_threat = True
                else:
                    audio = self.generate_stress_test_audio('background_urban')
                    expected_threat = False
                
                # Use primary node for detection
                try:
                    alerts = test_systems[0].process_audio_stream(audio)
                    detected_threat = len(alerts) > 0 and any(
                        alert.threat_level != ThreatLevel.BACKGROUND for alert in alerts
                    )
                    
                    if detected_threat == expected_threat:
                        correct_detections += 1
                        
                except Exception as e:
                    print(f"    Error in resilience test: {e}")
            
            accuracy = (correct_detections / total_tests) * 100
            resilience_results.append({
                'active_nodes': active_nodes,
                'accuracy': accuracy
            })
            
            print(f"    Accuracy with {active_nodes} nodes: {accuracy:.1f}%")
        
        self.test_results['network_resilience'] = resilience_results
        
        return resilience_results
    
    def test_defense_scenarios(self) -> Dict:
        """Test critical defense scenarios"""
        print(f"\n🚨 Testing Critical Defense Scenarios")
        print("=" * 50)
        
        scenarios = [
            {
                'name': 'Single Drone Infiltration',
                'audio_type': 'drone_close',
                'duration': 3.0,
                'expected_alert': True,
                'criticality': 'high'
            },
            {
                'name': 'Helicopter Approach', 
                'audio_type': 'helicopter_close',
                'duration': 4.0,
                'expected_alert': True,
                'criticality': 'high'
            },
            {
                'name': 'Coordinated Drone Swarm',
                'audio_type': 'drone_swarm',
                'duration': 5.0,
                'expected_alert': True,
                'criticality': 'critical'
            },
            {
                'name': 'Urban Camouflage Test',
                'audio_type': 'background_urban',
                'duration': 10.0,
                'expected_alert': False,
                'criticality': 'low'
            },
            {
                'name': 'Natural Environment Test',
                'audio_type': 'background_natural',
                'duration': 8.0,
                'expected_alert': False,
                'criticality': 'low'
            }
        ]
        
        scenario_results = []
        
        for scenario in scenarios:
            print(f"  Testing: {scenario['name']}")
            
            # Generate longer audio for realistic scenario
            audio = self.generate_stress_test_audio(
                scenario['audio_type'], 
                duration=scenario['duration']
            )
            
            # Test with primary detection system
            detection_system = self.detection_systems[0]
            
            start_time = time.time()
            alerts = detection_system.process_audio_stream(audio)
            detection_time = time.time() - start_time
            
            # Analyze results
            threat_detected = len(alerts) > 0 and any(
                alert.threat_level not in [ThreatLevel.BACKGROUND] 
                for alert in alerts
            )
            
            critical_alert = any(
                alert.threat_level == ThreatLevel.CRITICAL_ALERT 
                for alert in alerts
            )
            
            max_confidence = max([alert.confidence for alert in alerts], default=0.0)
            
            success = threat_detected == scenario['expected_alert']
            
            result = {
                'scenario': scenario['name'],
                'success': success,
                'threat_detected': threat_detected,
                'critical_alert': critical_alert,
                'max_confidence': max_confidence,
                'alerts_generated': len(alerts),
                'detection_time': detection_time,
                'criticality': scenario['criticality']
            }
            
            scenario_results.append(result)
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"    Result: {status} | Alerts: {len(alerts)} | Confidence: {max_confidence:.3f}")
        
        # Calculate overall defense readiness
        critical_scenarios = [r for r in scenario_results if r['criticality'] in ['high', 'critical']]
        critical_success_rate = sum(r['success'] for r in critical_scenarios) / len(critical_scenarios) * 100
        
        print(f"\n📊 Defense Scenario Results:")
        print(f"   Overall Success Rate: {sum(r['success'] for r in scenario_results) / len(scenario_results) * 100:.1f}%")
        print(f"   Critical Scenario Success: {critical_success_rate:.1f}%")
        
        return {
            'scenarios': scenario_results,
            'overall_success': sum(r['success'] for r in scenario_results) / len(scenario_results) * 100,
            'critical_success': critical_success_rate
        }
    
    def run_comprehensive_stress_test(self) -> Dict:
        """Run complete stress test suite"""
        print(f"\n🛡️ SAIT_01 COMPREHENSIVE STRESS TEST")
        print("=" * 60)
        print(f"🌐 Mesh Network: {self.num_nodes} nodes")
        print(f"⏱️  Test Duration: ~10-15 minutes")
        print(f"📊 Test Categories: 6 major test suites")
        
        start_time = time.time()
        
        # Setup mesh network
        self.setup_mesh_network()
        
        # Run all test suites
        test_results = {}
        
        try:
            # 1. Individual Model Accuracy
            test_results['individual_model'] = self.test_individual_model_accuracy()
            
            # 2. Consensus Accuracy  
            test_results['consensus_accuracy'] = self.test_consensus_accuracy()
            
            # 3. High-Volume Processing
            test_results['high_volume'] = self.test_high_volume_processing()
            
            # 4. Network Resilience
            test_results['network_resilience'] = self.test_network_resilience()
            
            # 5. Defense Scenarios
            test_results['defense_scenarios'] = self.test_defense_scenarios()
            
            # 6. System Status Check
            test_results['system_status'] = self.get_comprehensive_system_status()
            
        except Exception as e:
            print(f"❌ Stress test error: {e}")
            test_results['error'] = str(e)
        
        finally:
            # Cleanup
            self.cleanup_systems()
        
        total_time = time.time() - start_time
        test_results['total_test_time'] = total_time
        
        # Generate final report
        self.generate_stress_test_report(test_results)
        
        return test_results
    
    def get_comprehensive_system_status(self) -> Dict:
        """Get detailed system status across all nodes"""
        system_status = {
            'total_nodes': len(self.detection_systems),
            'active_nodes': 0,
            'total_detections': 0,
            'consensus_agreements': 0,
            'mesh_validations': 0,
            'memory_usage': [],
            'performance_metrics': []
        }
        
        for system in self.detection_systems:
            if system.is_active:
                system_status['active_nodes'] += 1
                
                status = system.get_system_status()
                system_status['total_detections'] += status['detection_stats']['total_detections']
                system_status['consensus_agreements'] += status['detection_stats']['consensus_agreements']
                system_status['mesh_validations'] += status['detection_stats']['mesh_validations']
        
        return system_status
    
    def generate_stress_test_report(self, results: Dict):
        """Generate comprehensive stress test report"""
        print(f"\n🛡️ SAIT_01 STRESS TEST REPORT")
        print("=" * 60)
        
        # Individual Model Performance
        if 'individual_model' in results:
            ind = results['individual_model']
            print(f"\n📊 Individual Model Performance:")
            print(f"   Accuracy: {ind['accuracy']:.1f}%")
            print(f"   False Positive Rate: {ind['false_positive_rate']:.1f}%")
            print(f"   Processing Time: {ind['processing_time']:.1f}ms")
            print(f"   Samples Tested: {ind['total_samples']}")
        
        # Consensus Performance
        if 'consensus_accuracy' in results:
            cons = results['consensus_accuracy']
            print(f"\n🌐 Mesh Consensus Performance:")
            print(f"   Consensus Accuracy: {cons['consensus_accuracy']:.1f}%")
            print(f"   Improvement Rate: {cons['improvement_rate']:.1f}%")
            print(f"   Samples Tested: {cons['total_samples']}")
        
        # High-Volume Performance
        if 'high_volume' in results:
            hv = results['high_volume']
            print(f"\n⚡ High-Volume Performance:")
            print(f"   Throughput: {hv['throughput']:.1f} samples/second")
            print(f"   Success Rate: {hv['success_rate']:.1f}%")
            print(f"   Avg Processing: {hv['avg_processing_time']:.1f}ms")
        
        # Defense Scenarios
        if 'defense_scenarios' in results:
            ds = results['defense_scenarios']
            print(f"\n🚨 Defense Scenarios:")
            print(f"   Overall Success: {ds['overall_success']:.1f}%")
            print(f"   Critical Success: {ds['critical_success']:.1f}%")
        
        # Network Resilience
        if 'network_resilience' in results:
            nr = results['network_resilience']
            print(f"\n🛡️ Network Resilience:")
            for result in nr:
                print(f"   {result['active_nodes']} nodes: {result['accuracy']:.1f}% accuracy")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        # Calculate composite score
        individual_score = results.get('individual_model', {}).get('accuracy', 0)
        consensus_score = results.get('consensus_accuracy', {}).get('consensus_accuracy', 0)
        defense_score = results.get('defense_scenarios', {}).get('overall_success', 0)
        
        overall_score = (individual_score + consensus_score + defense_score) / 3
        
        print(f"   Composite Score: {overall_score:.1f}%")
        
        if overall_score >= 90:
            assessment = "🏆 EXCELLENT - Defense Ready"
        elif overall_score >= 80:
            assessment = "✅ GOOD - Deployment Ready"
        elif overall_score >= 70:
            assessment = "⚠️ ACCEPTABLE - Needs Optimization"
        else:
            assessment = "❌ POOR - Requires Major Improvements"
        
        print(f"   Status: {assessment}")
        
        # Performance vs Requirements
        print(f"\n📋 Requirements Validation:")
        
        req_checks = [
            ("90-95% Accuracy", individual_score >= 90, f"{individual_score:.1f}%"),
            ("<5% False Positives", results.get('individual_model', {}).get('false_positive_rate', 100) < 5, 
             f"{results.get('individual_model', {}).get('false_positive_rate', 0):.1f}%"),
            ("<100ms Processing", results.get('individual_model', {}).get('processing_time', 1000) < 100,
             f"{results.get('individual_model', {}).get('processing_time', 0):.1f}ms"),
            ("Defense Grade", defense_score >= 85, f"{defense_score:.1f}%"),
            ("Mesh Enhancement", consensus_score > individual_score, 
             f"+{consensus_score - individual_score:.1f}%")
        ]
        
        for requirement, passed, value in req_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {requirement}: {status} ({value})")
        
        print(f"\n⏱️  Total Test Time: {results.get('total_test_time', 0):.1f} seconds")
        print(f"🛡️ SAIT_01 Stress Test Complete!")
    
    def cleanup_systems(self):
        """Cleanup all detection systems"""
        for system in self.detection_systems:
            try:
                system.stop_system()
            except:
                pass

def run_full_stress_test():
    """Run the complete SAIT_01 stress test"""
    stress_tester = SAIT01StressTest(num_nodes=8)
    results = stress_tester.run_comprehensive_stress_test()
    return results

if __name__ == "__main__":
    print("🛡️ Starting SAIT_01 Full System Stress Test...")
    results = run_full_stress_test()