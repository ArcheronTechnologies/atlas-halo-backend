#!/usr/bin/env python3
"""
Adaptive Unified Defense Framework for SAIT_01
Phase 2.2 Implementation - Multi-Layer Defense Architecture

Implements 4-layer defense system:
- Layer 1: Input sanitization
- Layer 2: Feature-level detection  
- Layer 3: Output validation
- Layer 4: Network consensus

With dynamic defense level adjustment and real-time performance optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time

# Try to import torchaudio, fall back to basic implementations if not available
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not available, using simplified implementations")

try:
    from memory_based_defense import MemoryBasedDefenseDB, UniversalDefenseSystem
    MEMORY_DEFENSE_AVAILABLE = True
except ImportError:
    MEMORY_DEFENSE_AVAILABLE = False
    print("Warning: memory_based_defense not available, skipping memory defense")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DefenseLevel(Enum):
    """Defense levels for adaptive protection"""
    MINIMAL = 1      # Basic input validation only
    STANDARD = 2     # Input + feature detection
    ENHANCED = 3     # Full 3-layer defense
    MAXIMUM = 4      # All layers + network consensus
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class ThreatLevel(Enum):
    """Threat severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


@dataclass
class DefenseConfig:
    """Configuration for adaptive defense system"""
    defense_level: DefenseLevel = DefenseLevel.STANDARD
    input_sanitization_enabled: bool = True
    feature_detection_enabled: bool = True
    output_validation_enabled: bool = True
    network_consensus_enabled: bool = False
    adaptive_adjustment: bool = True
    performance_threshold_ms: float = 50.0  # Max processing time
    
    # Thresholds for each layer
    input_anomaly_threshold: float = 0.7
    feature_anomaly_threshold: float = 0.6
    output_confidence_threshold: float = 0.8
    consensus_agreement_threshold: float = 0.7


class InputSanitizationLayer:
    """Layer 1: Input sanitization and preprocessing"""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.sample_rate = 16000
        
        # Statistical baselines for anomaly detection
        self.noise_floor_db = -60.0
        self.max_amplitude = 0.95
        self.frequency_range = (50, 8000)  # Hz
        
        logger.info("Initialized InputSanitizationLayer")
    
    def sanitize_audio(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sanitize and validate input audio"""
        
        sanitization_result = {
            'is_valid': True,
            'issues_detected': [],
            'modifications_applied': [],
            'anomaly_score': 0.0
        }
        
        # Convert to numpy for processing
        audio_np = waveform.squeeze().numpy().copy()
        
        # 1. Amplitude clipping detection and correction
        if np.max(np.abs(audio_np)) > self.max_amplitude:
            sanitization_result['issues_detected'].append('amplitude_clipping')
            audio_np = np.clip(audio_np, -self.max_amplitude, self.max_amplitude)
            sanitization_result['modifications_applied'].append('amplitude_limiting')
            sanitization_result['anomaly_score'] += 0.3
        
        # 2. DC offset removal
        dc_offset = np.mean(audio_np)
        if abs(dc_offset) > 0.01:
            audio_np = audio_np - dc_offset
            sanitization_result['modifications_applied'].append('dc_offset_removal')
        
        # 3. Noise floor validation
        rms_energy = np.sqrt(np.mean(audio_np ** 2))
        energy_db = 20 * np.log10(rms_energy + 1e-8)
        
        if energy_db < self.noise_floor_db:
            sanitization_result['issues_detected'].append('below_noise_floor')
            sanitization_result['anomaly_score'] += 0.4
        
        # 4. Frequency content validation
        fft = np.fft.fft(audio_np)
        freqs = np.fft.fftfreq(len(audio_np), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        # Check for out-of-band energy
        valid_freq_mask = (np.abs(freqs) >= self.frequency_range[0]) & (np.abs(freqs) <= self.frequency_range[1])
        in_band_energy = np.sum(magnitude[valid_freq_mask])
        total_energy = np.sum(magnitude)
        
        if in_band_energy / (total_energy + 1e-8) < 0.5:
            sanitization_result['issues_detected'].append('out_of_band_energy')
            sanitization_result['anomaly_score'] += 0.3
        
        # 5. Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(audio_np)
        sanitization_result['anomaly_score'] += statistical_anomalies['anomaly_score']
        if statistical_anomalies['anomalies']:
            sanitization_result['issues_detected'].extend(statistical_anomalies['anomalies'])
        
        # Overall validation
        if sanitization_result['anomaly_score'] > self.config.input_anomaly_threshold:
            sanitization_result['is_valid'] = False
        
        # Convert back to tensor
        sanitized_waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        
        return sanitized_waveform, sanitization_result
    
    def _detect_statistical_anomalies(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect statistical anomalies in audio signal"""
        
        anomalies = []
        anomaly_score = 0.0
        
        # 1. Dynamic range check
        dynamic_range = np.max(audio) - np.min(audio)
        if dynamic_range < 0.001:  # Too small dynamic range
            anomalies.append('low_dynamic_range')
            anomaly_score += 0.2
        
        # 2. Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        zcr = zero_crossings / len(audio)
        if zcr < 0.001 or zcr > 0.8:  # Unusual ZCR
            anomalies.append('unusual_zero_crossing_rate')
            anomaly_score += 0.15
        
        # 3. Periodicity check (potential synthetic audio)
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        max_autocorr = np.max(autocorr[100:]) / (autocorr[0] + 1e-8)
        
        if max_autocorr > 0.95:  # Highly periodic
            anomalies.append('high_periodicity')
            anomaly_score += 0.25
        
        # 4. Spectral irregularities
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Check for sharp spectral peaks (potential jamming)
        magnitude_norm = magnitude / (np.max(magnitude) + 1e-8)
        sharp_peaks = np.sum(magnitude_norm > 0.9)
        
        if sharp_peaks > 3:
            anomalies.append('sharp_spectral_peaks')
            anomaly_score += 0.2
        
        return {
            'anomalies': anomalies,
            'anomaly_score': min(anomaly_score, 1.0)
        }


class FeatureDetectionLayer:
    """Layer 2: Feature-level attack detection"""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.feature_extractors = self._initialize_feature_extractors()
        
        # Adversarial pattern detection
        self.adversarial_signatures = self._load_adversarial_signatures()
        
        logger.info("Initialized FeatureDetectionLayer")
    
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize multiple feature extractors for robust detection"""
        
        extractors = {}
        
        if TORCHAUDIO_AVAILABLE:
            # MFCC extractor
            extractors['mfcc'] = {
                'transform': torch.nn.Sequential(
                    torchaudio.transforms.MelSpectrogram(
                        sample_rate=16000,
                        n_fft=512,
                        hop_length=256,
                        n_mels=40
                    ),
                    torchaudio.transforms.AmplitudeToDB()
                ),
                'feature_dim': 40
            }
            
            # Spectral features
            extractors['spectral'] = {
                'transform': torchaudio.transforms.Spectrogram(
                    n_fft=512,
                    hop_length=256
                ),
                'feature_dim': 257
            }
        else:
            # Simplified extractors without torchaudio
            extractors['mfcc'] = {
                'transform': None,  # Will use custom implementation
                'feature_dim': 40
            }
            
            extractors['spectral'] = {
                'transform': None,  # Will use custom implementation
                'feature_dim': 257
            }
        
        # Chromagram features
        extractors['chroma'] = {
            'transform': None,  # Custom implementation
            'feature_dim': 12
        }
        
        return extractors
    
    def _load_adversarial_signatures(self) -> Dict[str, np.ndarray]:
        """Load known adversarial attack signatures"""
        
        # These would be learned from training data
        # For now, create synthetic signatures for common attacks
        signatures = {}
        
        # FGSM-like attack signature (high-frequency noise)
        signatures['fgsm'] = np.random.normal(0, 0.1, 40)
        
        # PGD-like attack signature (structured perturbations)
        signatures['pgd'] = np.sin(np.linspace(0, 4*np.pi, 40)) * 0.2
        
        # C&W-like attack signature (smooth perturbations)
        signatures['cw'] = np.exp(-np.linspace(0, 5, 40)) * 0.3
        
        # Audio replay attack signature
        signatures['replay'] = np.ones(40) * 0.05  # Uniform low-level pattern
        
        return signatures
    
    def detect_adversarial_features(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """Detect adversarial patterns in audio features"""
        
        detection_result = {
            'is_adversarial': False,
            'attack_type': None,
            'confidence': 0.0,
            'feature_anomalies': {},
            'signature_matches': {}
        }
        
        # Extract multiple feature representations
        features = self._extract_multi_modal_features(waveform)
        
        # Analyze each feature type for adversarial patterns
        for feature_type, feature_data in features.items():
            anomaly_result = self._analyze_feature_anomalies(feature_data, feature_type)
            detection_result['feature_anomalies'][feature_type] = anomaly_result
        
        # Check against known adversarial signatures
        mfcc_features = features.get('mfcc', np.zeros(40))
        signature_matches = self._match_adversarial_signatures(mfcc_features)
        detection_result['signature_matches'] = signature_matches
        
        # Determine if adversarial attack detected
        max_anomaly_score = max([
            result['anomaly_score'] for result in detection_result['feature_anomalies'].values()
        ], default=0.0)
        
        max_signature_score = max([
            match['similarity'] for match in signature_matches.values()
        ], default=0.0)
        
        overall_score = max(max_anomaly_score, max_signature_score)
        
        if overall_score > self.config.feature_anomaly_threshold:
            detection_result['is_adversarial'] = True
            detection_result['confidence'] = overall_score
            
            # Determine most likely attack type
            if max_signature_score > max_anomaly_score:
                best_match = max(signature_matches.items(), key=lambda x: x[1]['similarity'])
                detection_result['attack_type'] = best_match[0]
            else:
                detection_result['attack_type'] = 'unknown_adversarial'
        
        return detection_result
    
    def _extract_multi_modal_features(self, waveform: torch.Tensor) -> Dict[str, np.ndarray]:
        """Extract multiple types of features for robust analysis"""
        
        features = {}
        
        if TORCHAUDIO_AVAILABLE:
            # MFCC features
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, hop_length=256, n_mels=40
            )(waveform)
            mfcc = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            features['mfcc'] = torch.mean(mfcc, dim=2).squeeze().numpy()
        else:
            # Simplified MFCC-like features using basic spectral analysis
            audio_np = waveform.squeeze().numpy()
            fft = np.fft.fft(audio_np)
            magnitude = np.abs(fft[:len(fft)//2])
            
            # Create mel-like filter bank (simplified)
            mel_features = []
            n_mels = 40
            for i in range(n_mels):
                start_bin = int(i * len(magnitude) / n_mels)
                end_bin = int((i + 1) * len(magnitude) / n_mels)
                mel_energy = np.mean(magnitude[start_bin:end_bin])
                mel_features.append(np.log(mel_energy + 1e-8))
            
            features['mfcc'] = np.array(mel_features)
        
        # Spectral centroid and bandwidth
        stft = torch.stft(waveform.squeeze(), n_fft=512, hop_length=256, return_complex=True)
        magnitude = torch.abs(stft)
        
        freqs = torch.fft.fftfreq(512, 1/16000)[:257]
        spectral_centroid = torch.sum(freqs.unsqueeze(1) * magnitude, dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)
        features['spectral_centroid'] = torch.mean(spectral_centroid).numpy()
        
        # Zero crossing rate
        zcr = torch.sum(torch.diff(torch.sign(waveform)) != 0).float() / waveform.shape[-1]
        features['zcr'] = zcr.numpy()
        
        # RMS energy
        rms = torch.sqrt(torch.mean(waveform ** 2))
        features['rms'] = rms.numpy()
        
        return features
    
    def _analyze_feature_anomalies(self, features: Union[np.ndarray, float], 
                                 feature_type: str) -> Dict[str, Any]:
        """Analyze features for anomalous patterns"""
        
        anomaly_result = {
            'anomaly_score': 0.0,
            'anomalies_detected': [],
            'feature_stats': {}
        }
        
        if isinstance(features, np.ndarray):
            # Multi-dimensional feature analysis
            if feature_type == 'mfcc':
                # Check for unusual MFCC patterns
                mfcc_mean = np.mean(features)
                mfcc_std = np.std(features)
                mfcc_range = np.max(features) - np.min(features)
                
                anomaly_result['feature_stats'] = {
                    'mean': float(mfcc_mean),
                    'std': float(mfcc_std),
                    'range': float(mfcc_range)
                }
                
                # Anomaly detection rules
                if mfcc_std < 0.1:  # Too uniform
                    anomaly_result['anomalies_detected'].append('low_variance')
                    anomaly_result['anomaly_score'] += 0.3
                
                if mfcc_range > 100:  # Too wide range
                    anomaly_result['anomalies_detected'].append('excessive_range')
                    anomaly_result['anomaly_score'] += 0.4
                
                # Check for adversarial-like high-frequency components
                if np.sum(np.abs(np.diff(features))) > 50:
                    anomaly_result['anomalies_detected'].append('high_frequency_noise')
                    anomaly_result['anomaly_score'] += 0.5
        
        else:
            # Scalar feature analysis
            if feature_type == 'spectral_centroid':
                if features < 100 or features > 8000:
                    anomaly_result['anomalies_detected'].append('unusual_spectral_centroid')
                    anomaly_result['anomaly_score'] += 0.3
            
            elif feature_type == 'zcr':
                if features < 0.001 or features > 0.8:
                    anomaly_result['anomalies_detected'].append('unusual_zcr')
                    anomaly_result['anomaly_score'] += 0.2
            
            elif feature_type == 'rms':
                if features < 0.001 or features > 0.9:
                    anomaly_result['anomalies_detected'].append('unusual_energy')
                    anomaly_result['anomaly_score'] += 0.2
        
        return anomaly_result
    
    def _match_adversarial_signatures(self, features: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Match features against known adversarial signatures"""
        
        matches = {}
        
        for signature_name, signature_pattern in self.adversarial_signatures.items():
            # Ensure same dimensionality
            if len(features) != len(signature_pattern):
                continue
            
            # Compute similarity metrics
            cosine_sim = np.dot(features, signature_pattern) / (
                np.linalg.norm(features) * np.linalg.norm(signature_pattern) + 1e-8
            )
            
            euclidean_dist = np.linalg.norm(features - signature_pattern)
            correlation = np.corrcoef(features, signature_pattern)[0, 1]
            
            # Combined similarity score
            similarity = (abs(cosine_sim) + abs(correlation)) / 2 - euclidean_dist / 100
            similarity = max(0, min(1, similarity))
            
            matches[signature_name] = {
                'similarity': similarity,
                'cosine_similarity': cosine_sim,
                'euclidean_distance': euclidean_dist,
                'correlation': correlation if not np.isnan(correlation) else 0.0
            }
        
        return matches


class OutputValidationLayer:
    """Layer 3: Output validation and confidence analysis"""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.confidence_models = self._initialize_confidence_models()
        
        logger.info("Initialized OutputValidationLayer")
    
    def _initialize_confidence_models(self) -> Dict[str, Any]:
        """Initialize confidence estimation models"""
        
        # Temperature scaling for confidence calibration
        models = {
            'temperature_scaling': {
                'temperature': 1.5,  # Learned parameter for calibration
                'enabled': True
            },
            'monte_carlo_dropout': {
                'num_samples': 10,
                'dropout_rate': 0.1,
                'enabled': True
            },
            'ensemble_confidence': {
                'num_models': 3,
                'enabled': True
            }
        }
        
        return models
    
    def validate_model_output(self, model_outputs: torch.Tensor, 
                            input_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Validate model outputs for adversarial manipulation"""
        
        validation_result = {
            'is_valid': True,
            'confidence_score': 0.0,
            'validation_issues': [],
            'calibrated_outputs': None,
            'uncertainty_metrics': {}
        }
        
        # Apply temperature scaling for calibration
        calibrated_outputs = self._apply_temperature_scaling(model_outputs)
        validation_result['calibrated_outputs'] = calibrated_outputs
        
        # Compute various uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(calibrated_outputs)
        validation_result['uncertainty_metrics'] = uncertainty_metrics
        
        # Validate output consistency
        consistency_result = self._validate_output_consistency(calibrated_outputs)
        
        # Check for adversarial output patterns
        adversarial_result = self._detect_adversarial_outputs(calibrated_outputs)
        
        # Combine validation results
        overall_confidence = self._combine_confidence_scores([
            consistency_result['confidence'],
            adversarial_result['confidence'],
            uncertainty_metrics['prediction_confidence']
        ])
        
        validation_result['confidence_score'] = overall_confidence
        
        if overall_confidence < self.config.output_confidence_threshold:
            validation_result['is_valid'] = False
            validation_result['validation_issues'].extend(consistency_result['issues'])
            validation_result['validation_issues'].extend(adversarial_result['issues'])
        
        return validation_result
    
    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling for output calibration"""
        
        temperature = self.confidence_models['temperature_scaling']['temperature']
        return logits / temperature
    
    def _compute_uncertainty_metrics(self, outputs: torch.Tensor) -> Dict[str, float]:
        """Compute various uncertainty metrics"""
        
        probs = torch.softmax(outputs, dim=-1)
        
        metrics = {}
        
        # Prediction confidence (max probability)
        max_prob = torch.max(probs, dim=-1)[0]
        metrics['prediction_confidence'] = float(torch.mean(max_prob))
        
        # Entropy-based uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        metrics['entropy'] = float(torch.mean(entropy))
        
        # Top-2 difference
        top2_probs = torch.topk(probs, 2, dim=-1)[0]
        top2_diff = top2_probs[:, 0] - top2_probs[:, 1]
        metrics['top2_difference'] = float(torch.mean(top2_diff))
        
        # Variance across predictions
        metrics['prediction_variance'] = float(torch.var(probs))
        
        return metrics
    
    def _validate_output_consistency(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Validate output consistency and detect anomalies"""
        
        result = {
            'confidence': 1.0,
            'issues': []
        }
        
        probs = torch.softmax(outputs, dim=-1)
        
        # Check for extreme probabilities (potential adversarial manipulation)
        max_prob = torch.max(probs)
        min_prob = torch.min(probs)
        
        if max_prob > 0.999:  # Too confident
            result['issues'].append('overconfident_prediction')
            result['confidence'] *= 0.7
        
        if min_prob > 0.01 and torch.numel(probs) > 10:  # Too uniform
            result['issues'].append('uniform_distribution')
            result['confidence'] *= 0.8
        
        # Check for NaN or infinite values
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            result['issues'].append('invalid_output_values')
            result['confidence'] = 0.0
        
        # Check probability sum (should be close to 1.0)
        prob_sum = torch.sum(probs)
        if abs(prob_sum - 1.0) > 0.01:
            result['issues'].append('probability_sum_error')
            result['confidence'] *= 0.9
        
        return result
    
    def _detect_adversarial_outputs(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Detect potential adversarial manipulation in outputs"""
        
        result = {
            'confidence': 1.0,
            'issues': []
        }
        
        probs = torch.softmax(outputs, dim=-1)
        
        # Check for adversarial patterns in probability distribution
        # 1. Unusual sharpness
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        if entropy < 0.1:  # Too sharp
            result['issues'].append('unusually_sharp_distribution')
            result['confidence'] *= 0.8
        
        # 2. Check for gradient-based attack signatures
        # High-frequency oscillations in logits
        if len(outputs.shape) > 1 and outputs.shape[-1] > 2:
            logit_diff = torch.diff(outputs, dim=-1)
            if torch.std(logit_diff) > 5.0:  # High variance in adjacent logits
                result['issues'].append('high_logit_variance')
                result['confidence'] *= 0.7
        
        # 3. Check for known adversarial probability patterns
        sorted_probs = torch.sort(probs, descending=True)[0]
        if len(sorted_probs) >= 3:
            # Check for "ladder" pattern (common in some attacks)
            diff1 = sorted_probs[0] - sorted_probs[1]
            diff2 = sorted_probs[1] - sorted_probs[2]
            if diff1 > 0.3 and diff2 > 0.3 and abs(diff1 - diff2) < 0.05:
                result['issues'].append('ladder_probability_pattern')
                result['confidence'] *= 0.6
        
        return result
    
    def _combine_confidence_scores(self, scores: List[float]) -> float:
        """Combine multiple confidence scores"""
        
        # Use geometric mean for conservative combination
        valid_scores = [s for s in scores if s > 0]
        if not valid_scores:
            return 0.0
        
        geometric_mean = np.prod(valid_scores) ** (1.0 / len(valid_scores))
        return float(geometric_mean)


class NetworkConsensusLayer:
    """Layer 4: Network consensus validation"""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.node_trust_scores = {}  # Track trust scores for network nodes
        
        logger.info("Initialized NetworkConsensusLayer")
    
    def validate_network_consensus(self, local_prediction: Dict[str, Any], 
                                 network_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate prediction against network consensus"""
        
        consensus_result = {
            'consensus_reached': False,
            'agreement_score': 0.0,
            'conflicting_nodes': [],
            'trusted_prediction': local_prediction,
            'network_confidence': 0.0
        }
        
        if not network_predictions:
            # No network data available
            consensus_result['consensus_reached'] = True
            consensus_result['agreement_score'] = 1.0
            consensus_result['network_confidence'] = local_prediction.get('confidence', 0.0)
            return consensus_result
        
        # Analyze network predictions
        all_predictions = [local_prediction] + network_predictions
        
        # Calculate agreement metrics
        agreement_metrics = self._calculate_agreement_metrics(all_predictions)
        consensus_result.update(agreement_metrics)
        
        # Determine if consensus is reached
        if agreement_metrics['agreement_score'] >= self.config.consensus_agreement_threshold:
            consensus_result['consensus_reached'] = True
            
            # Create consensus prediction
            consensus_prediction = self._create_consensus_prediction(all_predictions)
            consensus_result['trusted_prediction'] = consensus_prediction
        
        else:
            # Identify conflicting nodes
            consensus_result['conflicting_nodes'] = self._identify_conflicting_nodes(
                local_prediction, network_predictions
            )
        
        return consensus_result
    
    def _calculate_agreement_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate agreement metrics across network predictions"""
        
        if len(predictions) < 2:
            return {'agreement_score': 1.0, 'prediction_variance': 0.0}
        
        # Extract prediction classes and confidences
        classes = []
        confidences = []
        
        for pred in predictions:
            classes.append(pred.get('predicted_class', 0))
            confidences.append(pred.get('confidence', 0.0))
        
        # Calculate class agreement
        from collections import Counter
        class_counts = Counter(classes)
        most_common_class, most_common_count = class_counts.most_common(1)[0]
        class_agreement = most_common_count / len(classes)
        
        # Calculate confidence variance
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        
        # Combined agreement score
        agreement_score = class_agreement * (1.0 - min(confidence_variance, 0.5))
        
        return {
            'agreement_score': agreement_score,
            'class_agreement': class_agreement,
            'confidence_variance': confidence_variance,
            'most_common_class': most_common_class,
            'prediction_distribution': dict(class_counts)
        }
    
    def _create_consensus_prediction(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create consensus prediction from network inputs"""
        
        # Weighted voting based on node trust scores
        class_votes = {}
        total_weight = 0.0
        
        for i, pred in enumerate(predictions):
            node_id = pred.get('node_id', f'node_{i}')
            trust_score = self.node_trust_scores.get(node_id, 1.0)
            
            predicted_class = pred.get('predicted_class', 0)
            confidence = pred.get('confidence', 0.0)
            
            # Weight = trust_score * confidence
            weight = trust_score * confidence
            
            if predicted_class not in class_votes:
                class_votes[predicted_class] = 0.0
            
            class_votes[predicted_class] += weight
            total_weight += weight
        
        # Normalize votes
        if total_weight > 0:
            for class_id in class_votes:
                class_votes[class_id] /= total_weight
        
        # Select consensus class
        consensus_class = max(class_votes.keys(), key=lambda k: class_votes[k])
        consensus_confidence = class_votes[consensus_class]
        
        return {
            'predicted_class': consensus_class,
            'confidence': consensus_confidence,
            'vote_distribution': class_votes,
            'consensus_method': 'weighted_voting'
        }
    
    def _identify_conflicting_nodes(self, local_prediction: Dict[str, Any], 
                                  network_predictions: List[Dict[str, Any]]) -> List[str]:
        """Identify nodes with conflicting predictions"""
        
        local_class = local_prediction.get('predicted_class', 0)
        conflicting_nodes = []
        
        for pred in network_predictions:
            node_id = pred.get('node_id', 'unknown')
            pred_class = pred.get('predicted_class', 0)
            
            if pred_class != local_class:
                conflicting_nodes.append(node_id)
        
        return conflicting_nodes
    
    def update_node_trust(self, node_id: str, performance_score: float):
        """Update trust score for a network node"""
        
        if node_id not in self.node_trust_scores:
            self.node_trust_scores[node_id] = 1.0
        
        # Exponential moving average for trust score
        alpha = 0.1  # Learning rate
        self.node_trust_scores[node_id] = (
            alpha * performance_score + (1 - alpha) * self.node_trust_scores[node_id]
        )
        
        # Clamp to [0.1, 1.0] range
        self.node_trust_scores[node_id] = max(0.1, min(1.0, self.node_trust_scores[node_id]))


class AdaptiveUnifiedDefenseFramework:
    """Main adaptive defense framework coordinating all layers"""
    
    def __init__(self, config: DefenseConfig, memory_defense_db: Optional[MemoryBasedDefenseDB] = None):
        self.config = config
        
        # Initialize defense layers
        self.input_layer = InputSanitizationLayer(config)
        self.feature_layer = FeatureDetectionLayer(config)
        self.output_layer = OutputValidationLayer(config)
        self.consensus_layer = NetworkConsensusLayer(config)
        
        # Initialize memory-based defense if provided
        self.memory_defense = None
        if memory_defense_db and MEMORY_DEFENSE_AVAILABLE:
            self.memory_defense = UniversalDefenseSystem(memory_defense_db)
        
        # Performance monitoring
        self.performance_history = []
        self.threat_history = []
        
        logger.info(f"Initialized AdaptiveUnifiedDefenseFramework with {config.defense_level} level")
    
    def defend_audio_input(self, waveform: torch.Tensor, 
                          model_inference_func: callable,
                          network_predictions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Comprehensive defense pipeline for audio input"""
        
        start_time = time.time()
        
        defense_result = {
            'is_safe': True,
            'threat_level': ThreatLevel.NONE,
            'threats_detected': [],
            'defense_layers_results': {},
            'final_prediction': None,
            'processing_time_ms': 0.0,
            'defense_level_used': self.config.defense_level
        }
        
        try:
            # Layer 1: Input Sanitization
            if self.config.input_sanitization_enabled:
                sanitized_waveform, sanitization_result = self.input_layer.sanitize_audio(waveform)
                defense_result['defense_layers_results']['input_sanitization'] = sanitization_result
                
                if not sanitization_result['is_valid']:
                    defense_result['is_safe'] = False
                    defense_result['threats_detected'].append('input_anomaly')
                    defense_result['threat_level'] = ThreatLevel.MEDIUM
            else:
                sanitized_waveform = waveform
            
            # Memory-based defense check
            if self.memory_defense:
                memory_validation = self.memory_defense.validate_audio_input(sanitized_waveform)
                defense_result['defense_layers_results']['memory_defense'] = memory_validation
                
                if not memory_validation['is_valid']:
                    defense_result['is_safe'] = False
                    defense_result['threats_detected'].extend(memory_validation['threats_detected'])
                    if ThreatLevel.HIGH > defense_result['threat_level']:
                        defense_result['threat_level'] = ThreatLevel.HIGH
            
            # Layer 2: Feature-level Detection
            if self.config.feature_detection_enabled:
                feature_result = self.feature_layer.detect_adversarial_features(sanitized_waveform)
                defense_result['defense_layers_results']['feature_detection'] = feature_result
                
                if feature_result['is_adversarial']:
                    defense_result['is_safe'] = False
                    defense_result['threats_detected'].append(f"adversarial_attack_{feature_result['attack_type']}")
                    defense_result['threat_level'] = ThreatLevel.HIGH
            
            # Run model inference on sanitized input
            if defense_result['is_safe'] or self.config.defense_level == DefenseLevel.MINIMAL:
                model_outputs = model_inference_func(sanitized_waveform)
                
                # Layer 3: Output Validation
                if self.config.output_validation_enabled:
                    output_validation = self.output_layer.validate_model_output(model_outputs)
                    defense_result['defense_layers_results']['output_validation'] = output_validation
                    
                    if not output_validation['is_valid']:
                        defense_result['is_safe'] = False
                        defense_result['threats_detected'].append('output_manipulation')
                        defense_result['threat_level'] = ThreatLevel.MEDIUM
                    
                    # Use calibrated outputs
                    calibrated_outputs = output_validation.get('calibrated_outputs', model_outputs)
                else:
                    calibrated_outputs = model_outputs
                
                # Convert outputs to prediction format
                probs = torch.softmax(calibrated_outputs, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = torch.max(probs, dim=-1)[0].item()
                
                local_prediction = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'node_id': 'local'
                }
                
                # Layer 4: Network Consensus
                if self.config.network_consensus_enabled and network_predictions:
                    consensus_result = self.consensus_layer.validate_network_consensus(
                        local_prediction, network_predictions
                    )
                    defense_result['defense_layers_results']['network_consensus'] = consensus_result
                    
                    if not consensus_result['consensus_reached']:
                        defense_result['threats_detected'].append('network_consensus_failure')
                        if ThreatLevel.MEDIUM > defense_result['threat_level']:
                            defense_result['threat_level'] = ThreatLevel.MEDIUM
                    
                    defense_result['final_prediction'] = consensus_result['trusted_prediction']
                else:
                    defense_result['final_prediction'] = local_prediction
            
        except Exception as e:
            logger.error(f"Error in defense pipeline: {e}")
            defense_result['is_safe'] = False
            defense_result['threat_level'] = ThreatLevel.CRITICAL
            defense_result['threats_detected'].append('defense_system_error')
        
        # Record performance
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        defense_result['processing_time_ms'] = processing_time
        
        # Adaptive defense level adjustment
        if self.config.adaptive_adjustment:
            self._adjust_defense_level(defense_result)
        
        # Update performance history
        self.performance_history.append(processing_time)
        self.threat_history.append(defense_result['threat_level'])
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            self.threat_history = self.threat_history[-100:]
        
        return defense_result
    
    def _adjust_defense_level(self, defense_result: Dict[str, Any]):
        """Dynamically adjust defense level based on threat detection and performance"""
        
        current_threat_level = defense_result['threat_level']
        processing_time = defense_result['processing_time_ms']
        
        # Increase defense level if threats detected
        if current_threat_level >= ThreatLevel.HIGH and self.config.defense_level < DefenseLevel.MAXIMUM:
            self.config.defense_level = DefenseLevel(min(self.config.defense_level.value + 1, 4))
            logger.info(f"Increased defense level to {self.config.defense_level}")
        
        # Decrease defense level if performance is poor and no recent threats
        elif (processing_time > self.config.performance_threshold_ms and 
              len(self.threat_history) >= 10 and 
              all(t <= ThreatLevel.LOW for t in self.threat_history[-10:])):
            
            if self.config.defense_level > DefenseLevel.MINIMAL:
                self.config.defense_level = DefenseLevel(self.config.defense_level.value - 1)
                logger.info(f"Decreased defense level to {self.config.defense_level} for performance")
        
        # Update layer configurations based on new defense level
        self._update_layer_configurations()
    
    def _update_layer_configurations(self):
        """Update layer enable/disable based on current defense level"""
        
        level = self.config.defense_level
        
        self.config.input_sanitization_enabled = level >= DefenseLevel.MINIMAL
        self.config.feature_detection_enabled = level >= DefenseLevel.STANDARD
        self.config.output_validation_enabled = level >= DefenseLevel.ENHANCED
        self.config.network_consensus_enabled = level >= DefenseLevel.MAXIMUM
    
    def get_defense_statistics(self) -> Dict[str, Any]:
        """Get comprehensive defense system statistics"""
        
        stats = {
            'current_defense_level': self.config.defense_level.name,
            'total_processed': len(self.performance_history),
            'average_processing_time_ms': np.mean(self.performance_history) if self.performance_history else 0.0,
            'threat_detection_rate': sum(1 for t in self.threat_history if t > ThreatLevel.NONE) / len(self.threat_history) if self.threat_history else 0.0,
            'recent_threat_levels': [t.name for t in self.threat_history[-10:]],
            'performance_within_threshold': sum(1 for t in self.performance_history if t <= self.config.performance_threshold_ms) / len(self.performance_history) if self.performance_history else 1.0
        }
        
        return stats


def main():
    """Test adaptive unified defense framework"""
    
    logger.info("🛡️ TESTING ADAPTIVE UNIFIED DEFENSE FRAMEWORK")
    logger.info("=" * 60)
    
    # Create defense configuration
    config = DefenseConfig(
        defense_level=DefenseLevel.ENHANCED,
        adaptive_adjustment=True,
        performance_threshold_ms=50.0
    )
    
    # Initialize defense framework
    defense_framework = AdaptiveUnifiedDefenseFramework(config)
    
    # Create mock model inference function
    def mock_model_inference(waveform):
        # Simulate model inference returning logits
        return torch.randn(1, 27)  # 27 classes
    
    # Test with synthetic audio
    test_waveform = torch.randn(1, 16000)  # 1 second of audio
    
    logger.info("🧪 Testing defense framework with synthetic audio")
    
    # Test defense pipeline
    defense_result = defense_framework.defend_audio_input(
        test_waveform, 
        mock_model_inference
    )
    
    logger.info(f"Defense Result:")
    logger.info(f"  Is safe: {defense_result['is_safe']}")
    logger.info(f"  Threat level: {defense_result['threat_level'].name}")
    logger.info(f"  Threats detected: {defense_result['threats_detected']}")
    logger.info(f"  Processing time: {defense_result['processing_time_ms']:.2f} ms")
    logger.info(f"  Defense level used: {defense_result['defense_level_used'].name}")
    
    # Test with multiple inputs to see adaptive behavior
    logger.info("\n🔄 Testing adaptive defense level adjustment")
    
    for i in range(5):
        # Create increasingly noisy input to trigger defense adaptation
        noisy_waveform = test_waveform + torch.randn_like(test_waveform) * (i * 0.1)
        
        result = defense_framework.defend_audio_input(noisy_waveform, mock_model_inference)
        logger.info(f"  Test {i+1}: Threat level = {result['threat_level'].name}, "
                   f"Defense level = {result['defense_level_used'].name}")
    
    # Get final statistics
    stats = defense_framework.get_defense_statistics()
    logger.info(f"\n📊 Defense Statistics:")
    logger.info(f"  Current defense level: {stats['current_defense_level']}")
    logger.info(f"  Average processing time: {stats['average_processing_time_ms']:.2f} ms")
    logger.info(f"  Threat detection rate: {stats['threat_detection_rate']:.1%}")
    logger.info(f"  Performance within threshold: {stats['performance_within_threshold']:.1%}")
    
    logger.info("\n✅ Adaptive Unified Defense Framework Test Complete")
    logger.info("🎯 Phase 2.2 Implementation Ready")


if __name__ == "__main__":
    main()