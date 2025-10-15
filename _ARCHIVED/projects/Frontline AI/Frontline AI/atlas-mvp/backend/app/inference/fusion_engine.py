import asyncio
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import threading

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock modules for class definition
    class MockTorch:
        class Tensor:
            def __init__(self, *args, **kwargs):
                pass
            def unsqueeze(self, *args):
                return self
            def to(self, *args):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return []
            def tolist(self):
                return []
            def item(self):
                return 0.5
        
        @staticmethod
        def FloatTensor(*args):
            return MockTorch.Tensor()
        
        @staticmethod
        def ones(*args):
            return MockTorch.Tensor()
        
        @staticmethod
        def stack(*args, **kwargs):
            return MockTorch.Tensor()
        
        @staticmethod
        def cat(*args, **kwargs):
            return MockTorch.Tensor()
        
        @staticmethod
        def argmax(*args):
            return MockTorch.Tensor()
        
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False
            
            class cuda:
                @staticmethod
                def is_available():
                    return False
    
    class MockNN:
        class Module:
            def __init__(self):
                pass
            def to(self, device):
                return self
            def eval(self):
                return self
        
        class Sequential:
            def __init__(self, *args):
                pass
        
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        
        class ReLU:
            def __init__(self):
                pass
        
        class Dropout:
            def __init__(self, *args):
                pass
        
        class MultiheadAttention:
            def __init__(self, *args, **kwargs):
                pass
        
        class Parameter:
            def __init__(self, tensor):
                pass
    
    class MockF:
        @staticmethod
        def softmax(*args, **kwargs):
            return MockTorch.Tensor()
    
    torch = MockTorch()
    nn = MockNN()
    F = MockF()
    logging.warning("PyTorch not available for fusion engine")

logger = logging.getLogger(__name__)

class MultimodalFusionModel(nn.Module):
    """Multimodal fusion model combining video and audio features"""
    
    def __init__(self, video_feature_dim=2048, audio_feature_dim=512, num_classes=5):
        super().__init__()
        
        # Video branch
        self.video_encoder = nn.Sequential(
            nn.Linear(video_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Audio branch  
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for modality fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(512, 256),  # video + audio features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(128, num_classes)
        
        # Modality weight learning
        self.modality_weights = nn.Parameter(torch.ones(2))  # video, audio
        
    def forward(self, video_features, audio_features):
        # Encode features
        video_encoded = self.video_encoder(video_features)  # [batch, 256]
        audio_encoded = self.audio_encoder(audio_features)  # [batch, 256]
        
        # Stack for attention (sequence length = 2, one for each modality)
        modality_features = torch.stack([video_encoded, audio_encoded], dim=1)  # [batch, 2, 256]
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            modality_features, modality_features, modality_features
        )  # [batch, 2, 256]
        
        # Apply learned modality weights
        weights = F.softmax(self.modality_weights, dim=0)
        weighted_features = attended_features * weights.unsqueeze(0).unsqueeze(2)
        
        # Fuse modalities
        fused_features = torch.cat([
            weighted_features[:, 0, :],  # weighted video
            weighted_features[:, 1, :]   # weighted audio
        ], dim=1)  # [batch, 512]
        
        # Final fusion and classification
        fusion_output = self.fusion_layer(fused_features)  # [batch, 128]
        logits = self.classifier(fusion_output)  # [batch, num_classes]
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'modality_weights': weights,
            'video_features': video_encoded,
            'audio_features': audio_encoded,
            'fused_features': fusion_output
        }

class TemporalBuffer:
    """Buffer to maintain temporal alignment between modalities"""
    
    def __init__(self, max_age_seconds=2.0, max_items=50):
        self.max_age = timedelta(seconds=max_age_seconds)
        self.max_items = max_items
        self.video_buffer = []
        self.audio_buffer = []
        self.lock = threading.Lock()
    
    def add_video_detection(self, detection: Dict):
        """Add video detection to buffer"""
        with self.lock:
            detection['timestamp_obj'] = datetime.fromisoformat(detection['timestamp'].replace('Z', '+00:00'))
            self.video_buffer.append(detection)
            self._cleanup_buffer(self.video_buffer)
    
    def add_audio_detection(self, detection: Dict):
        """Add audio detection to buffer"""
        with self.lock:
            detection['timestamp_obj'] = datetime.fromisoformat(detection['timestamp'].replace('Z', '+00:00'))
            self.audio_buffer.append(detection)
            self._cleanup_buffer(self.audio_buffer)
    
    def _cleanup_buffer(self, buffer: List):
        """Remove old items from buffer"""
        now = datetime.utcnow().replace(tzinfo=None)
        
        # Remove items older than max_age
        buffer[:] = [item for item in buffer if 
                    (now - item['timestamp_obj'].replace(tzinfo=None)) < self.max_age]
        
        # Keep only most recent items
        if len(buffer) > self.max_items:
            buffer[:] = buffer[-self.max_items:]
    
    def get_temporal_pairs(self, time_window_ms=500) -> List[Tuple[Dict, Dict]]:
        """Get video-audio pairs that are temporally close"""
        pairs = []
        time_window = timedelta(milliseconds=time_window_ms)
        
        with self.lock:
            for video_detection in self.video_buffer:
                video_time = video_detection['timestamp_obj'].replace(tzinfo=None)
                
                # Find audio detections within time window
                for audio_detection in self.audio_buffer:
                    audio_time = audio_detection['timestamp_obj'].replace(tzinfo=None)
                    
                    if abs(video_time - audio_time) <= time_window:
                        pairs.append((video_detection, audio_detection))
        
        return pairs
    
    def get_recent_detections(self) -> Dict[str, List]:
        """Get recent detections from both modalities"""
        with self.lock:
            return {
                'video': list(self.video_buffer),
                'audio': list(self.audio_buffer)
            }

class FusionEngine:
    """Main fusion engine coordinating multimodal processing"""
    
    def __init__(self):
        self.fusion_model = None
        self.temporal_buffer = TemporalBuffer()
        self.device = 'cpu'
        self.is_running = False
        self.fusion_thread = None
        
        # Fusion classes (combined from video and audio)
        self.fusion_classes = [
            "person_activity",      # person + person_speaking
            "vehicle_activity",     # car/truck + vehicle_engine  
            "security_threat",      # weapon + gunshot/breaking_glass
            "emergency_event",      # any detection + alarm
            "normal_activity"       # baseline class
        ]
        
        self.fusion_thresholds = {
            "person_activity": 0.6,
            "vehicle_activity": 0.7,
            "security_threat": 0.9,
            "emergency_event": 0.8,
            "normal_activity": 0.3
        }
        
        self.broadcast_callbacks = {}
        
    async def initialize(self):
        """Initialize fusion engine"""
        if not TORCH_AVAILABLE:
            logger.warning("Fusion engine requires PyTorch - using simple rule-based fusion")
            return True
            
        try:
            # Initialize fusion model
            self.fusion_model = MultimodalFusionModel(
                video_feature_dim=2048,  # From video encoder
                audio_feature_dim=512,   # From audio encoder
                num_classes=len(self.fusion_classes)
            )
            
            # Set device
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using Apple Metal Performance Shaders (MPS) for fusion")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA GPU for fusion")
            else:
                self.device = "cpu"
                logger.info("Using CPU for fusion")
                
            self.fusion_model.to(self.device)
            
            # Load pretrained fusion model if available
            await self._load_fusion_model()
            
            logger.info("Fusion engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize fusion engine: {e}")
            return False
    
    async def _load_fusion_model(self):
        """Load pretrained fusion model"""
        try:
            # In a real implementation, load pretrained weights
            logger.info("Using randomly initialized fusion model (no pretrained weights)")
            
        except Exception as e:
            logger.warning(f"Could not load fusion model: {e}")
    
    def set_broadcast_callbacks(self, callbacks: Dict):
        """Set callbacks for broadcasting fusion results"""
        self.broadcast_callbacks = callbacks
    
    async def start_fusion_processing(self):
        """Start fusion processing"""
        if self.is_running:
            logger.warning("Fusion processing already running")
            return
            
        self.is_running = True
        
        # Start fusion processing thread
        self.fusion_thread = threading.Thread(target=self._fusion_processing_loop)
        self.fusion_thread.daemon = True
        self.fusion_thread.start()
        
        logger.info("Fusion processing started")
    
    def _fusion_processing_loop(self):
        """Main fusion processing loop"""
        logger.info("Fusion processing loop started")
        
        while self.is_running:
            try:
                # Get temporal pairs for fusion
                temporal_pairs = self.temporal_buffer.get_temporal_pairs()
                
                if temporal_pairs:
                    # Process each temporal pair
                    for video_detection, audio_detection in temporal_pairs:
                        fusion_result = self._fuse_detections(video_detection, audio_detection)
                        
                        if fusion_result:
                            # Handle fusion result asynchronously
                            asyncio.run_coroutine_threadsafe(
                                self._handle_fusion_result(fusion_result),
                                asyncio.get_event_loop()
                            )
                
                # Also check for rule-based fusion opportunities
                recent_detections = self.temporal_buffer.get_recent_detections()
                rule_based_results = self._rule_based_fusion(recent_detections)
                
                for result in rule_based_results:
                    asyncio.run_coroutine_threadsafe(
                        self._handle_fusion_result(result),
                        asyncio.get_event_loop()
                    )
                
                # Sleep to avoid overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in fusion processing loop: {e}")
                time.sleep(1.0)
        
        logger.info("Fusion processing loop ended")
    
    def _fuse_detections(self, video_detection: Dict, audio_detection: Dict) -> Optional[Dict]:
        """Fuse video and audio detections using neural model or rules"""
        try:
            if self.fusion_model and TORCH_AVAILABLE:
                return self._neural_fusion(video_detection, audio_detection)
            else:
                return self._rule_based_single_fusion(video_detection, audio_detection)
                
        except Exception as e:
            logger.error(f"Error in fusion: {e}")
            return None
    
    def _neural_fusion(self, video_detection: Dict, audio_detection: Dict) -> Optional[Dict]:
        """Perform neural fusion using the trained model"""
        try:
            # Extract/create features (simplified for demo)
            video_features = self._extract_video_features(video_detection)
            audio_features = self._extract_audio_features(audio_detection)
            
            if video_features is None or audio_features is None:
                return None
            
            # Convert to tensors
            video_tensor = torch.FloatTensor(video_features).unsqueeze(0).to(self.device)
            audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
            
            # Run fusion model
            self.fusion_model.eval()
            with torch.no_grad():
                outputs = self.fusion_model(video_tensor, audio_tensor)
                probabilities = F.softmax(outputs['logits'], dim=1)[0]
            
            # Find best fusion class
            best_class_idx = torch.argmax(probabilities).item()
            best_class = self.fusion_classes[best_class_idx]
            best_confidence = probabilities[best_class_idx].item()
            
            # Check threshold
            threshold = self.fusion_thresholds.get(best_class, 0.5)
            if best_confidence >= threshold:
                return {
                    'class_name': best_class,
                    'confidence': best_confidence,
                    'video_detection': video_detection,
                    'audio_detection': audio_detection,
                    'fusion_type': 'neural',
                    'attention_weights': outputs['attention_weights'].cpu().numpy().tolist(),
                    'modality_weights': outputs['modality_weights'].cpu().numpy().tolist(),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Neural fusion error: {e}")
            return None
    
    def _rule_based_single_fusion(self, video_detection: Dict, audio_detection: Dict) -> Optional[Dict]:
        """Rule-based fusion for a single video-audio pair"""
        video_class = video_detection['class_name']
        audio_class = audio_detection['class_name']
        
        # Fusion rules
        fusion_rules = {
            ('person', 'person_speaking'): 'person_activity',
            ('car', 'vehicle_engine'): 'vehicle_activity',
            ('truck', 'vehicle_engine'): 'vehicle_activity',
            ('weapon', 'gunshot'): 'security_threat',
            ('weapon', 'breaking_glass'): 'security_threat',
        }
        
        # Check for exact matches
        rule_key = (video_class, audio_class)
        if rule_key in fusion_rules:
            fusion_class = fusion_rules[rule_key]
            
            # Combine confidences (weighted average)
            video_weight = 0.6  # Slightly favor video
            audio_weight = 0.4
            combined_confidence = (
                video_detection['confidence'] * video_weight + 
                audio_detection['confidence'] * audio_weight
            )
            
            threshold = self.fusion_thresholds.get(fusion_class, 0.5)
            if combined_confidence >= threshold:
                return {
                    'class_name': fusion_class,
                    'confidence': combined_confidence,
                    'video_detection': video_detection,
                    'audio_detection': audio_detection,
                    'fusion_type': 'rule_based',
                    'rule_applied': f"{video_class} + {audio_class} -> {fusion_class}",
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return None
    
    def _rule_based_fusion(self, recent_detections: Dict) -> List[Dict]:
        """Rule-based fusion looking at recent detection patterns"""
        results = []
        
        try:
            video_detections = recent_detections.get('video', [])
            audio_detections = recent_detections.get('audio', [])
            
            # Check for emergency patterns
            has_alarm_audio = any(d['class_name'] == 'alarm' for d in audio_detections[-5:])
            has_recent_video = len(video_detections) > 0 and \
                              (datetime.utcnow() - video_detections[-1]['timestamp_obj'].replace(tzinfo=None)).total_seconds() < 3
            
            if has_alarm_audio and has_recent_video:
                results.append({
                    'class_name': 'emergency_event',
                    'confidence': 0.85,
                    'fusion_type': 'rule_based_pattern',
                    'pattern': 'alarm_with_activity',
                    'contributing_detections': {
                        'video': [d for d in video_detections[-3:]],
                        'audio': [d for d in audio_detections if d['class_name'] == 'alarm'][-1:]
                    },
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Check for security threat patterns  
            recent_weapon_video = [d for d in video_detections[-10:] if d['class_name'] == 'weapon']
            recent_threat_audio = [d for d in audio_detections[-10:] if d['class_name'] in ['gunshot', 'breaking_glass']]
            
            if recent_weapon_video and recent_threat_audio:
                # Check if they're within reasonable time window
                weapon_time = recent_weapon_video[-1]['timestamp_obj'].replace(tzinfo=None)
                threat_time = recent_threat_audio[-1]['timestamp_obj'].replace(tzinfo=None)
                
                if abs((weapon_time - threat_time).total_seconds()) < 5.0:
                    results.append({
                        'class_name': 'security_threat',
                        'confidence': 0.95,
                        'fusion_type': 'rule_based_pattern', 
                        'pattern': 'weapon_with_threat_sound',
                        'contributing_detections': {
                            'video': recent_weapon_video[-1:],
                            'audio': recent_threat_audio[-1:]
                        },
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Rule-based fusion error: {e}")
        
        return results
    
    def _extract_video_features(self, video_detection: Dict) -> Optional[np.ndarray]:
        """Extract features from video detection (simplified)"""
        try:
            # In a real implementation, this would extract features from the video frame
            # For now, create synthetic features based on detection
            class_encoding = {
                'person': [1, 0, 0, 0, 0],
                'car': [0, 1, 0, 0, 0], 
                'truck': [0, 0, 1, 0, 0],
                'motorcycle': [0, 0, 0, 1, 0],
                'weapon': [0, 0, 0, 0, 1]
            }
            
            base_features = class_encoding.get(video_detection['class_name'], [0, 0, 0, 0, 0])
            confidence_features = [video_detection['confidence']] * 5
            
            # Pad to expected dimension (2048)
            features = np.array(base_features + confidence_features + [0.0] * (2048 - 10))
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Video feature extraction error: {e}")
            return None
    
    def _extract_audio_features(self, audio_detection: Dict) -> Optional[np.ndarray]:
        """Extract features from audio detection (simplified)"""
        try:
            # In a real implementation, this would use actual audio features
            class_encoding = {
                'person_speaking': [1, 0, 0, 0, 0],
                'vehicle_engine': [0, 1, 0, 0, 0],
                'gunshot': [0, 0, 1, 0, 0],
                'breaking_glass': [0, 0, 0, 1, 0],
                'alarm': [0, 0, 0, 0, 1]
            }
            
            base_features = class_encoding.get(audio_detection['class_name'], [0, 0, 0, 0, 0])
            confidence_features = [audio_detection['confidence']] * 5
            
            # Pad to expected dimension (512)
            features = np.array(base_features + confidence_features + [0.0] * (512 - 10))
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio feature extraction error: {e}")
            return None
    
    async def _handle_fusion_result(self, fusion_result: Dict):
        """Handle fusion result"""
        try:
            # Broadcast fusion result
            if 'broadcast_detection' in self.broadcast_callbacks:
                # Mark as fusion result
                fusion_result['modality'] = 'fusion'
                await self.broadcast_callbacks['broadcast_detection'](fusion_result)
            
            # Generate alerts for high-priority fusion events
            if fusion_result['class_name'] in ['security_threat', 'emergency_event']:
                alert_data = {
                    'type': 'fusion_alert',
                    'message': f"Multimodal event detected: {fusion_result['class_name']} ({fusion_result['confidence']:.2f})",
                    'timestamp': fusion_result['timestamp'],
                    'fusion_result': fusion_result
                }
                if 'broadcast_alert' in self.broadcast_callbacks:
                    await self.broadcast_callbacks['broadcast_alert'](alert_data)
                    
        except Exception as e:
            logger.error(f"Error handling fusion result: {e}")
    
    async def add_video_detection(self, detection: Dict):
        """Add video detection for fusion processing"""
        self.temporal_buffer.add_video_detection(detection)
    
    async def add_audio_detection(self, detection: Dict):
        """Add audio detection for fusion processing"""
        self.temporal_buffer.add_audio_detection(detection)
    
    async def stop_fusion_processing(self):
        """Stop fusion processing"""
        self.is_running = False
        
        if self.fusion_thread and self.fusion_thread.is_alive():
            self.fusion_thread.join(timeout=2.0)
        
        logger.info("Fusion processing stopped")
    
    async def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion engine statistics"""
        recent = self.temporal_buffer.get_recent_detections()
        
        return {
            'is_running': self.is_running,
            'model_available': self.fusion_model is not None,
            'device': self.device,
            'recent_video_count': len(recent['video']),
            'recent_audio_count': len(recent['audio']),
            'temporal_pairs_available': len(self.temporal_buffer.get_temporal_pairs()),
            'fusion_classes': self.fusion_classes
        }
    
    async def update_fusion_thresholds(self, thresholds: Dict[str, float]):
        """Update fusion detection thresholds"""
        self.fusion_thresholds.update(thresholds)
        logger.info(f"Updated fusion thresholds: {thresholds}")
    
    async def cleanup(self):
        """Clean up fusion engine"""
        await self.stop_fusion_processing()
        logger.info("Fusion engine cleaned up")