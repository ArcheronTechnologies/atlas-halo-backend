import asyncio
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import queue

try:
    import pyaudio
    import librosa
    import torch
    import torch.nn as nn
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    # Create mock nn module for class definition
    class MockNN:
        class Module:
            def __init__(self):
                pass
            def __call__(self, *args, **kwargs):
                pass
        
        class Conv2d:
            def __init__(self, *args, **kwargs):
                pass
        
        class MaxPool2d:
            def __init__(self, *args, **kwargs):
                pass
                
        class AdaptiveAvgPool2d:
            def __init__(self, *args, **kwargs):
                pass
        
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        
        class Dropout:
            def __init__(self, *args, **kwargs):
                pass
        
        class ReLU:
            def __init__(self, *args, **kwargs):
                pass
    
    nn = MockNN()
    torch = None
    pyaudio = None
    librosa = None
    logging.warning("Audio processing not available. Install pyaudio, librosa, torch for audio functionality.")

logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Extract audio features for classification"""
    
    def __init__(self, sample_rate=16000, n_mels=128, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.update_sample_rate(sample_rate)

    def update_sample_rate(self, sample_rate: int):
        """Update cached parameters when sample rate changes"""
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)  # 25ms
        self.hop_length = int(0.01 * sample_rate)     # 10ms
        
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features"""
        try:
            # Ensure audio is mono and correct sample rate
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            features = {}
            
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.frame_length * 2
            )
            features['mfcc'] = mfccs
            
            # Mel-scale spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.frame_length * 2
            )
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec)
            
            # Spectral features
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            
            # Zero crossing rate (useful for distinguishing voiced/unvoiced)
            features['zcr'] = librosa.feature.zero_crossing_rate(
                audio_data, hop_length=self.hop_length
            )
            
            # Chroma features (pitch content)
            features['chroma'] = librosa.feature.chroma_stft(
                y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
            )
            
            # RMS energy
            features['rms'] = librosa.feature.rms(
                y=audio_data, hop_length=self.hop_length
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}

class SimpleAudioClassifier(nn.Module):
    """Simple CNN for audio classification"""
    
    def __init__(self, input_size=128, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, 1, mel_bins, time_frames)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class AudioEngine:
    """Audio capture and processing engine"""
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16 if AUDIO_AVAILABLE else None
        
        self.audio_interface = None
        self.stream = None
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        
        # Audio processing components
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        self.audio_classifier = None
        self.device = 'cpu'
        
        # Audio classes for detection
        self.audio_classes = ["person_speaking", "vehicle_engine", "gunshot", "breaking_glass", "alarm"]
        self.audio_thresholds = {
            "person_speaking": 0.6,
            "vehicle_engine": 0.7,
            "gunshot": 0.8,
            "breaking_glass": 0.7,
            "alarm": 0.7
        }
        
        # Buffer for audio data (store last 3 seconds)
        self.audio_buffer_size = sample_rate * 3  # 3 seconds
        self.audio_buffer = np.zeros(self.audio_buffer_size, dtype=np.float32)
        self.buffer_index = 0
        
        self.broadcast_callbacks = {}
        
    async def initialize(self):
        """Initialize audio engine"""
        if not AUDIO_AVAILABLE:
            logger.warning("Audio processing not available - missing dependencies")
            return False
            
        try:
            self.audio_interface = pyaudio.PyAudio()
            
            # Adjust sample rate to match default input device for stability
            try:
                default_device = self.audio_interface.get_default_input_device_info()
                if default_device:
                    default_rate = int(default_device.get('defaultSampleRate', self.sample_rate))
                    if default_rate and abs(default_rate - self.sample_rate) > 1:
                        logger.info(f"Updating audio sample rate to {default_rate} Hz to match default input device")
                        self._configure_sample_rate(default_rate)
            except Exception as device_error:
                logger.warning(f"Unable to query default audio device: {device_error}")

            # Initialize audio classifier
            self.audio_classifier = SimpleAudioClassifier(
                input_size=128, 
                num_classes=len(self.audio_classes)
            )
            
            # Set device
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA GPU for audio")
            elif torch.backends.mps.is_available():
                logger.info("MPS available for audio, defaulting to CPU to avoid adaptive pooling limitations")
            else:
                logger.info("Using CPU for audio")
                
            self.audio_classifier.to(self.device)
            
            # Load pretrained weights if available
            await self._load_audio_model()
            
            logger.info("Audio engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio engine: {e}")
            return False
    
    async def _load_audio_model(self):
        """Load pretrained audio model or initialize random weights"""
        try:
            # In a real implementation, you would load a pretrained model
            # For now, we'll use random initialization
            logger.info("Using randomly initialized audio model (no pretrained weights)")
            
        except Exception as e:
            logger.warning(f"Could not load audio model: {e}")
    
    def set_broadcast_callbacks(self, callbacks: Dict):
        """Set callbacks for broadcasting audio detections"""
        self.broadcast_callbacks = callbacks
    
    async def start_audio_capture(self):
        """Start audio capture from microphone"""
        if not AUDIO_AVAILABLE:
            logger.warning("Cannot start audio capture - dependencies not available")
            return False
            
        if self.is_running:
            logger.warning("Audio capture already running")
            return True
            
        try:
            input_kwargs = {
                'format': self.format,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size,
                'stream_callback': self._audio_callback
            }

            try:
                default_device = self.audio_interface.get_default_input_device_info()
                if default_device:
                    input_kwargs['input_device_index'] = default_device.get('index')
            except Exception as device_error:
                logger.warning(f"Could not determine default microphone index: {device_error}")

            self.stream = self.audio_interface.open(**input_kwargs)
            
            self.is_running = True
            self.stream.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._audio_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info("Audio capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio data"""
        try:
            if not self.audio_queue.full():
                self.audio_queue.put(in_data)
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def _audio_processing_loop(self):
        """Process audio data in separate thread"""
        logger.info("Audio processing loop started")
        
        while self.is_running:
            try:
                # Get audio data from queue (timeout to allow checking is_running)
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Convert to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to circular buffer
                chunk_len = len(audio_np)
                if self.buffer_index + chunk_len <= self.audio_buffer_size:
                    self.audio_buffer[self.buffer_index:self.buffer_index + chunk_len] = audio_np
                    self.buffer_index += chunk_len
                else:
                    # Wrap around
                    remaining = self.audio_buffer_size - self.buffer_index
                    self.audio_buffer[self.buffer_index:] = audio_np[:remaining]
                    self.audio_buffer[:chunk_len - remaining] = audio_np[remaining:]
                    self.buffer_index = chunk_len - remaining
                
                # Process every few chunks to avoid overwhelming the system
                if self.audio_queue.qsize() < 5:  # Process when queue is relatively empty
                    detections = self._process_audio_buffer()
                    if detections:
                        # Use asyncio to handle detections in main thread
                        asyncio.run_coroutine_threadsafe(
                            self._handle_audio_detections(detections),
                            asyncio.get_event_loop()
                        )
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Audio processing loop ended")

    def _configure_sample_rate(self, sample_rate: int):
        """Apply a new sample rate across dependent components."""
        self.sample_rate = int(sample_rate)
        self.feature_extractor.update_sample_rate(self.sample_rate)
        self.audio_buffer_size = self.sample_rate * 3
        self.audio_buffer = np.zeros(self.audio_buffer_size, dtype=np.float32)
        self.buffer_index = 0
        logger.info(f"Audio engine configured for {self.sample_rate} Hz input")
    
    def _process_audio_buffer(self) -> List[Dict]:
        """Process current audio buffer for classification"""
        try:
            if self.audio_classifier is None:
                return self._mock_audio_detections()
            
            # Extract features from current buffer
            features = self.feature_extractor.extract_features(self.audio_buffer)
            if not features:
                return []
            
            # Use mel spectrogram for classification
            mel_spec = features.get('mel_spectrogram')
            if mel_spec is None:
                return []
            
            # Prepare input for model (batch_size=1, channels=1, height, width)
            input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            self.audio_classifier.eval()
            with torch.no_grad():
                outputs = self.audio_classifier(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Convert to detections
            detections = []
            for i, class_name in enumerate(self.audio_classes):
                confidence = probabilities[i].item()
                threshold = self.audio_thresholds.get(class_name, 0.5)
                
                if confidence >= threshold:
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'modality': 'audio',
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
            return []
    
    def _mock_audio_detections(self) -> List[Dict]:
        """Generate mock audio detections for testing"""
        import random
        
        detections = []
        if random.random() < 0.1:  # 10% chance of audio detection
            detections.append({
                'class_name': random.choice(['person_speaking', 'vehicle_engine']),
                'confidence': random.uniform(0.6, 0.9),
                'modality': 'audio',
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return detections
    
    async def _handle_audio_detections(self, detections: List[Dict]):
        """Handle audio detections"""
        for detection in detections:
            if 'broadcast_detection' in self.broadcast_callbacks:
                await self.broadcast_callbacks['broadcast_detection'](detection)
            
            # Generate alerts for high-priority audio events
            if detection['class_name'] in ['gunshot', 'breaking_glass', 'alarm']:
                alert_data = {
                    'type': 'audio_alert',
                    'message': f"Audio event detected: {detection['class_name']} ({detection['confidence']:.2f})",
                    'timestamp': detection['timestamp'],
                    'detection': detection
                }
                if 'broadcast_alert' in self.broadcast_callbacks:
                    await self.broadcast_callbacks['broadcast_alert'](alert_data)
    
    async def stop_audio_capture(self):
        """Stop audio capture"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Audio capture stopped")
    
    async def get_audio_stats(self) -> Dict[str, Any]:
        """Get audio engine statistics"""
        return {
            'is_running': self.is_running,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'buffer_size': self.audio_buffer_size,
            'queue_size': self.audio_queue.qsize() if AUDIO_AVAILABLE else 0,
            'available': AUDIO_AVAILABLE,
            'device': self.device
        }
    
    async def update_audio_thresholds(self, thresholds: Dict[str, float]):
        """Update audio detection thresholds"""
        self.audio_thresholds.update(thresholds)
        logger.info(f"Updated audio thresholds: {thresholds}")
    
    async def cleanup(self):
        """Clean up audio engine"""
        await self.stop_audio_capture()
        
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None
        
        logger.info("Audio engine cleaned up")
