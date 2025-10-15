import asyncio
import base64
import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import threading
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available. Install ultralytics for full functionality.")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("Core ML not available. Install coremltools for optimization.")

from app.database import db_manager
from app.inference.audio_engine import AudioEngine
from app.inference.fusion_engine import FusionEngine

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.camera = None
        self.is_running = False
        self.start_streaming = False
        self.main_loop = None
        self.detection_classes = ["person", "car", "truck", "motorcycle", "weapon"]
        self.confidence_thresholds = {
            "person": 0.5,
            "car": 0.5, 
            "truck": 0.5,
            "motorcycle": 0.5,
            "weapon": 0.3
        }
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.processing_times = []
        self.broadcast_callbacks = {}
        self.frame_skip_counter = 0
        
        # Multimodal components
        self.audio_engine = AudioEngine()
        self.fusion_engine = FusionEngine()
        self.multimodal_enabled = False
        
    async def initialize(self):
        try:
            await self._load_default_model()
            await self._initialize_camera()
            await self._load_config()
            
            # Initialize multimodal components
            audio_init = await self.audio_engine.initialize()
            fusion_init = await self.fusion_engine.initialize()
            
            if audio_init and fusion_init:
                self.multimodal_enabled = True
                logger.info("Multimodal processing enabled")
                
                # Set up callbacks for fusion
                self.audio_engine.set_broadcast_callbacks({
                    'broadcast_detection': self._handle_audio_detection
                })
                self.fusion_engine.set_broadcast_callbacks(self.broadcast_callbacks)
                
            else:
                self.multimodal_enabled = False
                logger.warning("Multimodal processing disabled - some components failed to initialize")
            
            logger.info("Inference engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            return False
    
    async def _load_default_model(self):
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available, using mock detection")
            return
        
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            active_model = await db_manager.get_active_model()
            if active_model and Path(active_model['path']).exists():
                model_path = active_model['path']
            else:
                model_path = "yolov8n.pt"
                logger.info("Downloading YOLOv8n model...")
            
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using Apple Metal Performance Shaders (MPS)")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA GPU")
            else:
                self.device = "cpu"
                logger.info("Using CPU")
                
            logger.info(f"Model loaded: {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    async def _initialize_camera(self):
        try:
            logger.info("Attempting to initialize camera...")
            # Always skip camera initialization for now to test processing loop
            logger.warning("Skipping camera initialization - using test frames only")
            self.camera = None
            return
            
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Could not read from camera")
                
            logger.info(f"Camera initialized: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            if self.camera:
                self.camera.release()
            self.camera = None
    
    async def _load_config(self):
        thresholds = await db_manager.get_config("confidence_thresholds", self.confidence_thresholds)
        if isinstance(thresholds, dict):
            self.confidence_thresholds.update(thresholds)
    
    def set_broadcast_callbacks(self, callbacks: Dict):
        self.broadcast_callbacks = callbacks
    
    async def start_processing(self):
        logger.error("=== START_PROCESSING CALLED - is_running: %s ===", self.is_running)
        if self.is_running:
            logger.error("Processing already running - RETURNING EARLY")
            return
        
        logger.error("Setting is_running to True")
        self.is_running = True
        logger.error("Starting video processing...")
        
        # Start multimodal processing if enabled
        if self.multimodal_enabled:
            logger.error("Starting multimodal processing")
            await self.audio_engine.start_audio_capture()
            await self.fusion_engine.start_fusion_processing()
            logger.error("Multimodal processing started")
        else:
            logger.error("Multimodal processing disabled, skipping")
        
        logger.error("Getting asyncio event loop for thread communication")
        self.main_loop = asyncio.get_event_loop()
        
        logger.error("Creating and starting processing thread directly")
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.error("Processing thread started - DONE")
        
    async def stop_processing(self):
        self.is_running = False
        self.start_streaming = False
        
        # Stop multimodal processing
        if self.multimodal_enabled:
            await self.audio_engine.stop_audio_capture()
            await self.fusion_engine.stop_fusion_processing()
            logger.info("Multimodal processing stopped")
        
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            try:
                self.processing_thread.join(timeout=5.0)
                if self.processing_thread.is_alive():
                    logger.warning("Processing thread didn't stop gracefully")
            except Exception as e:
                logger.error(f"Error stopping processing thread: {e}")
        logger.info("Video processing stopped")
    
    def _processing_loop(self):
        logger.info("Processing loop started - is_running: %s, start_streaming: %s", self.is_running, self.start_streaming)
        
        while self.is_running:
            try:
                # Force test frame mode for debugging
                logger.info("Loop iteration - generating test frame (is_running: %s, start_streaming: %s)", self.is_running, self.start_streaming)
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                # Add some test pattern with timestamp for debugging
                import time
                timestamp = int(time.time()) % 100
                frame[100:260, 200:440] = [0, 255, 0]  # Green rectangle
                cv2.putText(frame, f"TEST {timestamp}", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                start_time = time.time()
                
                if self.start_streaming and self.model:
                    detections = self._process_frame(frame)
                    annotated_frame = self._draw_detections(frame, detections)
                    
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._handle_detections(detections), 
                            self.main_loop
                        )
                    except Exception as e:
                        logger.debug(f"Detection handling error: {e}")
                else:
                    annotated_frame = frame
                
                if self.start_streaming:
                    # Only broadcast every 5th frame to reduce WebSocket load
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter >= 5:
                        self.frame_skip_counter = 0
                        frame_data = self._encode_frame(annotated_frame)
                        if frame_data and 'broadcast_frame' in self.broadcast_callbacks:
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    self.broadcast_callbacks['broadcast_frame'](frame_data),
                                    self.main_loop
                                )
                            except Exception as e:
                                logger.debug(f"Frame broadcast error: {e}")
                
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 30:
                    self.processing_times.pop(0)
                
                self._update_fps()
                
                target_fps = 30
                sleep_time = max(0, (1.0 / target_fps) - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Processing loop ended")
    
    def _process_frame(self, frame: np.ndarray) -> List[Dict]:
        if not self.model:
            return self._mock_detections(frame)
        
        try:
            results = self.model(frame, device=self.device, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        if class_name in self.detection_classes:
                            threshold = self.confidence_thresholds.get(class_name, 0.5)
                            if confidence >= threshold:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                detections.append({
                                    'class_name': class_name,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                                    'timestamp': datetime.utcnow().isoformat()
                                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []
    
    def _mock_detections(self, frame: np.ndarray) -> List[Dict]:
        import random
        detections = []
        
        if random.random() < 0.3:
            h, w = frame.shape[:2]
            detections.append({
                'class_name': random.choice(['person', 'car']),
                'confidence': random.uniform(0.5, 0.9),
                'bbox': [
                    random.randint(0, w//2),
                    random.randint(0, h//2), 
                    random.randint(50, 200),
                    random.randint(50, 200)
                ],
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return detections
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = frame.copy()
        
        for detection in detections:
            x, y, w, h = [int(v) for v in detection['bbox']]
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            color = self._get_class_color(class_name)
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            cv2.rectangle(annotated, 
                         (x, y - label_size[1] - 10),
                         (x + label_size[0], y), 
                         color, -1)
            
            cv2.putText(annotated, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(annotated, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'truck': (0, 0, 255),       # Red
            'motorcycle': (255, 255, 0), # Cyan
            'weapon': (0, 0, 255)       # Red
        }
        return colors.get(class_name, (128, 128, 128))
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])
            frame_data = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{frame_data}"
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return ""
    
    async def _handle_detections(self, detections: List[Dict]):
        for detection in detections:
            await db_manager.log_detection(
                detection['class_name'],
                detection['confidence'],
                detection['bbox']
            )
            
            # Mark as video detection
            detection['modality'] = 'video'
            
            if 'broadcast_detection' in self.broadcast_callbacks:
                await self.broadcast_callbacks['broadcast_detection'](detection)
            
            # Send to fusion engine for multimodal processing
            if self.multimodal_enabled:
                await self.fusion_engine.add_video_detection(detection)
            
            if detection['class_name'] == 'weapon':
                alert_data = {
                    'type': 'weapon_detected',
                    'message': f"Weapon detected with {detection['confidence']:.2f} confidence",
                    'timestamp': detection['timestamp'],
                    'detection': detection
                }
                if 'broadcast_alert' in self.broadcast_callbacks:
                    await self.broadcast_callbacks['broadcast_alert'](alert_data)
    
    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    async def get_stats(self) -> Dict[str, Any]:
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        stats = {
            'fps': self.fps,
            'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
            'is_running': self.is_running,
            'streaming': self.start_streaming,
            'model_loaded': self.model is not None,
            'camera_available': self.camera is not None and self.camera.isOpened(),
            'device': getattr(self, 'device', 'cpu'),
            'multimodal_enabled': self.multimodal_enabled
        }
        
        # Add multimodal stats if available
        if self.multimodal_enabled:
            stats['audio_stats'] = await self.audio_engine.get_audio_stats()
            stats['fusion_stats'] = await self.fusion_engine.get_fusion_stats()
        
        return stats
    
    async def update_confidence_thresholds(self, thresholds: Dict[str, float]):
        self.confidence_thresholds.update(thresholds)
        await db_manager.set_config("confidence_thresholds", self.confidence_thresholds)
    
    async def switch_model(self, model_path: str) -> bool:
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            if YOLO_AVAILABLE:
                new_model = YOLO(model_path)
                self.model = new_model
                self.model_path = model_path
                logger.info(f"Switched to model: {model_path}")
                return True
            else:
                logger.error("YOLO not available for model switching")
                return False
                
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    async def _handle_audio_detection(self, audio_detection: Dict):
        """Handle audio detection from audio engine"""
        try:
            # Send to fusion engine
            await self.fusion_engine.add_audio_detection(audio_detection)
            
            # Also broadcast audio detection directly
            if 'broadcast_detection' in self.broadcast_callbacks:
                await self.broadcast_callbacks['broadcast_detection'](audio_detection)
                
        except Exception as e:
            logger.error(f"Error handling audio detection: {e}")
    
    async def cleanup(self):
        logger.info("Starting inference engine cleanup...")
        await self.stop_processing()
        
        # Clean up multimodal components
        if self.multimodal_enabled:
            await self.audio_engine.cleanup()
            await self.fusion_engine.cleanup()
        
        # Ensure camera is properly released
        if self.camera:
            logger.info("Releasing camera...")
            self.camera.release()
            self.camera = None
        
        logger.info("Inference engine cleanup complete")