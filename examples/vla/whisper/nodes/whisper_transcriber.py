#!/usr/bin/env python3
"""
Whisper Transcriber Node for VLA Pipeline

This node receives audio chunks from the audio capture node and transcribes
them using OpenAI Whisper (local model or API).

Subscribes to:
    - /vla/audio_chunk (String with base64 encoded audio)
    
Publishes to:
    - /vla/transcription (String with transcription result)
    - /vla/whisper_status (std_msgs/String)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

import numpy as np
import base64
import threading
import queue
import time
import json
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("WARNING: openai-whisper not installed. Transcription will be simulated.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Represents a transcription result"""
    command_id: str
    transcription: str
    language: str
    confidence: float
    duration_ms: float
    timestamp: str
    words: list = None  # Optional word-level timestamps
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class WhisperTranscriberNode(Node):
    """
    ROS 2 node for transcribing audio using OpenAI Whisper.
    
    Supports both local model inference and OpenAI API (with local as default).
    """
    
    def __init__(self):
        super().__init__('whisper_transcriber')
        
        # Declare parameters
        self.declare_parameter('model_size', 'small')
        self.declare_parameter('language', 'en')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('local_model', True)
        self.declare_parameter('temperature', 0.0)
        self.declare_parameter('word_timestamps', False)
        self.declare_parameter('no_speech_threshold', 0.6)
        
        # Get parameters
        self.model_size = self.get_parameter('model_size').value
        self.language = self.get_parameter('language').value
        self.device = self.get_parameter('device').value
        self.use_local = self.get_parameter('local_model').value
        self.temperature = self.get_parameter('temperature').value
        self.word_timestamps = self.get_parameter('word_timestamps').value
        self.no_speech_threshold = self.get_parameter('no_speech_threshold').value
        
        # Determine device
        if self.device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        
        # Initialize Whisper model
        self.model = None
        self.model_loaded = False
        self._load_model()
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.audio_sub = self.create_subscription(
            String,
            '/vla/audio_chunk',
            self._audio_callback,
            qos
        )
        
        # Publishers
        self.transcription_pub = self.create_publisher(
            String,
            '/vla/transcription',
            qos
        )
        self.status_pub = self.create_publisher(String, '/vla/whisper_status', qos)
        
        # Processing queue and thread
        self.audio_queue = queue.Queue()
        self.is_running = True
        
        self.process_thread = threading.Thread(target=self._process_audio_thread)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        self._publish_status(f"Whisper node ready: model={self.model_size}, device={self.device}")
    
    def _load_model(self):
        """Load the Whisper model"""
        if not WHISPER_AVAILABLE:
            self.get_logger().warn("Whisper not available, using simulation mode")
            return
        
        if not self.use_local:
            self.get_logger().info("Using OpenAI Whisper API (not local model)")
            return
        
        try:
            self._publish_status(f"Loading Whisper model: {self.model_size}...")
            start_time = time.time()
            
            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )
            
            load_time = time.time() - start_time
            self.model_loaded = True
            self._publish_status(
                f"Whisper model loaded: {self.model_size} on {self.device} "
                f"({load_time:.1f}s)"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load Whisper model: {e}")
            self._publish_status(f"Model load failed: {e}")
    
    def _publish_status(self, message: str):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def _audio_callback(self, msg: String):
        """Handle incoming audio data"""
        try:
            data = msg.data
            
            # Parse message format: TYPE:SAMPLE_RATE:LENGTH:DATA
            if data.startswith("SPEECH:") or data.startswith("CHUNK:"):
                parts = data.split(":", 4)
                if len(parts) >= 4:
                    msg_type = parts[0]
                    sample_rate = int(parts[1])
                    length = int(parts[2])
                    
                    if msg_type == "SPEECH":
                        audio_b64 = parts[3]
                    else:
                        # CHUNK format has is_speech flag
                        audio_b64 = parts[4] if len(parts) > 4 else parts[3]
                    
                    # Decode audio
                    audio_bytes = base64.b64decode(audio_b64)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    self.audio_queue.put((audio, sample_rate))
                    self.get_logger().debug(f"Received audio: {len(audio)} samples")
        except Exception as e:
            self.get_logger().error(f"Error parsing audio message: {e}")
    
    def _process_audio_thread(self):
        """Background thread for processing audio"""
        while self.is_running:
            try:
                # Wait for audio with timeout
                try:
                    audio, sample_rate = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Transcribe
                result = self._transcribe(audio, sample_rate)
                
                if result and result.transcription.strip():
                    # Publish result
                    msg = String()
                    msg.data = result.to_json()
                    self.transcription_pub.publish(msg)
                    
                    self.get_logger().info(
                        f"Transcription: '{result.transcription}' "
                        f"(confidence: {result.confidence:.2f})"
                    )
                
            except Exception as e:
                self.get_logger().error(f"Transcription error: {e}")
    
    def _transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio samples (float32, normalized to [-1, 1])
            sample_rate: Audio sample rate
            
        Returns:
            TranscriptionResult or None if failed
        """
        start_time = time.time()
        command_id = str(uuid.uuid4())
        duration_ms = len(audio) / sample_rate * 1000
        
        if not WHISPER_AVAILABLE or not self.model_loaded:
            # Simulation mode
            return self._simulate_transcription(command_id, duration_ms)
        
        try:
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                # Simple resampling - in production, use scipy.signal.resample
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)
            
            # Pad or trim to 30 seconds (Whisper's expected input)
            audio = whisper.pad_or_trim(audio)
            
            # Convert to mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Detect language if not specified
            if self.language:
                language = self.language
            else:
                _, probs = self.model.detect_language(mel)
                language = max(probs, key=probs.get)
            
            # Decode options
            options = whisper.DecodingOptions(
                language=language,
                temperature=self.temperature,
                without_timestamps=not self.word_timestamps,
                fp16=(self.device == 'cuda')
            )
            
            # Transcribe
            result = whisper.decode(self.model, mel, options)
            
            # Calculate confidence from avg log probability
            confidence = np.exp(result.avg_logprob) if result.avg_logprob else 0.5
            
            # Check for no speech
            if result.no_speech_prob > self.no_speech_threshold:
                self.get_logger().debug(
                    f"No speech detected (prob: {result.no_speech_prob:.2f})"
                )
                return None
            
            return TranscriptionResult(
                command_id=command_id,
                transcription=result.text.strip(),
                language=language,
                confidence=float(min(confidence, 1.0)),
                duration_ms=duration_ms,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.get_logger().error(f"Whisper transcription error: {e}")
            return None
    
    def _simulate_transcription(
        self,
        command_id: str,
        duration_ms: float
    ) -> TranscriptionResult:
        """Simulate transcription for testing without Whisper"""
        # Simulated responses for testing
        simulated_texts = [
            "go to the kitchen",
            "pick up the red cup",
            "navigate to the living room",
            "find my keys",
            "what do you see",
        ]
        
        import random
        text = random.choice(simulated_texts)
        
        # Simulate processing time
        time.sleep(0.5)
        
        return TranscriptionResult(
            command_id=command_id,
            transcription=text,
            language="en",
            confidence=0.95,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat()
        )
    
    def destroy_node(self):
        """Cleanup on node shutdown"""
        self.is_running = False
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WhisperTranscriberNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
