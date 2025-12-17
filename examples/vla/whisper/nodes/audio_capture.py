#!/usr/bin/env python3
"""
Audio Capture Node for VLA Pipeline

This node captures audio from the microphone, applies Voice Activity Detection (VAD),
and publishes audio chunks for transcription by the Whisper node.

Subscribes to: None (captures from microphone)
Publishes to:
    - /vla/audio_chunk (vla_msgs/AudioChunk)
    - /vla/audio_level (std_msgs/Float32)
    - /vla/whisper_status (std_msgs/String)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32

import numpy as np
import threading
import queue
import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("WARNING: sounddevice not installed. Audio capture will be simulated.")

try:
    import webrtcvad
except ImportError:
    webrtcvad = None
    print("WARNING: webrtcvad not installed. VAD will use energy-based detection.")


class VADState(Enum):
    """Voice Activity Detection states"""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    SPEECH_END = "speech_end"


@dataclass
class AudioChunk:
    """Represents a chunk of audio data"""
    data: np.ndarray
    timestamp: float
    is_speech: bool
    energy_db: float


class VoiceActivityDetector:
    """
    Voice Activity Detection using energy-based or WebRTC VAD.
    
    Detects speech segments in audio stream and manages speech state transitions.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = -40.0,
        speech_pad_ms: int = 300,
        min_speech_ms: int = 500,
        max_speech_ms: int = 15000,
        silence_ms: int = 700,
        use_webrtc: bool = True
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.max_speech_samples = int(max_speech_ms * sample_rate / 1000)
        self.silence_samples = int(silence_ms * sample_rate / 1000)
        
        self.state = VADState.SILENCE
        self.speech_buffer = []
        self.speech_samples = 0
        self.silence_samples_count = 0
        self.padding_buffer = []
        
        # WebRTC VAD (if available)
        self.webrtc_vad = None
        if use_webrtc and webrtcvad is not None:
            try:
                self.webrtc_vad = webrtcvad.Vad(2)  # Mode 2: moderate aggressiveness
            except Exception as e:
                print(f"Could not initialize WebRTC VAD: {e}")
    
    def calculate_energy_db(self, audio: np.ndarray) -> float:
        """Calculate audio energy in decibels"""
        if len(audio) == 0:
            return -100.0
        
        # RMS energy
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if rms < 1e-10:
            return -100.0
        
        # Convert to dB (relative to max int16 value)
        db = 20 * np.log10(rms / 32768.0)
        return db
    
    def is_speech_frame(self, audio: np.ndarray) -> bool:
        """Determine if audio frame contains speech"""
        # Try WebRTC VAD first
        if self.webrtc_vad is not None:
            try:
                # WebRTC VAD requires 10, 20, or 30ms frames
                frame_bytes = (audio * 32767).astype(np.int16).tobytes()
                # Process in 30ms chunks
                frame_duration = len(audio) / self.sample_rate
                if frame_duration in [0.01, 0.02, 0.03]:
                    return self.webrtc_vad.is_speech(frame_bytes, self.sample_rate)
            except Exception:
                pass
        
        # Fallback to energy-based detection
        energy_db = self.calculate_energy_db(audio)
        return energy_db > self.energy_threshold
    
    def process(self, audio: np.ndarray) -> tuple[VADState, Optional[np.ndarray]]:
        """
        Process audio chunk and return VAD state and completed speech segment if any.
        
        Returns:
            tuple: (current_state, speech_segment or None)
        """
        is_speech = self.is_speech_frame(audio)
        completed_segment = None
        
        if self.state == VADState.SILENCE:
            # Keep padding buffer for context
            self.padding_buffer.append(audio)
            if len(self.padding_buffer) > 10:  # Keep last ~1 second
                self.padding_buffer.pop(0)
            
            if is_speech:
                # Transition to speech
                self.state = VADState.SPEECH_START
                # Include padding buffer as context
                self.speech_buffer = self.padding_buffer.copy()
                self.speech_buffer.append(audio)
                self.speech_samples = sum(len(chunk) for chunk in self.speech_buffer)
                self.silence_samples_count = 0
        
        elif self.state in [VADState.SPEECH_START, VADState.SPEECH]:
            self.state = VADState.SPEECH
            self.speech_buffer.append(audio)
            self.speech_samples += len(audio)
            
            if is_speech:
                self.silence_samples_count = 0
            else:
                self.silence_samples_count += len(audio)
            
            # Check for speech end conditions
            if self.silence_samples_count >= self.silence_samples:
                # Silence detected - end of speech
                if self.speech_samples >= self.min_speech_samples:
                    self.state = VADState.SPEECH_END
                    completed_segment = np.concatenate(self.speech_buffer)
                else:
                    # Too short - discard
                    self.state = VADState.SILENCE
                
                self.speech_buffer = []
                self.speech_samples = 0
                self.silence_samples_count = 0
                self.padding_buffer = []
            
            elif self.speech_samples >= self.max_speech_samples:
                # Max duration reached - force end
                self.state = VADState.SPEECH_END
                completed_segment = np.concatenate(self.speech_buffer)
                self.speech_buffer = []
                self.speech_samples = 0
                self.silence_samples_count = 0
                self.padding_buffer = []
        
        elif self.state == VADState.SPEECH_END:
            # Reset to silence state
            self.state = VADState.SILENCE
            self.padding_buffer = [audio]
        
        return self.state, completed_segment
    
    def reset(self):
        """Reset VAD state"""
        self.state = VADState.SILENCE
        self.speech_buffer = []
        self.speech_samples = 0
        self.silence_samples_count = 0
        self.padding_buffer = []


class AudioCaptureNode(Node):
    """
    ROS 2 node for capturing audio from microphone with VAD.
    
    Publishes audio chunks when speech is detected, along with audio level
    information for monitoring.
    """
    
    def __init__(self):
        super().__init__('audio_capture')
        
        # Declare parameters
        self.declare_parameter('device_id', -1)  # -1 = default device
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('chunk_duration_ms', 100)
        self.declare_parameter('vad.enabled', True)
        self.declare_parameter('vad.energy_threshold', -40.0)
        self.declare_parameter('vad.speech_pad_ms', 300)
        self.declare_parameter('vad.min_speech_duration_ms', 500)
        self.declare_parameter('vad.max_speech_duration_ms', 15000)
        self.declare_parameter('vad.silence_duration_ms', 700)
        
        # Get parameters
        device_id = self.get_parameter('device_id').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        chunk_duration_ms = self.get_parameter('chunk_duration_ms').value
        vad_enabled = self.get_parameter('vad.enabled').value
        
        self.device_id = None if device_id == -1 else device_id
        self.chunk_samples = int(self.sample_rate * chunk_duration_ms / 1000)
        
        # Initialize VAD
        self.vad = None
        if vad_enabled:
            self.vad = VoiceActivityDetector(
                sample_rate=self.sample_rate,
                energy_threshold=self.get_parameter('vad.energy_threshold').value,
                speech_pad_ms=self.get_parameter('vad.speech_pad_ms').value,
                min_speech_ms=self.get_parameter('vad.min_speech_duration_ms').value,
                max_speech_ms=self.get_parameter('vad.max_speech_duration_ms').value,
                silence_ms=self.get_parameter('vad.silence_duration_ms').value
            )
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers
        # Note: In a full implementation, we'd use custom vla_msgs
        # For now, we use standard messages
        self.audio_pub = self.create_publisher(String, '/vla/audio_chunk', qos)
        self.level_pub = self.create_publisher(Float32, '/vla/audio_level', qos)
        self.status_pub = self.create_publisher(String, '/vla/whisper_status', qos)
        
        # Audio queue and thread
        self.audio_queue = queue.Queue()
        self.is_running = True
        
        # Start audio capture thread
        self.capture_thread = threading.Thread(target=self._capture_audio_thread)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Processing timer
        self.process_timer = self.create_timer(0.05, self._process_audio)  # 50ms
        
        # Publish ready status
        self._publish_status("Audio capture node ready")
        self.get_logger().info(
            f"Audio capture initialized: device={self.device_id}, "
            f"rate={self.sample_rate}Hz, VAD={'enabled' if vad_enabled else 'disabled'}"
        )
    
    def _publish_status(self, message: str):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def _capture_audio_thread(self):
        """Background thread for audio capture"""
        if sd is None:
            self.get_logger().warn("sounddevice not available, using simulated audio")
            self._simulate_audio()
            return
        
        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_samples,
                callback=self._audio_callback
            ):
                self._publish_status("Listening on microphone...")
                while self.is_running:
                    time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f"Audio capture error: {e}")
            self._publish_status(f"Audio capture error: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            self.get_logger().warn(f"Audio status: {status}")
        
        # Convert to mono if needed and flatten
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self.audio_queue.put(audio.copy())
    
    def _simulate_audio(self):
        """Simulate audio input for testing without microphone"""
        self._publish_status("Simulating audio input (no microphone)")
        while self.is_running:
            # Generate silence with occasional simulated speech
            duration = self.chunk_samples / self.sample_rate
            audio = np.zeros(self.chunk_samples, dtype=np.float32)
            
            # Add some noise
            audio += np.random.randn(self.chunk_samples) * 0.001
            
            self.audio_queue.put(audio)
            time.sleep(duration)
    
    def _process_audio(self):
        """Process queued audio chunks"""
        chunks_processed = 0
        max_chunks_per_cycle = 10
        
        while not self.audio_queue.empty() and chunks_processed < max_chunks_per_cycle:
            try:
                audio = self.audio_queue.get_nowait()
                self._handle_audio_chunk(audio)
                chunks_processed += 1
            except queue.Empty:
                break
    
    def _handle_audio_chunk(self, audio: np.ndarray):
        """Handle a single audio chunk"""
        # Calculate and publish audio level
        if self.vad:
            energy_db = self.vad.calculate_energy_db(audio)
        else:
            rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            energy_db = 20 * np.log10(max(rms, 1e-10) / 1.0)
        
        level_msg = Float32()
        level_msg.data = float(energy_db)
        self.level_pub.publish(level_msg)
        
        # Process with VAD
        if self.vad:
            state, speech_segment = self.vad.process(audio)
            
            if speech_segment is not None:
                # Complete speech segment detected
                self._publish_speech_segment(speech_segment)
        else:
            # No VAD - publish all audio
            self._publish_audio_chunk(audio, is_speech=True)
    
    def _publish_speech_segment(self, audio: np.ndarray):
        """Publish a complete speech segment"""
        duration_ms = len(audio) / self.sample_rate * 1000
        self.get_logger().info(f"Speech detected: {duration_ms:.0f}ms")
        
        # Convert to base64 for transmission
        # In a real implementation, we'd use a custom message type
        import base64
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        msg = String()
        msg.data = f"SPEECH:{self.sample_rate}:{len(audio)}:{audio_b64}"
        self.audio_pub.publish(msg)
        
        self._publish_status(f"Speech segment published: {duration_ms:.0f}ms")
    
    def _publish_audio_chunk(self, audio: np.ndarray, is_speech: bool):
        """Publish a single audio chunk"""
        import base64
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        msg = String()
        msg.data = f"CHUNK:{self.sample_rate}:{len(audio)}:{int(is_speech)}:{audio_b64}"
        self.audio_pub.publish(msg)
    
    def destroy_node(self):
        """Cleanup on node shutdown"""
        self.is_running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
