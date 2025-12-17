#!/usr/bin/env python3
"""
Speech Synthesis Utilities for VLA Pipeline

Provides text-to-speech capabilities using pyttsx3 (offline) or gTTS (online).
"""

import os
import time
import threading
import queue
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class TTSEngine(Enum):
    """Available TTS engines"""
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    MOCK = "mock"


@dataclass
class SpeechRequest:
    """Request for speech synthesis"""
    text: str
    priority: str = "normal"  # low, normal, high
    callback: Optional[Callable] = None


class SpeechSynthesizer:
    """
    Text-to-Speech synthesizer with queue-based playback.
    
    Supports pyttsx3 for offline synthesis and gTTS for higher quality online synthesis.
    """
    
    def __init__(
        self,
        engine: str = "auto",
        rate: int = 175,
        volume: float = 1.0
    ):
        """
        Initialize the speech synthesizer.
        
        Args:
            engine: TTS engine ('pyttsx3', 'gtts', 'mock', or 'auto')
            rate: Speech rate (words per minute for pyttsx3)
            volume: Volume level (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        
        # Select engine
        if engine == "auto":
            if PYTTSX3_AVAILABLE:
                self.engine_type = TTSEngine.PYTTSX3
            elif GTTS_AVAILABLE:
                self.engine_type = TTSEngine.GTTS
            else:
                self.engine_type = TTSEngine.MOCK
        else:
            self.engine_type = TTSEngine(engine)
        
        # Initialize engine
        self.engine = None
        self._init_engine()
        
        # Speech queue
        self.speech_queue = queue.PriorityQueue()
        self.is_speaking = False
        self.is_running = True
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _init_engine(self):
        """Initialize the TTS engine"""
        if self.engine_type == TTSEngine.PYTTSX3:
            if PYTTSX3_AVAILABLE:
                try:
                    self.engine = pyttsx3.init()
                    self.engine.setProperty('rate', self.rate)
                    self.engine.setProperty('volume', self.volume)
                except Exception as e:
                    print(f"Failed to initialize pyttsx3: {e}")
                    self.engine_type = TTSEngine.MOCK
            else:
                self.engine_type = TTSEngine.MOCK
        
        elif self.engine_type == TTSEngine.GTTS:
            if GTTS_AVAILABLE:
                try:
                    pygame.mixer.init()
                except Exception as e:
                    print(f"Failed to initialize pygame mixer: {e}")
                    self.engine_type = TTSEngine.MOCK
            else:
                self.engine_type = TTSEngine.MOCK
        
        print(f"Speech synthesizer initialized with engine: {self.engine_type.value}")
    
    def say(
        self,
        text: str,
        priority: str = "normal",
        blocking: bool = False,
        callback: Optional[Callable] = None
    ):
        """
        Queue text for speech synthesis.
        
        Args:
            text: Text to speak
            priority: Speech priority ('low', 'normal', 'high')
            blocking: If True, wait for speech to complete
            callback: Function to call when speech completes
        """
        if not text or not text.strip():
            return
        
        # Map priority to queue priority (lower = higher priority)
        priority_map = {'high': 0, 'normal': 1, 'low': 2}
        queue_priority = priority_map.get(priority, 1)
        
        request = SpeechRequest(
            text=text.strip(),
            priority=priority,
            callback=callback
        )
        
        self.speech_queue.put((queue_priority, time.time(), request))
        
        if blocking:
            self.wait_until_done()
    
    def say_immediate(self, text: str):
        """
        Speak text immediately, interrupting any current speech.
        
        Args:
            text: Text to speak
        """
        # Clear queue
        self._clear_queue()
        
        # Add as high priority
        self.say(text, priority="high")
    
    def wait_until_done(self, timeout: float = 30.0):
        """Wait until all speech is complete"""
        start = time.time()
        while (self.is_speaking or not self.speech_queue.empty()) and \
              (time.time() - start) < timeout:
            time.sleep(0.1)
    
    def stop(self):
        """Stop current speech and clear queue"""
        self._clear_queue()
        
        if self.engine_type == TTSEngine.PYTTSX3 and self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        elif self.engine_type == TTSEngine.GTTS:
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    def shutdown(self):
        """Shutdown the synthesizer"""
        self.is_running = False
        self.stop()
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        if self.engine_type == TTSEngine.GTTS:
            try:
                pygame.mixer.quit()
            except:
                pass
    
    def _clear_queue(self):
        """Clear the speech queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
    
    def _worker_loop(self):
        """Background worker for processing speech queue"""
        while self.is_running:
            try:
                # Get next request (with timeout)
                try:
                    _, _, request = self.speech_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.is_speaking = True
                
                # Synthesize and play
                self._speak(request.text)
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback()
                    except Exception as e:
                        print(f"Speech callback error: {e}")
                
                self.is_speaking = False
                
            except Exception as e:
                print(f"Speech worker error: {e}")
                self.is_speaking = False
    
    def _speak(self, text: str):
        """Perform actual speech synthesis"""
        if self.engine_type == TTSEngine.PYTTSX3:
            self._speak_pyttsx3(text)
        elif self.engine_type == TTSEngine.GTTS:
            self._speak_gtts(text)
        else:
            self._speak_mock(text)
    
    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3"""
        if not self.engine:
            return
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"pyttsx3 error: {e}")
    
    def _speak_gtts(self, text: str):
        """Speak using gTTS + pygame"""
        if not GTTS_AVAILABLE:
            return
        
        try:
            import tempfile
            
            # Generate audio file
            tts = gTTS(text=text, lang='en')
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                filepath = f.name
                tts.save(filepath)
            
            # Play audio
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Cleanup
            try:
                os.remove(filepath)
            except:
                pass
                
        except Exception as e:
            print(f"gTTS error: {e}")
    
    def _speak_mock(self, text: str):
        """Mock speech for testing (just prints and delays)"""
        print(f"[SPEECH] {text}")
        # Estimate duration: ~100ms per word
        word_count = len(text.split())
        time.sleep(min(word_count * 0.1, 3.0))


# Global synthesizer instance
_synthesizer: Optional[SpeechSynthesizer] = None


def get_synthesizer() -> SpeechSynthesizer:
    """Get the global speech synthesizer instance"""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = SpeechSynthesizer()
    return _synthesizer


def say(text: str, priority: str = "normal", blocking: bool = False):
    """Convenience function to speak text"""
    get_synthesizer().say(text, priority, blocking)


def say_immediate(text: str):
    """Convenience function to speak immediately"""
    get_synthesizer().say_immediate(text)


def stop_speaking():
    """Convenience function to stop speech"""
    get_synthesizer().stop()


def shutdown_speech():
    """Shutdown the speech synthesizer"""
    global _synthesizer
    if _synthesizer:
        _synthesizer.shutdown()
        _synthesizer = None


# ROS 2 Node wrapper for speech synthesis
def create_speech_node():
    """Create a ROS 2 speech synthesis node"""
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
        class SpeechSynthesisNode(Node):
            def __init__(self):
                super().__init__('speech_synthesis')
                
                self.synthesizer = SpeechSynthesizer()
                
                qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    depth=10
                )
                
                self.speech_sub = self.create_subscription(
                    String,
                    '/vla/speech_output',
                    self._speech_callback,
                    qos
                )
                
                self.get_logger().info("Speech synthesis node ready")
            
            def _speech_callback(self, msg: String):
                text = msg.data
                if text:
                    self.get_logger().info(f"Speaking: {text[:50]}...")
                    self.synthesizer.say(text)
            
            def destroy_node(self):
                self.synthesizer.shutdown()
                super().destroy_node()
        
        return SpeechSynthesisNode
        
    except ImportError:
        print("rclpy not available for speech node")
        return None
