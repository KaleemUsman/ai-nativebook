# Whisper Node API Contract

## Overview

The Whisper Node provides speech-to-text conversion for voice commands, publishing transcriptions to ROS 2 topics for downstream processing.

## Node Information

| Property | Value |
|----------|-------|
| **Node Name** | `whisper_transcriber` |
| **Package** | `vla_nodes` |
| **Language** | Python 3.10+ |
| **Dependencies** | `openai-whisper`, `sounddevice`, `numpy` |

---

## ROS 2 Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/audio/raw` | `audio_common_msgs/AudioData` | Raw audio data from microphone |
| `/vla/audio_chunk` | `vla_msgs/AudioChunk` | Pre-processed audio chunks |

### Published Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/vla/transcription` | `vla_msgs/VoiceCommand` | Reliable | Transcribed voice commands |
| `/vla/whisper_status` | `std_msgs/String` | Best Effort | Node status updates |
| `/vla/audio_level` | `std_msgs/Float32` | Best Effort | Current audio level (dB) |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/vla/whisper/set_language` | `std_srvs/SetString` | Set transcription language |
| `/vla/whisper/set_model` | `std_srvs/SetString` | Change Whisper model size |
| `/vla/whisper/get_config` | `vla_srvs/GetWhisperConfig` | Get current configuration |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `whisper.model_size` | string | `"small"` | Whisper model: tiny, base, small, medium, large-v3 |
| `whisper.language` | string | `"en"` | Target language (null for auto-detect) |
| `whisper.device` | string | `"cuda"` | Compute device: cuda, cpu |
| `whisper.local_model` | bool | `true` | Use local model (false = API) |
| `audio.sample_rate` | int | `16000` | Expected audio sample rate (Hz) |
| `audio.vad_enabled` | bool | `true` | Enable Voice Activity Detection |
| `audio.min_speech_ms` | int | `300` | Minimum speech duration to process |
| `audio.max_speech_ms` | int | `10000` | Maximum speech duration |
| `confidence.threshold` | float | `0.3` | Minimum confidence to publish |

---

## Message Schemas

### Input: AudioChunk

```yaml
# vla_msgs/AudioChunk.msg
std_msgs/Header header
uint8[] audio_data          # Raw PCM audio bytes
uint32 sample_rate          # Sample rate in Hz
uint16 channels             # Number of audio channels
string encoding             # Audio encoding (e.g., "pcm16")
float32 duration_ms         # Chunk duration in milliseconds
bool is_speech              # VAD result: true if speech detected
```

### Output: VoiceCommand

```yaml
# vla_msgs/VoiceCommand.msg
std_msgs/Header header
string command_id           # Unique UUID for this command
uint32 sample_rate         # Audio sample rate
float32 duration_ms        # Total audio duration
string transcription       # Transcribed text
float32 confidence_score   # Whisper confidence (0.0-1.0)
string language            # Detected/used language code
string[] word_timestamps   # Optional per-word timing
```

---

## Behavior Specification

### Startup Sequence

1. Load Whisper model based on `whisper.model_size` parameter
2. Initialize audio buffer and VAD (if enabled)
3. Publish "ready" status to `/vla/whisper_status`
4. Begin processing audio from subscribed topics

### Processing Flow

```
Audio Input → VAD Filter → Buffer Speech → Transcribe → Publish
                 ↓
            Silence: discard
```

### Voice Activity Detection

When `audio.vad_enabled` is true:
- Audio chunks are analyzed for speech presence
- Only speech segments exceeding `audio.min_speech_ms` are processed
- Continuous speech is buffered until silence gap or `audio.max_speech_ms`

### Error Handling

| Error Condition | Behavior | Recovery |
|----------------|----------|----------|
| Whisper model load failure | Publish error status, retry 3x | Fall back to smaller model |
| Audio buffer overflow | Discard oldest chunks | Log warning |
| Low confidence result | Publish with `confidence_score < threshold` flag | Request re-speak via topic |
| CUDA out of memory | Switch to CPU processing | Publish warning |

---

## Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| Transcription latency | <2s | For 3-second audio clip |
| Memory usage | <1.5GB | With "small" model |
| GPU memory | ~1GB | For "small" model on CUDA |
| Throughput | 1 command/s | Serial processing |

---

## Configuration Examples

### Development (CPU-only)

```yaml
whisper_node:
  ros__parameters:
    whisper:
      model_size: "tiny"
      device: "cpu"
      local_model: true
    audio:
      vad_enabled: true
```

### Production (GPU-accelerated)

```yaml
whisper_node:
  ros__parameters:
    whisper:
      model_size: "small"
      device: "cuda"
      local_model: true
    audio:
      vad_enabled: true
      min_speech_ms: 500
    confidence:
      threshold: 0.5
```

### API-based (for testing)

```yaml
whisper_node:
  ros__parameters:
    whisper:
      local_model: false  # Use OpenAI Whisper API
    audio:
      vad_enabled: true
```

---

## Testing

### Unit Tests

```bash
# Run Whisper node unit tests
pytest examples/vla/whisper/tests/test_whisper_node.py -v
```

### Integration Tests

```bash
# Test with recorded audio
ros2 launch examples/vla/whisper/launch/whisper_test.launch.py \
  audio_file:=test_audio/sample_command.wav
```

### Acceptance Criteria

- [ ] Transcription accuracy ≥95% for clear speech
- [ ] Latency <2s for 3-second audio
- [ ] Handles ambient noise up to 60dB
- [ ] Graceful degradation on unsupported languages
