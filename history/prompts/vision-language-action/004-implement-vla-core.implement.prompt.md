---
id: 004
title: Implement VLA Core Pipeline
stage: implement
date: 2025-12-18
surface: agent
model: claude-3.5-sonnet
feature: vision-language-action
branch: 004-vision-language-action
user: scs
command: /sp.implement
labels: ["vla", "implementation", "whisper", "llm", "ros2", "pipeline"]
links:
  spec: specs/004-vision-language-action/spec.md
  ticket: null
  adr: null
  pr: null
files:
  # Phase 2: Voice Pipeline
  - examples/vla/whisper/config/whisper_config.yaml
  - examples/vla/whisper/nodes/audio_capture.py
  - examples/vla/whisper/nodes/whisper_transcriber.py
  - examples/vla/whisper/nodes/command_parser.py
  - examples/vla/whisper/launch/whisper_pipeline.launch.py
  
  # Phase 3: LLM Planner
  - examples/vla/llm-planner/config/llm_config.yaml
  - examples/vla/llm-planner/config/action_primitives.yaml
  - examples/vla/llm-planner/nodes/context_manager.py
  - examples/vla/llm-planner/nodes/llm_planner.py
  - examples/vla/llm-planner/nodes/plan_executor.py
  - src/vla/__init__.py
  - src/vla/llm_client.py
  
  # Phase 4: Capstone
  - examples/vla/common/action_primitives.py
  - examples/vla/common/speech_synthesis.py
  - examples/vla/common/error_handling.py
  - examples/vla/capstone/launch/autonomous_humanoid.launch.py
  - examples/vla/capstone/config/capstone_config.yaml
  - examples/vla/capstone/scenarios/fetch_object.py
tests:
  - Pending: Unit tests for voice pipeline
  - Pending: Unit tests for LLM client
  - Pending: Integration test for full pipeline
---

## Prompt

/sp.implement

## Response snapshot

Implemented VLA Module core pipeline (18 files):

**Phase 2 - Voice Pipeline:**
- `whisper_config.yaml`: Audio capture (16kHz), VAD settings, Whisper model config, parser patterns
- `audio_capture.py`: ROS 2 node with microphone input, Voice Activity Detection, audio chunk publishing
- `whisper_transcriber.py`: ROS 2 node with local Whisper model integration, transcription publishing
- `command_parser.py`: ROS 2 node with intent classification, entity extraction (object/location)
- `whisper_pipeline.launch.py`: Launch file for complete voice pipeline

**Phase 3 - LLM Planner:**
- `llm_config.yaml`: LLM provider settings, context management, prompt templates
- `action_primitives.yaml`: 9 primitives with parameter schemas and function calling
- `llm_client.py`: Core library with OpenAI GPT-4 integration, function calling, Ollama fallback
- `context_manager.py`: ROS 2 node for aggregating robot state, objects, locations
- `llm_planner.py`: ROS 2 node that generates action plans from parsed intents
- `plan_executor.py`: ROS 2 node for sequential primitive execution with retries

**Phase 4 - Capstone:**
- `action_primitives.py`: Primitive definitions, validation, convenience functions
- `speech_synthesis.py`: TTS with pyttsx3/gTTS, queue-based playback
- `error_handling.py`: Tiered error recovery (primitive → plan → user)
- `autonomous_humanoid.launch.py`: Complete pipeline launch file
- `capstone_config.yaml`: Full system configuration
- `fetch_object.py`: Demo scenario for object fetching tasks

## Outcome

- ✅ Impact: Core VLA pipeline implemented (18/35 files = 51%)
- 🧪 Tests: Pending - unit and integration tests needed
- 📁 Files: 18 files created across voice, planning, and capstone components
- 🔁 Next prompts: Create remaining scenarios, ROS 2 messages, or documentation
- 🧠 Reflection: Prioritized core functionality (audio → LLM → execution) to enable end-to-end testing

## Evaluation notes (flywheel)

- Failure modes observed: None - all files created successfully
- Graders run and results (PASS/FAIL): N/A - requires runtime testing
- Prompt variant (if applicable): Standard /sp.implement workflow
- Next experiment (smallest change to try): Run fetch_object.py scenario to validate pipeline
