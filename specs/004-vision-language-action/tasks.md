# Tasks: Vision-Language-Action (VLA)

**Branch**: `004-vision-language-action` | **Date**: 2025-12-18 | **Plan**: [plan.md](./plan.md)
**Status**: 🔄 In Progress - Core Implementation Complete

## Overview

This document contains the implementation checklist for Module 4: Vision-Language-Action (VLA). Tasks are organized by phase, aligned with user stories from the spec and mapped to success criteria.

---

## Phase 1: Environment Setup

**Goal**: Prepare development environment for VLA implementation.
**Estimated Duration**: 1-2 hours

### 1.1 ROS 2 Humble Setup

- [ ] **1.1.1** Verify ROS 2 Humble installation
  ```bash
  ros2 --version  # Should be Humble or newer
  source /opt/ros/humble/setup.bash
  ```
- [ ] **1.1.2** Install ROS 2 audio packages
  ```bash
  sudo apt install ros-humble-audio-common ros-humble-audio-common-msgs
  ```
- [ ] **1.1.3** Create VLA workspace
  ```bash
  mkdir -p ~/ros2_ws/src/vla_nodes
  cd ~/ros2_ws && colcon build
  ```
- **Acceptance**: ROS 2 commands execute without error

### 1.2 Python Environment Setup

- [ ] **1.2.1** Verify Python 3.10+ installation
  ```bash
  python3 --version  # Should be 3.10+
  ```
- [ ] **1.2.2** Create virtual environment (optional)
  ```bash
  python3 -m venv ~/vla_env
  source ~/vla_env/bin/activate
  ```
- [ ] **1.2.3** Install core dependencies
  ```bash
  pip install openai openai-whisper sounddevice pyttsx3 numpy scipy httpx
  ```
- [ ] **1.2.4** Install GPU support (if available)
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **Acceptance**: All Python packages import without error

### 1.3 Simulation Environment

- [ ] **1.3.1** Verify Module 2 (Gazebo/Unity) simulation is functional
- [ ] **1.3.2** Verify Module 3 (Isaac AI Brain) perception pipelines work
- [ ] **1.3.3** Test Nav2 navigation stack
  ```bash
  ros2 launch nav2_bringup navigation_launch.py
  ```
- [ ] **1.3.4** Configure environment variables
  ```bash
  export OPENAI_API_KEY="your-key-here"
  echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc
  ```
- **Acceptance**: Simulation launches with humanoid robot

---

## Phase 2: Voice-to-Action Pipeline 🎯 MVP

**Goal**: Implement voice command capture, Whisper transcription, and command parsing.
**Supports**: User Story 1 - Voice-to-Action Command Processing (P1)
**Estimated Duration**: 4-6 hours

### 2.1 Install and Configure OpenAI Whisper

- [ ] **2.1.1** Install Whisper locally
  ```bash
  pip install openai-whisper
  whisper --help  # Verify installation
  ```
- [ ] **2.1.2** Download Whisper model
  ```python
  import whisper
  model = whisper.load_model("small")  # Downloads ~500MB
  ```
- [ ] **2.1.3** Create Whisper configuration
  - [ ] Create `examples/vla/whisper/config/whisper_config.yaml`
  - [ ] Define model size, language, VAD settings
- [ ] **2.1.4** Test Whisper transcription manually
  ```bash
  whisper test_audio.wav --model small --language en
  ```
- **Acceptance**: Whisper transcribes test audio with ≥95% accuracy

### 2.2 Capture and Transcribe Voice Commands

- [ ] **2.2.1** Create audio capture node
  - [ ] Create `examples/vla/whisper/nodes/audio_capture.py`
  - [ ] Implement microphone input with `sounddevice`
  - [ ] Add Voice Activity Detection (VAD)
  - [ ] Publish audio chunks to `/vla/audio_chunk`
- [ ] **2.2.2** Create Whisper transcriber node
  - [ ] Create `examples/vla/whisper/nodes/whisper_transcriber.py`
  - [ ] Subscribe to `/vla/audio_chunk`
  - [ ] Process audio through Whisper model
  - [ ] Publish transcriptions to `/vla/transcription`
- [ ] **2.2.3** Create launch file
  - [ ] Create `examples/vla/whisper/launch/whisper_pipeline.launch.py`
- [ ] **2.2.4** Test voice capture pipeline
  ```bash
  ros2 launch examples/vla/whisper/launch/whisper_pipeline.launch.py
  ros2 topic echo /vla/transcription
  # Speak: "Go to the kitchen"
  ```
- **Acceptance**: Voice commands transcribed in real-time (SC-001: ≥95% accuracy)

### 2.3 Map Transcriptions to Structured ROS 2 Actions

- [ ] **2.3.1** Create command parser node
  - [ ] Create `examples/vla/whisper/nodes/command_parser.py`
  - [ ] Parse transcription into structured intent
  - [ ] Extract: action verb, target object, target location, modifiers
  - [ ] Handle multi-step commands (e.g., "Go to X and pick up Y")
  - [ ] Publish to `/vla/parsed_intent`
- [ ] **2.3.2** Define intent message types
  - [ ] Create `vla_msgs/msg/ParsedIntent.msg`
  - [ ] Create `vla_msgs/msg/VoiceCommand.msg`
- [ ] **2.3.3** Test command parsing
  ```bash
  ros2 topic echo /vla/parsed_intent
  # Speak: "Navigate to the living room"
  # Expected: intent_type=NAVIGATION, target_location="living room"
  ```
- **Acceptance**: Commands correctly parsed into structured intents

### 2.4 Phase 2 Testing

- [ ] **2.4.1** Unit tests for audio capture
- [ ] **2.4.2** Unit tests for Whisper transcription
- [ ] **2.4.3** Unit tests for command parsing
- [ ] **2.4.4** Integration test: voice → parsed intent
- **Acceptance**: All tests pass, SC-001 validated

---

## Phase 3: Cognitive Planning with LLMs

**Goal**: Implement LLM-based action planning with context management.
**Supports**: User Story 2 - LLM-Based Cognitive Planning (P2)
**Estimated Duration**: 6-8 hours

### 3.1 Integrate LLM API for Task Decomposition

- [ ] **3.1.1** Create LLM client library
  - [ ] Create `src/vla/llm_client.py`
  - [ ] Implement OpenAI GPT-4 integration
  - [ ] Add function calling for structured JSON output
  - [ ] Implement error handling and retries
- [ ] **3.1.2** Create context manager node
  - [ ] Create `examples/vla/llm-planner/nodes/context_manager.py`
  - [ ] Aggregate robot state from TF
  - [ ] Track detected objects from perception
  - [ ] Maintain task history buffer
  - [ ] Publish to `/vla/task_context`
- [ ] **3.1.3** Create LLM planner node
  - [ ] Create `examples/vla/llm-planner/nodes/llm_planner.py`
  - [ ] Subscribe to parsed intents and context
  - [ ] Generate action plans via LLM
  - [ ] Implement action server `/vla/plan_task`
- [ ] **3.1.4** Configure LLM settings
  - [ ] Create `examples/vla/llm-planner/config/llm_config.yaml`
  - [ ] Create `examples/vla/llm-planner/config/action_primitives.yaml`
- **Acceptance**: LLM generates structured action plans

### 3.2 Translate Commands into ROS 2 Action Sequences

- [ ] **3.2.1** Define action primitives library
  - [ ] Create `examples/vla/common/action_primitives.py`
  - [ ] Implement 9 primitives: navigate_to, look_at, scan_environment, identify_object, pick_up, place, say, wait, cancel
  - [ ] Define parameter schemas and validation
- [ ] **3.2.2** Implement plan generation prompts
  - [ ] Create system prompt with robot capabilities
  - [ ] Create function calling schema for action plans
  - [ ] Test with example commands
- [ ] **3.2.3** Test: "Clean the room" command
  ```bash
  ros2 action send_goal /vla/plan_task vla_msgs/action/PlanTask \
    "intent: {intent_type: 'MANIPULATION', action_verb: 'clean', target_location: 'room'}"
  # Expected: Plan with scan → identify_objects → pick_up → place sequence
  ```
- **Acceptance**: Complex commands decompose into valid action sequences

### 3.3 Handle Errors and Fallback Scenarios

- [ ] **3.3.1** Implement LLM fallback to Ollama
  - [ ] Configure Ollama with LLaMA 3 or Mistral
  - [ ] Add automatic fallback on API failure
- [ ] **3.3.2** Implement plan validation
  - [ ] Validate preconditions are satisfiable
  - [ ] Check for impossible action sequences
  - [ ] Return informative errors for impossible requests
- [ ] **3.3.3** Implement replanning action
  - [ ] Create `/vla/replan` action server
  - [ ] Accept failure context and generate new plan
- **Acceptance**: Graceful handling of API failures and invalid commands

### 3.4 Test Planning Pipeline

- [ ] **3.4.1** Test with navigation commands
  - [ ] "Go to the kitchen"
  - [ ] "Navigate to the table near the window"
- [ ] **3.4.2** Test with manipulation commands
  - [ ] "Pick up the red cup"
  - [ ] "Put the book on the shelf"
- [ ] **3.4.3** Test with complex multi-step commands
  - [ ] "Go to the kitchen and bring me a glass of water"
  - [ ] "Find my keys and put them on the desk"
- [ ] **3.4.4** Test with ambiguous/impossible commands
  - [ ] "Get me something to drink" (ambiguous)
  - [ ] "Fly to the moon" (impossible)
- **Acceptance**: SC-002: ≥90% of plans are executable

### 3.5 Phase 3 Testing

- [ ] **3.5.1** Unit tests for LLM client
- [ ] **3.5.2** Unit tests for context manager
- [ ] **3.5.3** Integration tests with mock LLM
- **Acceptance**: All tests pass

---

## Phase 4: Capstone Project – Autonomous Humanoid

**Goal**: Complete end-to-end autonomous pipeline integrating Modules 1-3.
**Supports**: User Story 3 - Autonomous Humanoid Capstone Integration (P3)
**Estimated Duration**: 8-12 hours

### 4.1 Integrate Modules 1–3

- [ ] **4.1.1** Create integration bridge
  - [ ] Create `src/vla/integration.py`
  - [ ] Interface with Module 2 simulation (Gazebo/Unity)
  - [ ] Interface with Module 3 perception (Isaac ROS)
  - [ ] Interface with Module 3 navigation (Nav2)
- [ ] **4.1.2** Create capstone launch file
  - [ ] Create `examples/vla/capstone/launch/autonomous_humanoid.launch.py`
  - [ ] Launch simulation environment
  - [ ] Launch perception pipelines
  - [ ] Launch navigation stack
  - [ ] Launch complete VLA pipeline
- [ ] **4.1.3** Create capstone configuration
  - [ ] Create `examples/vla/capstone/config/capstone_config.yaml`
- **Acceptance**: Full system launches without errors

### 4.2 Implement Full Pipeline

- [ ] **4.2.1** Create plan executor node
  - [ ] Create `examples/vla/llm-planner/nodes/plan_executor.py`
  - [ ] Implement action server `/vla/execute_plan`
  - [ ] Sequence primitive execution
  - [ ] Handle retries and failures
  - [ ] Publish execution status and feedback
- [ ] **4.2.2** Implement navigation primitives
  - [ ] Connect `navigate_to` to Nav2
  - [ ] Connect `look_at` to head/body controller
- [ ] **4.2.3** Implement perception primitives
  - [ ] Connect `scan_environment` to camera control
  - [ ] Connect `identify_object` to Module 3 detection
- [ ] **4.2.4** Implement manipulation primitives
  - [ ] Connect `pick_up` to arm controller/MoveIt
  - [ ] Connect `place` to arm controller
- [ ] **4.2.5** Implement feedback primitives
  - [ ] Create `examples/vla/common/speech_synthesis.py`
  - [ ] Connect `say` to TTS engine (pyttsx3/gTTS)
- **Acceptance**: Voice → Plan → Navigate → Identify → Manipulate works

### 4.3 Simulate Obstacle Navigation, Object Identification, and Manipulation

- [ ] **4.3.1** Create fetch object scenario
  - [ ] Create `examples/vla/capstone/scenarios/fetch_object.py`
  - [ ] Test: "Go to the desk and pick up the phone"
  - [ ] Validate: Navigation with obstacle avoidance
  - [ ] Validate: Object detection and localization
  - [ ] Validate: Grasp execution
- [ ] **4.3.2** Create navigate and report scenario
  - [ ] Create `examples/vla/capstone/scenarios/navigate_and_report.py`
  - [ ] Test: "Go to the kitchen and tell me what you see"
  - [ ] Validate: Navigation + scanning + speech output
- [ ] **4.3.3** Create multi-step task scenario
  - [ ] Create `examples/vla/capstone/scenarios/multi_step_task.py`
  - [ ] Test: "Find the red ball, pick it up, and bring it to me"
  - [ ] Validate: Full pipeline with multiple primitives
- **Acceptance**: Scenarios demonstrate complete pipeline

### 4.4 Validate End-to-End Task Execution

- [ ] **4.4.1** Implement error handling system
  - [ ] Create `examples/vla/common/error_handling.py`
  - [ ] Tiered recovery: primitive → plan → user fallback
  - [ ] Safety shutdown procedures
- [ ] **4.4.2** Test failure recovery
  - [ ] Object not found → scan environment → retry
  - [ ] Navigation blocked → replan path
  - [ ] Grasp failed → retry with different approach
- [ ] **4.4.3** Validate success criteria
  - [ ] SC-003: ≥80% task success rate
  - [ ] SC-004: Average completion <60s
  - [ ] SC-005: 95% meaningful error feedback
  - [ ] SC-006: <5s pipeline latency
- **Acceptance**: All success criteria met or documented exceptions

---

## Phase 5: Testing & Validation

**Goal**: Comprehensive testing and validation of the VLA system.
**Estimated Duration**: 4-6 hours

### 5.1 Run Multiple Autonomous Scenarios

- [ ] **5.1.1** Execute 10+ voice command variations
  - [ ] Simple navigation commands (5+)
  - [ ] Object manipulation commands (3+)
  - [ ] Complex multi-step commands (2+)
- [ ] **5.1.2** Record success/failure rates
- [ ] **5.1.3** Document failure modes and edge cases
- **Acceptance**: Comprehensive test coverage

### 5.2 Verify Accuracy Across Components

- [ ] **5.2.1** Voice recognition accuracy testing
  - [ ] Test with clear speech in quiet environment
  - [ ] Test with ambient noise (up to 60dB)
  - [ ] Test with different speaking rates
  - [ ] **Target**: SC-001 ≥95% accuracy
- [ ] **5.2.2** Planning accuracy testing
  - [ ] Test plan validity for standard commands
  - [ ] Test error handling for impossible commands
  - [ ] **Target**: SC-002 ≥90% executable plans
- [ ] **5.2.3** Navigation accuracy testing
  - [ ] Test goal reaching accuracy
  - [ ] Test obstacle avoidance
- [ ] **5.2.4** Manipulation accuracy testing
  - [ ] Test object detection accuracy
  - [ ] Test grasp success rate
- **Acceptance**: All components meet success criteria

### 5.3 Debug Integration Issues

- [ ] **5.3.1** Create integration test suite
  - [ ] Test Module 2 ↔ VLA interface
  - [ ] Test Module 3 ↔ VLA interface
  - [ ] Test timing and synchronization
- [ ] **5.3.2** Document known issues and workarounds
- [ ] **5.3.3** Create validation script
  - [ ] Create `examples/vla/capstone/validate_vla_pipeline.py`
- **Acceptance**: All integration tests pass

---

## Phase 6: Documentation

**Goal**: Create Docusaurus-ready documentation for the VLA module.
**Estimated Duration**: 4-6 hours

### 6.1 Voice-to-Action Chapter (Chapter 1)

- [ ] **6.1.1** Create `docs/modules/vla/voice-to-action.mdx`
- [ ] **6.1.2** Document Whisper integration
  - [ ] Model selection and performance tradeoffs
  - [ ] Audio configuration
  - [ ] VAD settings
- [ ] **6.1.3** Include runnable code blocks
- [ ] **6.1.4** Add architecture diagram
- [ ] **6.1.5** Add troubleshooting guide
- **Acceptance**: Documentation builds in Docusaurus

### 6.2 Cognitive Planning Chapter (Chapter 2)

- [ ] **6.2.1** Create `docs/modules/vla/cognitive-planning.mdx`
- [ ] **6.2.2** Document LLM integration
  - [ ] API configuration
  - [ ] Prompt engineering guidance
  - [ ] Function calling patterns
- [ ] **6.2.3** Explain action primitives
  - [ ] Primitive reference table
  - [ ] Parameter schemas
- [ ] **6.2.4** Include code examples
- [ ] **6.2.5** Add planning flow diagram
- **Acceptance**: Documentation builds in Docusaurus

### 6.3 Autonomous Humanoid Chapter (Chapter 3)

- [ ] **6.3.1** Create `docs/modules/vla/autonomous-humanoid.mdx`
- [ ] **6.3.2** Document capstone integration
  - [ ] Module 1-3 integration points
  - [ ] Pipeline architecture
- [ ] **6.3.3** Include demonstration walkthrough
  - [ ] Step-by-step screenshots
  - [ ] Expected outputs
- [ ] **6.3.4** Add performance tuning guide
- [ ] **6.3.5** Include demo video reference
- **Acceptance**: Documentation builds in Docusaurus

### 6.4 Cross-Module Consistency

- [ ] **6.4.1** Review terminology consistency with Modules 1-3
- [ ] **6.4.2** Verify code style consistency
- [ ] **6.4.3** Update module index/sidebar
- **Acceptance**: Consistent style across all modules

---

## Success Criteria Tracking

| Criterion | Description | Target | Phase | Status |
|-----------|-------------|--------|-------|--------|
| SC-001 | Voice transcription accuracy | ≥95% | Phase 2 | ⬜ |
| SC-002 | LLM plan executability | ≥90% | Phase 3 | ⬜ |
| SC-003 | End-to-end task success | ≥80% | Phase 4 | ⬜ |
| SC-004 | Task completion time | <60s | Phase 4 | ⬜ |
| SC-005 | Meaningful error feedback | ≥95% | Phase 4 | ⬜ |
| SC-006 | Pipeline latency | <5s | Phase 4 | ⬜ |
| SC-007 | Setup time | <30min | Quickstart | ⬜ |

---

## Risk Tracking

| Risk | Mitigation | Owner | Status |
|------|------------|-------|--------|
| LLM API latency/downtime | Ollama fallback | Phase 3.3 | ⬜ |
| Whisper noise sensitivity | VAD + preprocessing | Phase 2.1 | ⬜ |
| Module 2/3 integration | Clear API contracts | Phase 4.1 | ⬜ |
| Invalid action plans | Validation layer | Phase 3.3 | ⬜ |
| Hardware acceleration | CPU fallback | Phase 1.2 | ⬜ |

---

## File Checklist

### ROS 2 Nodes
- [x] `examples/vla/whisper/nodes/audio_capture.py`
- [x] `examples/vla/whisper/nodes/whisper_transcriber.py`
- [x] `examples/vla/whisper/nodes/command_parser.py`
- [x] `examples/vla/llm-planner/nodes/context_manager.py`
- [x] `examples/vla/llm-planner/nodes/llm_planner.py`
- [x] `examples/vla/llm-planner/nodes/plan_executor.py`

### Configuration
- [x] `examples/vla/whisper/config/whisper_config.yaml`
- [x] `examples/vla/llm-planner/config/llm_config.yaml`
- [x] `examples/vla/llm-planner/config/action_primitives.yaml`
- [x] `examples/vla/capstone/config/capstone_config.yaml`

### Launch Files
- [x] `examples/vla/whisper/launch/whisper_pipeline.launch.py`
- [ ] `examples/vla/llm-planner/launch/planner_pipeline.launch.py`
- [x] `examples/vla/capstone/launch/autonomous_humanoid.launch.py`

### Core Library
- [x] `src/vla/__init__.py`
- [ ] `src/vla/whisper_client.py`
- [x] `src/vla/llm_client.py`
- [ ] `src/vla/action_planner.py`
- [ ] `src/vla/task_executor.py`
- [ ] `src/vla/integration.py`

### Common Utilities
- [x] `examples/vla/common/action_primitives.py`
- [x] `examples/vla/common/speech_synthesis.py`
- [x] `examples/vla/common/error_handling.py`

### Scenarios
- [x] `examples/vla/capstone/scenarios/fetch_object.py`
- [ ] `examples/vla/capstone/scenarios/navigate_and_report.py`
- [ ] `examples/vla/capstone/scenarios/multi_step_task.py`

### Documentation
- [ ] `docs/modules/vla/voice-to-action.mdx`
- [ ] `docs/modules/vla/cognitive-planning.mdx`
- [ ] `docs/modules/vla/autonomous-humanoid.mdx`

### Messages (vla_msgs)
- [ ] `vla_msgs/msg/VoiceCommand.msg`
- [ ] `vla_msgs/msg/AudioChunk.msg`
- [ ] `vla_msgs/msg/ParsedIntent.msg`
- [ ] `vla_msgs/msg/ActionPlan.msg`
- [ ] `vla_msgs/msg/ActionPrimitive.msg`
- [ ] `vla_msgs/msg/TaskContext.msg`
- [ ] `vla_msgs/msg/ExecutionStatus.msg`
- [ ] `vla_msgs/msg/ExecutionResult.msg`
- [ ] `vla_msgs/action/PlanTask.action`
- [ ] `vla_msgs/action/ExecutePlan.action`
- [ ] `vla_msgs/action/Replan.action`
