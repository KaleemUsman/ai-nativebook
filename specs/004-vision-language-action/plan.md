# Implementation Plan: Vision-Language-Action (VLA)

**Branch**: `004-vision-language-action` | **Date**: 2025-12-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-vision-language-action/spec.md`

## Artifacts

| Document | Purpose |
|----------|---------|
| [spec.md](./spec.md) | User stories, requirements, success criteria |
| [research.md](./research.md) | Technology decisions and alternatives |
| [data-model.md](./data-model.md) | Entity definitions and ROS 2 messages |
| [quickstart.md](./quickstart.md) | Setup and getting started guide |
| [tasks.md](./tasks.md) | Implementation checklist |
| [contracts/whisper-node.md](./contracts/whisper-node.md) | Whisper node API contract |
| [contracts/llm-planner.md](./contracts/llm-planner.md) | LLM planner API contract |
| [contracts/action-executor.md](./contracts/action-executor.md) | Action executor API contract |

## Summary

Implementation of Module 4: Vision-Language-Action (VLA), enabling humanoid robots to interpret natural language commands, plan actions using LLMs, and execute tasks autonomously. The module integrates OpenAI Whisper for speech recognition, LLM APIs (GPT-4) for cognitive planning, and connects with ROS 2 action systems, perception pipelines (Module 3), and navigation (Module 2) for end-to-end autonomous task execution.

## Technical Context

**Language/Version**: Python 3.10+ for all nodes and services
**Primary Dependencies**: OpenAI Whisper, OpenAI GPT-4 API, ROS 2 Humble, rclpy, sounddevice, pyttsx3/gTTS
**Storage**: File-based (audio files, action logs, configuration YAML)
**Testing**: pytest for Python nodes, ROS 2 launch_testing for integration
**Target Platform**: Linux (Ubuntu 22.04 LTS) for ROS 2, Windows/Mac for development
**Project Type**: Multi-component ROS 2 system with documentation
**Performance Goals**: Voice transcription <2s, LLM planning <5s, end-to-end latency <10s
**Constraints**: Requires network connectivity for LLM API, GPU recommended for local Whisper
**Scale/Scope**: Single humanoid robot, single-user voice interaction, simulation environment

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Verification

**Technical Accuracy**: All Whisper, LLM, and ROS 2 implementations will be verified against official OpenAI and ROS documentation.

**Clean, Modular Docusaurus Documentation**: All content will be structured in clean, modular MDX format suitable for Docusaurus documentation with consistent styling.

**Code Correctness and API Alignment**: All code implementations will align with OpenAI API specifications and ROS 2 Humble patterns, with proper error handling.

**AI-Native Creation using Spec-Kit Plus + Claude Code**: Implementation will leverage Spec-Kit Plus for specification-driven development.

**Verification Standards**: All voice recognition and LLM claims will be verified through testing. All code examples will be tested and proven runnable.

**Uniform Terminology Across Modules**: Consistent terminology will be maintained with Modules 1-3 (ROS 2, Gazebo/Unity, Isaac).

### Gate Status
- [x] All constitution principles addressed in implementation approach
- [x] Technical accuracy verification planned for Whisper/LLM components
- [x] Docusaurus MDX format compliance confirmed
- [x] Cross-module terminology consistency maintained

## Project Structure

### Documentation (this feature)

```text
specs/004-vision-language-action/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API contracts)
│   ├── whisper-node.md
│   ├── llm-planner.md
│   └── action-executor.md
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
examples/vla/
├── whisper/                    # Voice-to-Action (Chapter 1)
│   ├── config/
│   │   └── whisper_config.yaml
│   ├── launch/
│   │   └── whisper_pipeline.launch.py
│   └── nodes/
│       ├── audio_capture.py
│       ├── whisper_transcriber.py
│       └── command_parser.py
├── llm-planner/                # Cognitive Planning (Chapter 2)
│   ├── config/
│   │   ├── llm_config.yaml
│   │   └── action_primitives.yaml
│   ├── launch/
│   │   └── planner_pipeline.launch.py
│   └── nodes/
│       ├── llm_planner.py
│       ├── context_manager.py
│       └── plan_executor.py
├── capstone/                   # Autonomous Humanoid (Chapter 3)
│   ├── config/
│   │   └── capstone_config.yaml
│   ├── launch/
│   │   └── autonomous_humanoid.launch.py
│   └── scenarios/
│       ├── fetch_object.py
│       ├── navigate_and_report.py
│       └── multi_step_task.py
└── common/                     # Shared utilities
    ├── action_primitives.py
    ├── speech_synthesis.py
    └── error_handling.py

src/vla/                        # Core VLA library
├── __init__.py
├── whisper_client.py           # Whisper API wrapper
├── llm_client.py               # LLM API wrapper
├── action_planner.py           # Plan generation
├── task_executor.py            # Plan execution
└── integration.py              # Module 1-3 integration

docs/modules/vla/
├── voice-to-action.mdx         # Chapter 1
├── cognitive-planning.mdx      # Chapter 2
└── autonomous-humanoid.mdx     # Chapter 3
```

**Structure Decision**: Multi-component ROS 2 system organized by chapter/capability. Core library in `src/vla/` for reusable components, examples in `examples/vla/` for chapter-specific implementations.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VLA Pipeline                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Audio      │    │   Whisper    │    │   Command Parser     │  │
│  │   Capture    │───▶│   ASR        │───▶│   (Text → Intent)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                    │                 │
│                                                    ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    LLM Planner (GPT-4)                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │   Context   │  │   Action    │  │   Plan Generation   │   │  │
│  │  │   Manager   │──│   Library   │──│   & Validation      │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Action Executor                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │   Nav2      │  │  Perception │  │   Manipulation      │   │  │
│  │  │ (Module 2)  │  │  (Module 3) │  │   Actions           │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│                                ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Feedback System                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │   Speech    │  │   Status    │  │   Error Recovery    │   │  │
│  │  │   Synthesis │  │   Display   │  │   & Reporting       │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Action Primitives

The system supports the following action primitives that can be composed into plans:

| Primitive | Parameters | Description |
|-----------|------------|-------------|
| `navigate_to` | location: str, pose: Pose | Navigate to a named location or pose |
| `look_at` | target: str, direction: Vector3 | Orient sensors toward target |
| `scan_environment` | area: str | Perform environmental scan |
| `identify_object` | object_name: str, category: str | Detect and localize object |
| `pick_up` | object_id: str | Grasp and lift object |
| `place` | location: str, pose: Pose | Place held object |
| `say` | text: str | Speak text via TTS |
| `wait` | duration: float | Pause execution |
| `cancel` | - | Cancel current action |

## Integration Points

### Module 2 (Gazebo/Unity Simulation)
- Navigation goals sent via Nav2 action interface
- Robot state received from simulation
- Environment configuration for scenarios

### Module 3 (Isaac AI Brain)
- Perception pipeline for object detection
- VSLAM for localization
- Sensor fusion for state estimation

### External APIs
- OpenAI Whisper API or local whisper model
- OpenAI GPT-4 API for planning
- Optional: Local LLM fallback (Ollama/llama.cpp)

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM API latency | Medium | High | Implement caching, local LLM fallback |
| Whisper accuracy in noise | Medium | Medium | Noise filtering, confidence thresholds |
| Network connectivity loss | Low | High | Graceful degradation, offline mode |
| Invalid action plans | Medium | Medium | Plan validation, safety checks |

## Complexity Tracking

> **No constitution violations requiring justification**

## Next Steps

1. Create research.md with API documentation review
2. Create data-model.md with message definitions
3. Create quickstart.md with setup instructions
4. Create API contracts for each component
5. Generate tasks.md with implementation checklist
