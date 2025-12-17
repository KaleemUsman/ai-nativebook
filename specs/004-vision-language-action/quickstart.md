# Quickstart Guide: Vision-Language-Action (VLA)

## Prerequisites

### System Requirements

- Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- Python 3.10 or higher
- ROS 2 Humble Hawksbill installed
- Microphone input device for voice commands
- GPU recommended for local Whisper (CUDA-compatible)
- Internet connection for OpenAI API access

### API Keys Required

- **OpenAI API Key**: For Whisper API (optional) and GPT-4 planning
- Set up your API key:
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```

### Dependencies from Previous Modules

- Module 2 (Gazebo/Unity): Simulation environment configured
- Module 3 (Isaac AI Brain): Perception pipelines and Nav2 navigation stack

---

## Installation

### 1. Clone and Set Up the VLA Package

```bash
# Navigate to your ROS 2 workspace
cd ~/ros2_ws/src

# The VLA examples are already in the repository
# Ensure you have the latest code
git pull origin main

# Install Python dependencies
pip install openai whisper sounddevice pyttsx3 numpy scipy
```

### 2. Build the ROS 2 Workspace

```bash
cd ~/ros2_ws
colcon build --packages-select vla_msgs vla_nodes
source install/setup.bash
```

### 3. Verify Installation

```bash
# Check that Whisper is installed
python3 -c "import whisper; print(whisper.__version__)"

# Check OpenAI connection
python3 -c "import openai; print('OpenAI SDK ready')"

# Verify ROS 2 nodes can be found
ros2 pkg list | grep vla
```

---

## Quick Start: Voice-to-Action Pipeline

### Step 1: Launch the Whisper Node

```bash
# Terminal 1: Launch the voice capture and transcription pipeline
ros2 launch examples/vla/whisper/launch/whisper_pipeline.launch.py
```

**Expected Output:**
```
[audio_capture]: Listening on default microphone...
[whisper_transcriber]: Loaded whisper model: small
[command_parser]: Ready to parse voice commands
```

### Step 2: Test Voice Input

```bash
# Terminal 2: Monitor transcriptions
ros2 topic echo /vla/transcription

# Speak a command like: "Go to the kitchen"
# Expected output:
# transcription: "Go to the kitchen"
# confidence: 0.97
# language: "en"
```

### Step 3: Verify Command Parsing

```bash
# Monitor parsed intents
ros2 topic echo /vla/parsed_intent

# For "Go to the kitchen", expect:
# intent_type: "NAVIGATION"
# action_verb: "go"
# target_location: "kitchen"
```

---

## Quick Start: LLM Planning

### Step 1: Configure the LLM Planner

```bash
# Create or edit the configuration file
nano examples/vla/llm-planner/config/llm_config.yaml
```

**Minimal Configuration:**
```yaml
llm_config:
  model: "gpt-4"
  temperature: 0.2
  max_tokens: 1024
  timeout_s: 10.0
  
action_primitives:
  enabled:
    - navigate_to
    - look_at
    - pick_up
    - place
    - scan_environment
    - identify_object
    - say
    - wait
```

### Step 2: Launch the Planner

```bash
# Terminal 3: Launch the LLM planning pipeline
ros2 launch examples/vla/llm-planner/launch/planner_pipeline.launch.py
```

### Step 3: Test Planning

```bash
# Send a test intent to the planner
ros2 action send_goal /vla/plan_task vla_msgs/action/PlanTask \
  "intent: {intent_type: 'MANIPULATION', action_verb: 'pick up', target_object: 'cup', target_location: 'table'}"

# Expected: LLM generates a plan with primitives:
# 1. navigate_to(table)
# 2. scan_environment(table_area)
# 3. identify_object(cup)
# 4. pick_up(cup_001)
```

---

## Quick Start: Full Pipeline Demo

### Prerequisites

Ensure you have completed:
- Module 2 setup (Gazebo simulation running)
- Module 3 setup (Perception and navigation active)

### Step 1: Launch the Complete VLA System

```bash
# Terminal 1: Launch simulation (from Module 2)
ros2 launch gazebo_unity humanoid_sim.launch.py

# Terminal 2: Launch perception (from Module 3)
ros2 launch examples/gazebo-unity/isaac-ros/launch/perception_pipeline.launch.py

# Terminal 3: Launch navigation (from Module 3)
ros2 launch examples/gazebo-unity/nav2/launch/navigation.launch.py

# Terminal 4: Launch complete VLA pipeline
ros2 launch examples/vla/capstone/launch/autonomous_humanoid.launch.py
```

### Step 2: Issue a Voice Command

Speak clearly: **"Go to the desk and pick up the phone"**

### Step 3: Observe the Execution

```bash
# Monitor task execution
ros2 topic echo /vla/task_status

# Monitor robot feedback
ros2 topic echo /vla/speech_output
```

**Expected Behavior:**
1. Robot verbalizes: "Understood. Planning to fetch the phone from the desk."
2. Robot navigates to the desk
3. Robot scans for the phone
4. Robot picks up the phone
5. Robot reports: "I have the phone. What should I do with it?"

---

## Configuration Files

### Voice Pipeline (`whisper_config.yaml`)

```yaml
audio:
  device_id: null  # null = default microphone
  sample_rate: 16000
  chunk_duration_ms: 100
  vad_enabled: true
  noise_threshold_db: -30

whisper:
  model_size: "small"  # tiny, base, small, medium, large-v3
  language: "en"       # null for auto-detect
  local_model: true    # false to use API
  device: "cuda"       # cuda, cpu

parser:
  confidence_threshold: 0.5
  unknown_intent_handling: "ask_clarification"
```

### LLM Planner (`llm_config.yaml`)

```yaml
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.2
  max_tokens: 1024
  timeout_s: 10.0
  fallback_model: "ollama/llama3"

context:
  max_history_items: 10
  include_robot_state: true
  include_detected_objects: true
  include_known_locations: true

planning:
  validate_preconditions: true
  require_confirmation_for: ["pick_up", "place"]
  max_plan_length: 15
```

---

## Troubleshooting

### Voice Recognition Issues

| Issue | Solution |
|-------|----------|
| No audio detected | Check microphone permissions and device ID |
| Low accuracy | Increase model size or reduce background noise |
| High latency | Use smaller model or enable GPU acceleration |
| "No speech detected" | Adjust VAD sensitivity in config |

### LLM Planning Issues

| Issue | Solution |
|-------|----------|
| API errors | Verify OPENAI_API_KEY is set correctly |
| Timeout | Increase timeout_s or check network connection |
| Invalid plans | Check action_primitives config matches capabilities |
| "Cannot plan" | Verify context manager is receiving perception data |

### Integration Issues

| Issue | Solution |
|-------|----------|
| Actions not executing | Verify Nav2 and perception nodes are running |
| Object not found | Check perception pipeline and detection confidence |
| Navigation fails | Verify costmap and goal poses are valid |

---

## Next Steps

- Follow the detailed documentation in `docs/modules/vla/`
- Explore the capstone scenarios in `examples/vla/capstone/scenarios/`
- Customize action primitives for your robot platform
- Add domain-specific vocabulary to improve command parsing
