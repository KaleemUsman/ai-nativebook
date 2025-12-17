# LLM Planner API Contract

## Overview

The LLM Planner translates natural language intents into structured action plans using GPT-4 or compatible LLMs, providing the "cognitive" layer for autonomous humanoid task execution.

## Node Information

| Property | Value |
|----------|-------|
| **Node Name** | `llm_planner` |
| **Package** | `vla_nodes` |
| **Language** | Python 3.10+ |
| **Dependencies** | `openai`, `pydantic`, `httpx` |

---

## ROS 2 Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vla/parsed_intent` | `vla_msgs/ParsedIntent` | Parsed user intents from command parser |
| `/vla/task_context` | `vla_msgs/TaskContext` | Environmental context for planning |
| `/vla/detected_objects` | `vision_msgs/Detection3DArray` | Current object detections |
| `/tf` | `tf2_msgs/TFMessage` | Robot and object transforms |

### Published Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/vla/action_plan` | `vla_msgs/ActionPlan` | Generated action plans |
| `/vla/planner_status` | `std_msgs/String` | Planner status updates |
| `/vla/plan_explanation` | `std_msgs/String` | Human-readable plan explanation |

### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/vla/plan_task` | `vla_msgs/action/PlanTask` | Request action plan for an intent |
| `/vla/replan` | `vla_msgs/action/Replan` | Request replanning after failure |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/vla/planner/validate_plan` | `vla_srvs/ValidatePlan` | Validate a proposed plan |
| `/vla/planner/get_capabilities` | `vla_srvs/GetCapabilities` | List available action primitives |
| `/vla/planner/set_model` | `std_srvs/SetString` | Change LLM model |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm.provider` | string | `"openai"` | LLM provider: openai, anthropic, ollama |
| `llm.model` | string | `"gpt-4"` | Model name |
| `llm.temperature` | float | `0.2` | Generation temperature (0.0-1.0) |
| `llm.max_tokens` | int | `1024` | Maximum response tokens |
| `llm.timeout_s` | float | `10.0` | API timeout in seconds |
| `llm.fallback_model` | string | `""` | Fallback model on primary failure |
| `planning.validate_plans` | bool | `true` | Validate plans before publishing |
| `planning.max_plan_length` | int | `15` | Maximum primitives per plan |
| `planning.require_confirmation` | list | `["pick_up", "place"]` | Actions requiring confirmation |
| `context.include_history` | bool | `true` | Include task history in prompts |
| `context.max_history` | int | `10` | Maximum history items |

---

## Message Schemas

### Input: ParsedIntent

```yaml
# vla_msgs/ParsedIntent.msg
std_msgs/Header header
string intent_id                # Unique UUID
string source_command_id        # Reference to source VoiceCommand
string intent_type              # NAVIGATION, MANIPULATION, QUERY, CANCEL, UNKNOWN
string action_verb              # Primary action verb
string target_object            # Target object name
string target_location          # Target location name
string[] modifiers              # Descriptive modifiers
string urgency                  # LOW, NORMAL, HIGH
float32 parse_confidence        # Parsing confidence (0.0-1.0)
```

### Input: TaskContext

```yaml
# vla_msgs/TaskContext.msg
std_msgs/Header header
geometry_msgs/Pose robot_pose   # Current robot pose
string gripper_state            # OPEN, CLOSED, HOLDING
string held_object_id           # ID of held object (empty if none)
vla_msgs/DetectedObject[] detected_objects
vla_msgs/NamedLocation[] known_locations
float32 battery_level           # 0.0-1.0
string[] active_capabilities
vla_msgs/ExecutionResult[] recent_history
```

### Output: ActionPlan

```yaml
# vla_msgs/ActionPlan.msg
std_msgs/Header header
string plan_id                  # Unique UUID
string source_intent_id         # Reference to intent
vla_msgs/ActionPrimitive[] primitives  # Ordered action sequence
float32 estimated_duration_s    # Estimated execution time
string[] preconditions          # Required preconditions
string plan_status              # PENDING, EXECUTING, COMPLETED, FAILED, CANCELLED
uint8 current_step_index        # Current execution position
string explanation              # Human-readable plan explanation
```

### ActionPrimitive Detail

```yaml
# vla_msgs/ActionPrimitive.msg
string primitive_id             # Unique UUID
string action_type              # navigate_to, pick_up, etc.
string parameters_json          # JSON-encoded parameters
string[] preconditions          # Pre-execution conditions
string[] expected_outcomes      # Post-execution expectations
float32 timeout_s               # Maximum execution time
uint8 retry_count               # Retry attempts on failure
string status                   # PENDING, EXECUTING, SUCCEEDED, FAILED
string error_message            # Error details if failed
```

---

## Action Definitions

### PlanTask Action

```yaml
# vla_msgs/action/PlanTask.action

# Goal
vla_msgs/ParsedIntent intent
vla_msgs/TaskContext context
bool require_explanation

---
# Result
bool success
vla_msgs/ActionPlan plan
string error_message
float32 planning_time_s

---
# Feedback
string current_stage            # "analyzing", "generating", "validating"
float32 progress                # 0.0-1.0
string status_message
```

### Replan Action

```yaml
# vla_msgs/action/Replan.action

# Goal
string failed_plan_id
string failed_primitive_id
string failure_reason
vla_msgs/TaskContext current_context

---
# Result
bool success
vla_msgs/ActionPlan new_plan
string error_message

---
# Feedback
string stage
string status_message
```

---

## Behavior Specification

### Planning Flow

```
Intent Received → Context Assembly → Prompt Construction → 
LLM Call → Response Parsing → Plan Validation → Publish Plan
```

### Prompt Template Structure

```
System Prompt:
- Role definition (robot action planner)
- Available action primitives with parameters
- Current robot capabilities

User Prompt:
- Current intent (action, target, location)
- Environmental context (objects, locations, robot state)
- Task history (recent commands and outcomes)
- Constraints and safety requirements

Function Call Schema:
- create_action_plan(primitives: List[ActionPrimitive])
```

### Plan Validation

Before publishing, plans are validated for:

1. **Structural validity**: All primitives have required fields
2. **Precondition feasibility**: Preconditions can be satisfied by previous steps
3. **Physical feasibility**: Actions are within robot capabilities
4. **Safety checks**: No dangerous action sequences
5. **Completeness**: Plan addresses the original intent

### Error Handling

| Error Condition | Behavior | Recovery |
|----------------|----------|----------|
| LLM API timeout | Retry up to 3 times | Fall back to `fallback_model` |
| Invalid JSON response | Log and retry with clarified prompt | Return error |
| Impossible plan | Validate, then replan with constraints | Ask user for guidance |
| Unknown primitives | Strip invalid primitives | Log warning |
| Rate limit exceeded | Exponential backoff | Queue requests |

### Context Management

The planner maintains context through:

1. **Object Registry**: Objects from perception + confidence decay over time
2. **Location Memory**: Named locations from Nav2 semantic map
3. **History Buffer**: Last N execution results for grounding
4. **Robot State**: Updated from TF and sensor topics

---

## LLM Integration

### OpenAI GPT-4 Configuration

```python
# Example function calling schema
tools = [{
    "type": "function",
    "function": {
        "name": "create_action_plan",
        "description": "Create a sequence of robot actions to accomplish a task",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of the plan"
                },
                "primitives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {"type": "string", "enum": [
                                "navigate_to", "look_at", "scan_environment",
                                "identify_object", "pick_up", "place", "say", "wait"
                            ]},
                            "parameters": {"type": "object"},
                            "preconditions": {"type": "array", "items": {"type": "string"}},
                            "expected_outcomes": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["action_type", "parameters"]
                    }
                }
            },
            "required": ["explanation", "primitives"]
        }
    }
}]
```

### Ollama Fallback Configuration

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "llama3"
  num_ctx: 8192
  temperature: 0.2
```

---

## Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| Planning latency | <5s | Including LLM API call |
| Context assembly | <100ms | From cached data |
| Plan validation | <200ms | Local validation |
| Memory usage | <500MB | Node process |

---

## Testing

### Unit Tests

```bash
pytest examples/vla/llm-planner/tests/test_planner.py -v
```

### Integration Tests

```bash
# Test with mock LLM
ros2 launch examples/vla/llm-planner/launch/planner_test.launch.py \
  use_mock_llm:=true

# Test with real LLM
ros2 launch examples/vla/llm-planner/launch/planner_test.launch.py
```

### Acceptance Criteria

- [ ] 90%+ of plans are executable (valid action sequences)
- [ ] Planning latency <5s (excluding network latency)
- [ ] Handles ambiguous commands with clarification request
- [ ] Graceful fallback to local LLM when API unavailable
