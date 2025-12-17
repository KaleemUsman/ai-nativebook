# Action Executor API Contract

## Overview

The Action Executor is responsible for translating action plans into actual robot behaviors, coordinating with Nav2, perception pipelines, and manipulation subsystems to execute tasks autonomously.

## Node Information

| Property | Value |
|----------|-------|
| **Node Name** | `action_executor` |
| **Package** | `vla_nodes` |
| **Language** | Python 3.10+ |
| **Dependencies** | `nav2_msgs`, `moveit_msgs`, `tf2_ros` |

---

## ROS 2 Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vla/action_plan` | `vla_msgs/ActionPlan` | Plans to execute |
| `/vla/cancel_request` | `std_msgs/String` | Cancel current execution |
| `/tf` | `tf2_msgs/TFMessage` | Transform updates |
| `/robot/state` | `sensor_msgs/JointState` | Robot joint states |
| `/perception/objects` | `vision_msgs/Detection3DArray` | Object detections |

### Published Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/vla/execution_status` | `vla_msgs/ExecutionStatus` | Current execution state |
| `/vla/execution_result` | `vla_msgs/ExecutionResult` | Completed execution results |
| `/vla/speech_output` | `std_msgs/String` | Text for speech synthesis |
| `/vla/primitive_feedback` | `vla_msgs/PrimitiveFeedback` | Per-primitive feedback |

### Action Clients (Used by this node)

| Action | Type | Description |
|--------|------|-------------|
| `/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | Nav2 navigation |
| `/compute_path_to_pose` | `nav2_msgs/action/ComputePathToPose` | Path planning |
| `/follow_waypoints` | `nav2_msgs/action/FollowWaypoints` | Waypoint following |
| `/move_group` | `moveit_msgs/action/MoveGroup` | Arm manipulation |
| `/gripper_command` | `control_msgs/action/GripperCommand` | Gripper control |

### Action Servers (Provided by this node)

| Action | Type | Description |
|--------|------|-------------|
| `/vla/execute_plan` | `vla_msgs/action/ExecutePlan` | Execute a complete action plan |
| `/vla/execute_primitive` | `vla_msgs/action/ExecutePrimitive` | Execute a single primitive |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/vla/executor/pause` | `std_srvs/Trigger` | Pause current execution |
| `/vla/executor/resume` | `std_srvs/Trigger` | Resume paused execution |
| `/vla/executor/abort` | `std_srvs/Trigger` | Abort and cleanup |
| `/vla/executor/get_status` | `vla_srvs/GetExecutorStatus` | Get detailed status |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `execution.max_retries` | int | `3` | Maximum retries per primitive |
| `execution.primitive_timeout_s` | float | `60.0` | Default primitive timeout |
| `execution.abort_on_failure` | bool | `false` | Abort plan on first failure |
| `execution.require_confirmation` | list | `[]` | Primitives requiring user confirmation |
| `navigation.goal_tolerance_m` | float | `0.1` | Navigation position tolerance |
| `navigation.yaw_tolerance_rad` | float | `0.1` | Navigation orientation tolerance |
| `manipulation.grasp_timeout_s` | float | `30.0` | Grasp operation timeout |
| `feedback.publish_rate_hz` | float | `10.0` | Feedback publication rate |
| `safety.max_velocity` | float | `0.5` | Maximum robot velocity (m/s) |
| `safety.collision_check` | bool | `true` | Enable collision checking |

---

## Message Schemas

### ExecutePlan Action

```yaml
# vla_msgs/action/ExecutePlan.action

# Goal
vla_msgs/ActionPlan plan
bool skip_confirmation           # Skip confirmation for all primitives
bool continue_on_failure         # Continue even if primitives fail

---
# Result
bool success
string plan_id
uint8 completed_primitives
uint8 failed_primitives
vla_msgs/ExecutionResult[] primitive_results
string final_status              # COMPLETED, PARTIAL, FAILED, ABORTED
float32 total_duration_s

---
# Feedback
string plan_id
uint8 current_primitive_index
uint8 total_primitives
string current_action_type
string current_status            # EXECUTING, RETRYING, WAITING_CONFIRMATION
float32 progress                 # 0.0-1.0
string status_message
```

### ExecutePrimitive Action

```yaml
# vla_msgs/action/ExecutePrimitive.action

# Goal
vla_msgs/ActionPrimitive primitive
vla_msgs/TaskContext context

---
# Result
bool success
vla_msgs/ExecutionResult result
string[] state_changes           # Post-execution state changes

---
# Feedback
string primitive_id
string status
float32 progress
string[] status_messages
```

### ExecutionStatus Message

```yaml
# vla_msgs/ExecutionStatus.msg
std_msgs/Header header
string plan_id
string executor_state            # IDLE, EXECUTING, PAUSED, ERROR
string current_primitive_id
uint8 current_step_index
uint8 total_steps
float32 progress
string status_message
builtin_interfaces/Duration elapsed_time
builtin_interfaces/Duration estimated_remaining
```

### ExecutionResult Message

```yaml
# vla_msgs/ExecutionResult.msg
std_msgs/Header header
string result_id
string source_id                 # Plan or primitive ID
bool success
int32 status_code
string error_type                # NONE, TIMEOUT, PRECONDITION_FAILED, EXECUTION_FAILED, ABORTED
string error_message
builtin_interfaces/Time start_time
builtin_interfaces/Time end_time
float32 actual_duration_s
string[] feedback_messages
string return_data_json          # Action-specific return data
```

---

## Primitive Execution Specifications

### navigate_to

**Parameters:**
```json
{
  "location": "kitchen",       // Named location OR
  "pose": {                    // Explicit pose
    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}
  },
  "approach_distance": 0.5     // Optional: stop this far from target
}
```

**Execution:**
1. Resolve location name to pose (if using named location)
2. Send goal to `/navigate_to_pose`
3. Monitor navigation progress
4. Verify final pose within tolerance

**Success Criteria:** Robot within `goal_tolerance_m` of target

---

### look_at

**Parameters:**
```json
{
  "target": "cup",             // Object name OR
  "target_id": "cup_001",      // Specific object ID OR
  "direction": {"x": 1.0, "y": 0, "z": 0},  // Direction vector
  "duration_s": 2.0            // How long to look
}
```

**Execution:**
1. Resolve target to 3D position
2. Calculate required head/body orientation
3. Command joint controllers
4. Verify sensor field-of-view includes target

**Success Criteria:** Target visible in perception data

---

### scan_environment

**Parameters:**
```json
{
  "area": "table_area",        // Named area to scan
  "mode": "full",              // quick, full, detailed
  "object_classes": ["cup", "phone"]  // Optional: specific classes to find
}
```

**Execution:**
1. Execute scanning motion pattern based on mode
2. Aggregate perception detections
3. Update object registry with findings

**Return Data:** List of detected objects with positions

---

### identify_object

**Parameters:**
```json
{
  "object_name": "cup",        // Object class name
  "color": "red",              // Optional: color filter
  "location_hint": "table"     // Optional: where to look first
}
```

**Execution:**
1. Query perception pipeline for matching objects
2. If not found, scan environment
3. Select best match based on criteria

**Return Data:** Object ID and pose of identified object

---

### pick_up

**Parameters:**
```json
{
  "object_id": "cup_001",      // Object to pick up
  "grasp_type": "power",       // power, precision, pinch
  "approach_direction": "top"  // top, side, front
}
```

**Execution:**
1. Verify object is graspable and reachable
2. Plan approach trajectory
3. Execute approach motion
4. Close gripper with force feedback
5. Verify grasp success (force sensor / weight check)
6. Execute lift motion

**Success Criteria:** Gripper holding object, force sensors confirm grasp

---

### place

**Parameters:**
```json
{
  "location": "table",         // Named location OR
  "pose": {...},               // Explicit placement pose
  "place_style": "gentle"      // drop, gentle, precise
}
```

**Execution:**
1. Navigate to placement location if needed
2. Plan placement trajectory
3. Execute placement motion
4. Open gripper
5. Retract arm

**Success Criteria:** Object detected at placement location

---

### say

**Parameters:**
```json
{
  "text": "I found the cup on the table",
  "priority": "normal",        // low, normal, high
  "wait_for_completion": true
}
```

**Execution:**
1. Publish text to `/vla/speech_output`
2. If `wait_for_completion`, wait for TTS to finish

---

### wait

**Parameters:**
```json
{
  "duration": 5.0,             // Seconds to wait OR
  "condition": "object_stable" // Condition to wait for
}
```

**Execution:**
1. If duration: sleep for specified time
2. If condition: poll condition until true or timeout

---

## Error Handling

### Error Taxonomy

| Error Type | Code | Description | Recovery |
|------------|------|-------------|----------|
| `NONE` | 0 | No error | N/A |
| `TIMEOUT` | 1 | Operation exceeded timeout | Retry or abort |
| `PRECONDITION_FAILED` | 2 | Preconditions not met | Check and report |
| `EXECUTION_FAILED` | 3 | Action execution failed | Retry with different parameters |
| `NAVIGATION_FAILED` | 10 | Nav2 goal failed | Replan path or abort |
| `MANIPULATION_FAILED` | 20 | Grasp/place failed | Retry grasp or abort |
| `OBJECT_NOT_FOUND` | 30 | Target object not detected | Scan or ask user |
| `COLLISION_DETECTED` | 40 | Motion would cause collision | Replan or abort |
| `HARDWARE_ERROR` | 50 | Hardware malfunction | Safety stop, report |
| `USER_ABORTED` | 60 | User cancelled execution | Cleanup and stop |

### Retry Logic

```python
for attempt in range(max_retries):
    result = execute_primitive(primitive)
    if result.success:
        return result
    
    if result.error_type == PRECONDITION_FAILED:
        # Try to satisfy preconditions
        handle_precondition_failure(primitive, result)
    elif result.error_type in [TIMEOUT, EXECUTION_FAILED]:
        # Adjust parameters and retry
        primitive = adjust_parameters(primitive, attempt)
    else:
        # Non-recoverable error
        break
```

---

## State Machine

```
IDLE ──[receive plan]──> VALIDATING ──[valid]──> EXECUTING
  ^                           |                      |
  |                      [invalid]            [primitive complete]
  |                           v                      |
  |                        ERROR                     v
  |                           |              NEXT_PRIMITIVE
  |                      [reported]                  |
  |                           |              [all done]
  └───────────────────────────┴──────────────────────┘
                                                     
EXECUTING ──[pause request]──> PAUSED ──[resume]──> EXECUTING
     |                            |
     └──[abort request]──> ABORTING ──> IDLE
```

---

## Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| Plan startup latency | <500ms | From receipt to first primitive |
| Primitive dispatch | <100ms | Time to start primitive action |
| Status update rate | 10 Hz | Feedback publication |
| Memory usage | <300MB | Node process |

---

## Testing

### Unit Tests

```bash
pytest examples/vla/capstone/tests/test_executor.py -v
```

### Integration Tests

```bash
# Test with simulation
ros2 launch examples/vla/capstone/launch/executor_test.launch.py \
  use_simulation:=true
```

### Acceptance Criteria

- [ ] 80%+ single-object fetch task success rate
- [ ] Average task completion <60s
- [ ] Proper error reporting for 95%+ of failures
- [ ] Clean abort handling with state restoration
