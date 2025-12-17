#!/usr/bin/env python3
"""
Plan Executor Node for VLA Pipeline

Executes action plans by coordinating with navigation, perception, and manipulation
subsystems. Handles retries, error recovery, and status reporting.

Subscribes to:
    - /vla/action_plan (ActionPlan)
    
Publishes to:
    - /vla/execution_status (ExecutionStatus)
    - /vla/execution_result (ExecutionResult)
    - /vla/speech_output (String)

Action Servers:
    - /vla/execute_plan (ExecutePlan)
    - /vla/execute_primitive (ExecutePrimitive)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

import json
import time
import uuid
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from threading import Thread, Event
import queue


class PrimitiveStatus(Enum):
    """Status of primitive execution"""
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"


class PlanStatus(Enum):
    """Status of plan execution"""
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class ExecutionStatus:
    """Current execution status"""
    plan_id: str
    status: str
    current_primitive_index: int
    total_primitives: int
    current_action_type: str
    progress: float
    message: str
    elapsed_time_s: float
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ExecutionResult:
    """Result of execution"""
    result_id: str
    source_id: str
    success: bool
    status_code: int
    error_type: str
    error_message: str
    start_time: str
    end_time: str
    actual_duration_s: float
    feedback_messages: List[str]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class PrimitiveExecutor:
    """
    Executes individual action primitives.
    
    In a full implementation, this would interface with actual robot subsystems.
    Here we provide simulation and ROS 2 action client patterns.
    """
    
    def __init__(self, node: 'PlanExecutorNode'):
        self.node = node
        self.logger = node.get_logger()
        
        # Primitive handlers
        self.handlers: Dict[str, Callable] = {
            'navigate_to': self._execute_navigate_to,
            'look_at': self._execute_look_at,
            'scan_environment': self._execute_scan_environment,
            'identify_object': self._execute_identify_object,
            'pick_up': self._execute_pick_up,
            'place': self._execute_place,
            'say': self._execute_say,
            'wait': self._execute_wait,
            'cancel': self._execute_cancel,
        }
    
    def execute(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        timeout_s: float = 30.0
    ) -> tuple[bool, str]:
        """
        Execute a primitive action.
        
        Returns:
            Tuple of (success, error_message)
        """
        if action_type not in self.handlers:
            return False, f"Unknown action type: {action_type}"
        
        try:
            handler = self.handlers[action_type]
            return handler(parameters, timeout_s)
        except Exception as e:
            self.logger.error(f"Primitive execution error: {e}")
            return False, str(e)
    
    def _execute_navigate_to(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute navigation primitive"""
        location = params.get('location')
        pose = params.get('pose')
        
        if not location and not pose:
            return False, "navigate_to requires 'location' or 'pose'"
        
        self.logger.info(f"Navigating to: {location or pose}")
        
        # Simulate navigation
        # In production: send goal to Nav2 action server
        time.sleep(2.0)  # Simulate travel time
        
        return True, ""
    
    def _execute_look_at(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute look_at primitive"""
        target = params.get('target') or params.get('target_id')
        
        self.logger.info(f"Looking at: {target}")
        
        # Simulate head movement
        time.sleep(0.5)
        
        return True, ""
    
    def _execute_scan_environment(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute scan_environment primitive"""
        mode = params.get('mode', 'full')
        
        self.logger.info(f"Scanning environment (mode: {mode})")
        
        # Simulate scanning
        durations = {'quick': 1.0, 'full': 3.0, 'detailed': 5.0}
        time.sleep(durations.get(mode, 2.0))
        
        return True, ""
    
    def _execute_identify_object(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute identify_object primitive"""
        object_name = params.get('object_name')
        
        if not object_name:
            return False, "identify_object requires 'object_name'"
        
        self.logger.info(f"Identifying object: {object_name}")
        
        # Simulate object detection
        time.sleep(1.5)
        
        # Simulate detection result (80% success rate)
        import random
        if random.random() < 0.8:
            return True, ""
        else:
            return False, f"Object '{object_name}' not found"
    
    def _execute_pick_up(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute pick_up primitive"""
        object_id = params.get('object_id')
        grasp_type = params.get('grasp_type', 'power')
        
        if not object_id:
            return False, "pick_up requires 'object_id'"
        
        self.logger.info(f"Picking up object: {object_id} (grasp: {grasp_type})")
        
        # Simulate grasp
        time.sleep(3.0)
        
        return True, ""
    
    def _execute_place(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute place primitive"""
        location = params.get('location')
        pose = params.get('pose')
        
        self.logger.info(f"Placing object at: {location or pose}")
        
        # Simulate placement
        time.sleep(2.0)
        
        return True, ""
    
    def _execute_say(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute say primitive"""
        text = params.get('text')
        
        if not text:
            return False, "say requires 'text'"
        
        self.logger.info(f"Speaking: '{text}'")
        
        # Publish to speech output
        msg = String()
        msg.data = text
        self.node.speech_pub.publish(msg)
        
        # Simulate TTS duration (rough estimate)
        estimated_duration = len(text) * 0.05  # ~50ms per character
        time.sleep(min(estimated_duration, 5.0))
        
        return True, ""
    
    def _execute_wait(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute wait primitive"""
        duration = params.get('duration', 1.0)
        
        self.logger.info(f"Waiting for {duration}s")
        
        time.sleep(min(duration, timeout))
        
        return True, ""
    
    def _execute_cancel(self, params: Dict, timeout: float) -> tuple[bool, str]:
        """Execute cancel primitive"""
        self.logger.info("Cancelling current action")
        
        # In production: send cancel to all active action clients
        
        return True, ""


class PlanExecutorNode(Node):
    """
    ROS 2 node for executing action plans.
    
    Coordinates primitive execution, handles retries, and reports status.
    """
    
    def __init__(self):
        super().__init__('plan_executor')
        
        # Declare parameters
        self.declare_parameter('execution.max_retries', 3)
        self.declare_parameter('execution.primitive_timeout_s', 60.0)
        self.declare_parameter('execution.abort_on_failure', False)
        self.declare_parameter('feedback.publish_rate_hz', 10.0)
        
        self.max_retries = self.get_parameter('execution.max_retries').value
        self.primitive_timeout = self.get_parameter('execution.primitive_timeout_s').value
        self.abort_on_failure = self.get_parameter('execution.abort_on_failure').value
        publish_rate = self.get_parameter('feedback.publish_rate_hz').value
        
        # State
        self.current_plan: Optional[Dict] = None
        self.current_step = 0
        self.plan_status = PlanStatus.PENDING
        self.execution_start_time: Optional[float] = None
        self.stop_event = Event()
        self.pause_event = Event()
        self.pause_event.set()  # Not paused by default
        
        # Primitive executor
        self.executor = PrimitiveExecutor(self)
        
        # Plan queue
        self.plan_queue = queue.Queue()
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.plan_sub = self.create_subscription(
            String,
            '/vla/action_plan',
            self._plan_callback,
            qos
        )
        
        self.cancel_sub = self.create_subscription(
            String,
            '/vla/cancel_request',
            self._cancel_callback,
            qos
        )
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/vla/execution_status', qos)
        self.result_pub = self.create_publisher(String, '/vla/execution_result', qos)
        self.speech_pub = self.create_publisher(String, '/vla/speech_output', qos)
        self.planner_status_pub = self.create_publisher(String, '/vla/planner_status', qos)
        
        # Execution thread
        self.execution_thread = Thread(target=self._execution_loop, daemon=True)
        self.is_running = True
        self.execution_thread.start()
        
        # Status timer
        self.status_timer = self.create_timer(1.0 / publish_rate, self._publish_status)
        
        self._log_status("Plan executor ready")
    
    def _log_status(self, message: str):
        """Log and publish status"""
        msg = String()
        msg.data = message
        self.planner_status_pub.publish(msg)
        self.get_logger().info(message)
    
    def _plan_callback(self, msg: String):
        """Handle incoming action plan"""
        try:
            plan = json.loads(msg.data)
            self.plan_queue.put(plan)
            self.get_logger().info(
                f"Received plan: {plan.get('plan_id', 'unknown')[:8]}... "
                f"with {len(plan.get('primitives', []))} steps"
            )
        except Exception as e:
            self.get_logger().error(f"Error parsing plan: {e}")
    
    def _cancel_callback(self, msg: String):
        """Handle cancel request"""
        self.get_logger().info("Cancel requested")
        self.stop_event.set()
    
    def _publish_status(self):
        """Publish current execution status"""
        if not self.current_plan:
            return
        
        elapsed = time.time() - self.execution_start_time if self.execution_start_time else 0
        total_steps = len(self.current_plan.get('primitives', []))
        progress = self.current_step / total_steps if total_steps > 0 else 0
        
        current_action = ""
        if self.current_step < total_steps:
            primitives = self.current_plan.get('primitives', [])
            current_action = primitives[self.current_step].get('action_type', '')
        
        status = ExecutionStatus(
            plan_id=self.current_plan.get('plan_id', ''),
            status=self.plan_status.value,
            current_primitive_index=self.current_step,
            total_primitives=total_steps,
            current_action_type=current_action,
            progress=progress,
            message=f"Executing step {self.current_step + 1}/{total_steps}",
            elapsed_time_s=elapsed
        )
        
        msg = String()
        msg.data = status.to_json()
        self.status_pub.publish(msg)
    
    def _execution_loop(self):
        """Background thread for plan execution"""
        while self.is_running:
            try:
                # Wait for plan
                try:
                    plan = self.plan_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Execute plan
                self._execute_plan(plan)
                
            except Exception as e:
                self.get_logger().error(f"Execution loop error: {e}")
    
    def _execute_plan(self, plan: Dict):
        """Execute a complete action plan"""
        self.current_plan = plan
        self.current_step = 0
        self.plan_status = PlanStatus.EXECUTING
        self.execution_start_time = time.time()
        self.stop_event.clear()
        
        primitives = plan.get('primitives', [])
        plan_id = plan.get('plan_id', str(uuid.uuid4()))
        
        self.get_logger().info(f"Executing plan: {plan_id[:8]}...")
        
        results = []
        failed = False
        
        for i, primitive in enumerate(primitives):
            self.current_step = i
            
            # Check for stop request
            if self.stop_event.is_set():
                self.plan_status = PlanStatus.ABORTED
                break
            
            # Wait if paused
            self.pause_event.wait()
            
            # Execute primitive with retries
            action_type = primitive.get('action_type', '')
            parameters = primitive.get('parameters', {})
            timeout = primitive.get('timeout_s', self.primitive_timeout)
            max_retries = primitive.get('retry_count', self.max_retries)
            
            success = False
            error_msg = ""
            
            for attempt in range(max_retries + 1):
                self.get_logger().info(
                    f"Step {i+1}/{len(primitives)}: {action_type}" +
                    (f" (attempt {attempt+1})" if attempt > 0 else "")
                )
                
                success, error_msg = self.executor.execute(
                    action_type, parameters, timeout
                )
                
                if success:
                    break
                
                if attempt < max_retries:
                    time.sleep(1.0)  # Brief delay before retry
            
            results.append({
                'action_type': action_type,
                'success': success,
                'error_message': error_msg
            })
            
            if not success:
                self.get_logger().warn(f"Step {i+1} failed: {error_msg}")
                if self.abort_on_failure:
                    failed = True
                    break
        
        # Determine final status
        if self.stop_event.is_set():
            self.plan_status = PlanStatus.ABORTED
        elif failed:
            self.plan_status = PlanStatus.FAILED
        else:
            self.plan_status = PlanStatus.COMPLETED
        
        # Publish result
        execution_time = time.time() - self.execution_start_time
        successful_steps = sum(1 for r in results if r['success'])
        
        result = ExecutionResult(
            result_id=str(uuid.uuid4()),
            source_id=plan_id,
            success=self.plan_status == PlanStatus.COMPLETED,
            status_code=0 if self.plan_status == PlanStatus.COMPLETED else 1,
            error_type="" if self.plan_status == PlanStatus.COMPLETED else "EXECUTION_FAILED",
            error_message="" if self.plan_status == PlanStatus.COMPLETED else "One or more steps failed",
            start_time=datetime.fromtimestamp(self.execution_start_time).isoformat(),
            end_time=datetime.now().isoformat(),
            actual_duration_s=execution_time,
            feedback_messages=[f"Completed {successful_steps}/{len(primitives)} steps"]
        )
        
        msg = String()
        msg.data = result.to_json()
        self.result_pub.publish(msg)
        
        self.get_logger().info(
            f"Plan execution {self.plan_status.value}: "
            f"{successful_steps}/{len(primitives)} steps in {execution_time:.1f}s"
        )
        
        # Reset state
        self.current_plan = None
        self.current_step = 0
        self.execution_start_time = None
    
    def destroy_node(self):
        """Cleanup"""
        self.is_running = False
        self.stop_event.set()
        if self.execution_thread.is_alive():
            self.execution_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlanExecutorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
