#!/usr/bin/env python3
"""
Context Manager Node for VLA Pipeline

Aggregates environmental context for LLM planning including:
- Robot state (pose, gripper state)
- Detected objects from perception
- Known locations
- Task history

Subscribes to:
    - /tf (Transform updates)
    - /perception/objects (Object detections)
    - /vla/execution_result (Task history)
    
Publishes to:
    - /vla/task_context (TaskContext)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from threading import Lock


@dataclass
class DetectedObject:
    """Represents a detected object"""
    object_id: str
    name: str
    confidence: float
    position: Dict[str, float]  # x, y, z
    last_seen: float  # Unix timestamp
    color: Optional[str] = None
    graspable: bool = True


@dataclass
class NamedLocation:
    """Represents a named location"""
    name: str
    description: str
    pose: Dict[str, float]  # x, y, z, yaw


@dataclass
class ExecutionResult:
    """Represents a past execution result"""
    action_type: str
    success: bool
    timestamp: float
    summary: str


@dataclass
class TaskContext:
    """Complete task context for LLM planning"""
    robot_pose: Dict[str, float]
    gripper_state: str  # OPEN, CLOSED, HOLDING
    held_object: Optional[str]
    detected_objects: List[Dict]
    known_locations: List[str]
    battery_level: float
    active_capabilities: List[str]
    recent_history: List[Dict]
    timestamp: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class ContextManagerNode(Node):
    """
    ROS 2 node for managing task context.
    
    Aggregates robot state, perception data, and task history into
    a unified context for LLM-based planning.
    """
    
    def __init__(self):
        super().__init__('context_manager')
        
        # Declare parameters
        self.declare_parameter('object_timeout_s', 60.0)
        self.declare_parameter('max_history_items', 10)
        self.declare_parameter('publish_rate_hz', 2.0)
        
        self.object_timeout = self.get_parameter('object_timeout_s').value
        self.max_history = self.get_parameter('max_history_items').value
        publish_rate = self.get_parameter('publish_rate_hz').value
        
        # Context state
        self.lock = Lock()
        
        # Robot state
        self.robot_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.gripper_state = "OPEN"
        self.held_object: Optional[str] = None
        self.battery_level = 1.0
        
        # Object registry
        self.detected_objects: Dict[str, DetectedObject] = {}
        
        # Location registry
        self.known_locations: Dict[str, NamedLocation] = {}
        self._init_default_locations()
        
        # Task history
        self.execution_history: List[ExecutionResult] = []
        
        # Capabilities
        self.active_capabilities = [
            "navigation",
            "perception",
            "manipulation",
            "speech"
        ]
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        # Note: In production, subscribe to actual sensor topics
        self.objects_sub = self.create_subscription(
            String,
            '/perception/objects',
            self._objects_callback,
            qos
        )
        
        self.robot_state_sub = self.create_subscription(
            String,
            '/robot/state',
            self._robot_state_callback,
            qos
        )
        
        self.execution_result_sub = self.create_subscription(
            String,
            '/vla/execution_result',
            self._execution_result_callback,
            qos
        )
        
        # Publishers
        self.context_pub = self.create_publisher(
            String,
            '/vla/task_context',
            qos
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/vla/planner_status',
            qos
        )
        
        # Timer for periodic context publishing
        self.publish_timer = self.create_timer(
            1.0 / publish_rate,
            self._publish_context
        )
        
        # Timer for cleanup of stale objects
        self.cleanup_timer = self.create_timer(5.0, self._cleanup_stale_objects)
        
        self._publish_status("Context manager ready")
    
    def _init_default_locations(self):
        """Initialize default known locations"""
        defaults = [
            NamedLocation("home", "Starting position", {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}),
            NamedLocation("kitchen", "Kitchen area", {"x": 5.0, "y": 2.0, "z": 0.0, "yaw": 1.57}),
            NamedLocation("living_room", "Living room", {"x": 3.0, "y": -2.0, "z": 0.0, "yaw": 0.0}),
            NamedLocation("office", "Office area", {"x": -3.0, "y": 4.0, "z": 0.0, "yaw": 3.14}),
            NamedLocation("table", "Main table", {"x": 2.5, "y": 1.0, "z": 0.0, "yaw": 0.0}),
        ]
        
        for loc in defaults:
            self.known_locations[loc.name] = loc
    
    def _publish_status(self, message: str):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def _objects_callback(self, msg: String):
        """Handle incoming object detections"""
        try:
            data = json.loads(msg.data)
            objects = data.get("objects", [])
            
            with self.lock:
                current_time = time.time()
                for obj in objects:
                    obj_id = obj.get("id", str(hash(obj.get("name", ""))))
                    
                    detected = DetectedObject(
                        object_id=obj_id,
                        name=obj.get("name", "unknown"),
                        confidence=obj.get("confidence", 0.5),
                        position=obj.get("position", {"x": 0, "y": 0, "z": 0}),
                        last_seen=current_time,
                        color=obj.get("color"),
                        graspable=obj.get("graspable", True)
                    )
                    
                    self.detected_objects[obj_id] = detected
                    
            self.get_logger().debug(f"Updated {len(objects)} objects")
            
        except Exception as e:
            self.get_logger().error(f"Error processing objects: {e}")
    
    def _robot_state_callback(self, msg: String):
        """Handle robot state updates"""
        try:
            data = json.loads(msg.data)
            
            with self.lock:
                if "pose" in data:
                    self.robot_pose = data["pose"]
                
                if "gripper_state" in data:
                    self.gripper_state = data["gripper_state"]
                
                if "held_object" in data:
                    self.held_object = data["held_object"]
                
                if "battery_level" in data:
                    self.battery_level = data["battery_level"]
                    
        except Exception as e:
            self.get_logger().error(f"Error processing robot state: {e}")
    
    def _execution_result_callback(self, msg: String):
        """Handle execution result for history tracking"""
        try:
            data = json.loads(msg.data)
            
            result = ExecutionResult(
                action_type=data.get("action_type", "unknown"),
                success=data.get("success", False),
                timestamp=time.time(),
                summary=data.get("summary", "")
            )
            
            with self.lock:
                self.execution_history.append(result)
                # Trim history
                if len(self.execution_history) > self.max_history:
                    self.execution_history = self.execution_history[-self.max_history:]
                    
        except Exception as e:
            self.get_logger().error(f"Error processing execution result: {e}")
    
    def _cleanup_stale_objects(self):
        """Remove objects that haven't been seen recently"""
        current_time = time.time()
        
        with self.lock:
            stale_ids = [
                obj_id for obj_id, obj in self.detected_objects.items()
                if current_time - obj.last_seen > self.object_timeout
            ]
            
            for obj_id in stale_ids:
                del self.detected_objects[obj_id]
            
            if stale_ids:
                self.get_logger().debug(f"Removed {len(stale_ids)} stale objects")
    
    def _publish_context(self):
        """Publish current task context"""
        with self.lock:
            # Build context
            context = TaskContext(
                robot_pose=self.robot_pose.copy(),
                gripper_state=self.gripper_state,
                held_object=self.held_object,
                detected_objects=[
                    {
                        "id": obj.object_id,
                        "name": obj.name,
                        "confidence": obj.confidence,
                        "position": obj.position,
                        "color": obj.color,
                        "graspable": obj.graspable
                    }
                    for obj in self.detected_objects.values()
                ],
                known_locations=list(self.known_locations.keys()),
                battery_level=self.battery_level,
                active_capabilities=self.active_capabilities.copy(),
                recent_history=[
                    {
                        "action": r.action_type,
                        "success": r.success,
                        "summary": r.summary
                    }
                    for r in self.execution_history[-5:]
                ],
                timestamp=datetime.now().isoformat()
            )
        
        # Publish
        msg = String()
        msg.data = context.to_json()
        self.context_pub.publish(msg)
    
    def get_context(self) -> TaskContext:
        """Get current context (for direct use)"""
        with self.lock:
            return TaskContext(
                robot_pose=self.robot_pose.copy(),
                gripper_state=self.gripper_state,
                held_object=self.held_object,
                detected_objects=[asdict(obj) for obj in self.detected_objects.values()],
                known_locations=list(self.known_locations.keys()),
                battery_level=self.battery_level,
                active_capabilities=self.active_capabilities.copy(),
                recent_history=[asdict(r) for r in self.execution_history[-5:]],
                timestamp=datetime.now().isoformat()
            )
    
    def add_location(self, name: str, description: str, pose: Dict[str, float]):
        """Add a new known location"""
        with self.lock:
            self.known_locations[name] = NamedLocation(name, description, pose)
            self.get_logger().info(f"Added location: {name}")
    
    def update_robot_state(
        self,
        pose: Optional[Dict] = None,
        gripper_state: Optional[str] = None,
        held_object: Optional[str] = None
    ):
        """Update robot state directly"""
        with self.lock:
            if pose:
                self.robot_pose = pose
            if gripper_state:
                self.gripper_state = gripper_state
            if held_object is not None:
                self.held_object = held_object


def main(args=None):
    rclpy.init(args=args)
    node = ContextManagerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
