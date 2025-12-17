#!/usr/bin/env python3
"""
LLM Planner Node for VLA Pipeline

Receives parsed intents and context, generates action plans using LLM,
and publishes validated plans for execution.

Subscribes to:
    - /vla/parsed_intent (ParsedIntent)
    - /vla/task_context (TaskContext)
    
Publishes to:
    - /vla/action_plan (ActionPlan)
    - /vla/plan_explanation (String)
    - /vla/planner_status (String)

Action Servers:
    - /vla/plan_task (PlanTask)
    - /vla/replan (Replan)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

import json
import time
import uuid
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Lock

# Import LLM client
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

try:
    from vla.llm_client import LLMClient, ActionPlan, create_llm_client
except ImportError:
    # Fallback for when package is not installed
    LLMClient = None
    ActionPlan = None


@dataclass
class PlanTaskGoal:
    """Goal for plan_task action"""
    intent_id: str
    intent_type: str
    command: str
    target_object: Optional[str]
    target_location: Optional[str]
    modifiers: List[str]
    context: Dict[str, Any]


@dataclass
class PlanTaskResult:
    """Result for plan_task action"""
    success: bool
    plan: Optional[Dict]
    error_message: str
    planning_time_s: float


class LLMPlannerNode(Node):
    """
    ROS 2 node for LLM-based action planning.
    
    Integrates with OpenAI GPT-4 (or Ollama fallback) to generate
    action plans from natural language intents.
    """
    
    def __init__(self):
        super().__init__('llm_planner')
        
        # Declare parameters
        self.declare_parameter('llm.provider', 'openai')
        self.declare_parameter('llm.model', 'gpt-4')
        self.declare_parameter('llm.temperature', 0.2)
        self.declare_parameter('llm.max_tokens', 1024)
        self.declare_parameter('llm.timeout_s', 15.0)
        self.declare_parameter('llm.max_retries', 3)
        self.declare_parameter('llm.fallback_enabled', True)
        self.declare_parameter('llm.fallback_provider', 'ollama')
        self.declare_parameter('llm.fallback_model', 'llama3')
        self.declare_parameter('planning.validate_plans', True)
        self.declare_parameter('planning.max_plan_length', 15)
        
        # Get parameters
        provider = self.get_parameter('llm.provider').value
        model = self.get_parameter('llm.model').value
        temperature = self.get_parameter('llm.temperature').value
        max_tokens = self.get_parameter('llm.max_tokens').value
        timeout_s = self.get_parameter('llm.timeout_s').value
        max_retries = self.get_parameter('llm.max_retries').value
        fallback_enabled = self.get_parameter('llm.fallback_enabled').value
        fallback_provider = self.get_parameter('llm.fallback_provider').value
        fallback_model = self.get_parameter('llm.fallback_model').value
        self.validate_plans = self.get_parameter('planning.validate_plans').value
        self.max_plan_length = self.get_parameter('planning.max_plan_length').value
        
        # Initialize LLM client
        self.llm_client: Optional[LLMClient] = None
        if LLMClient:
            self.llm_client = LLMClient(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                max_retries=max_retries,
                fallback_provider=fallback_provider if fallback_enabled else None,
                fallback_model=fallback_model if fallback_enabled else None
            )
            self.get_logger().info(
                f"LLM Planner initialized: provider={provider}, model={model}"
            )
        else:
            self.get_logger().warn("LLM client not available, using simulation mode")
        
        # State
        self.lock = Lock()
        self.current_context: Optional[Dict] = None
        self.last_plan: Optional[Dict] = None
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.intent_sub = self.create_subscription(
            String,
            '/vla/parsed_intent',
            self._intent_callback,
            qos
        )
        
        self.context_sub = self.create_subscription(
            String,
            '/vla/task_context',
            self._context_callback,
            qos
        )
        
        # Publishers
        self.plan_pub = self.create_publisher(String, '/vla/action_plan', qos)
        self.explanation_pub = self.create_publisher(String, '/vla/plan_explanation', qos)
        self.status_pub = self.create_publisher(String, '/vla/planner_status', qos)
        
        self._publish_status("LLM Planner ready")
    
    def _publish_status(self, message: str):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def _context_callback(self, msg: String):
        """Handle incoming context updates"""
        try:
            with self.lock:
                self.current_context = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Error parsing context: {e}")
    
    def _intent_callback(self, msg: String):
        """Handle incoming parsed intent and generate plan"""
        try:
            intent = json.loads(msg.data)
            
            self.get_logger().info(
                f"Received intent: type={intent.get('intent_type')}, "
                f"command='{intent.get('raw_transcription', '')[:50]}'"
            )
            
            # Generate plan
            plan = self._generate_plan(intent)
            
            if plan:
                # Publish plan
                plan_msg = String()
                if hasattr(plan, 'to_json'):
                    plan_msg.data = plan.to_json()
                else:
                    plan_msg.data = json.dumps(plan)
                self.plan_pub.publish(plan_msg)
                
                # Publish explanation
                explanation_msg = String()
                explanation_msg.data = plan.get('explanation', '') if isinstance(plan, dict) else getattr(plan, 'explanation', '')
                self.explanation_pub.publish(explanation_msg)
                
                self.get_logger().info(f"Published action plan with {len(plan.get('primitives', []) if isinstance(plan, dict) else plan.primitives)} steps")
            else:
                self._publish_status("Failed to generate action plan")
                
        except Exception as e:
            self.get_logger().error(f"Error processing intent: {e}")
            import traceback
            self.get_logger().debug(traceback.format_exc())
    
    def _generate_plan(self, intent: Dict) -> Optional[Any]:
        """Generate an action plan from parsed intent"""
        start_time = time.time()
        
        # Get current context
        with self.lock:
            context = self.current_context.copy() if self.current_context else {}
        
        # Build context for LLM
        llm_context = {
            "robot_pose": context.get("robot_pose", {"x": 0, "y": 0, "z": 0, "yaw": 0}),
            "gripper_state": context.get("gripper_state", "OPEN"),
            "held_object": context.get("held_object"),
            "detected_objects": context.get("detected_objects", []),
            "known_locations": context.get("known_locations", ["home"]),
            "recent_actions": [
                h.get("action", "") for h in context.get("recent_history", [])
            ]
        }
        
        # Use LLM client if available
        if self.llm_client:
            plan = self.llm_client.generate_plan(
                command=intent.get("raw_transcription", ""),
                intent_type=intent.get("intent_type", "UNKNOWN"),
                target_object=intent.get("target_object"),
                target_location=intent.get("target_location"),
                modifiers=intent.get("modifiers", []),
                context=llm_context
            )
            
            if plan and self.validate_plans:
                is_valid, errors = self.llm_client.validate_plan(plan)
                if not is_valid:
                    self.get_logger().warn(f"Plan validation failed: {errors}")
                    # Could try to fix or regenerate
            
            if plan:
                planning_time = time.time() - start_time
                self.get_logger().info(f"Plan generated in {planning_time:.2f}s")
                return plan.to_dict() if hasattr(plan, 'to_dict') else plan
        
        # Fallback: simulate plan generation
        return self._simulate_plan_generation(intent, llm_context)
    
    def _simulate_plan_generation(
        self,
        intent: Dict,
        context: Dict
    ) -> Dict:
        """Simulate plan generation for testing without LLM"""
        intent_type = intent.get("intent_type", "UNKNOWN")
        target_object = intent.get("target_object")
        target_location = intent.get("target_location")
        
        primitives = []
        
        # Start with acknowledgment
        primitives.append({
            "action_type": "say",
            "parameters": {"text": f"Understood. I'll {intent.get('action_verb', 'help')} now.", "priority": "normal"},
            "preconditions": [],
            "expected_outcomes": ["message_delivered"]
        })
        
        if intent_type == "NAVIGATION" and target_location:
            primitives.append({
                "action_type": "navigate_to",
                "parameters": {"location": target_location},
                "preconditions": ["robot_mobile", "navigation_enabled"],
                "expected_outcomes": ["robot_at_location"]
            })
            primitives.append({
                "action_type": "say",
                "parameters": {"text": f"I've arrived at the {target_location}.", "priority": "normal"},
                "preconditions": [],
                "expected_outcomes": ["message_delivered"]
            })
        
        elif intent_type == "MANIPULATION" and target_object:
            # Navigate first if location specified
            if target_location:
                primitives.append({
                    "action_type": "navigate_to",
                    "parameters": {"location": target_location},
                    "preconditions": ["robot_mobile"],
                    "expected_outcomes": ["robot_at_location"]
                })
            
            # Identify object
            primitives.append({
                "action_type": "identify_object",
                "parameters": {"object_name": target_object},
                "preconditions": ["perception_enabled"],
                "expected_outcomes": ["object_found", "object_localized"]
            })
            
            # Pick up
            primitives.append({
                "action_type": "pick_up",
                "parameters": {"object_id": f"{target_object}_001", "grasp_type": "power"},
                "preconditions": ["gripper_empty", "object_reachable"],
                "expected_outcomes": ["object_held"]
            })
            
            primitives.append({
                "action_type": "say",
                "parameters": {"text": f"I've picked up the {target_object}.", "priority": "normal"},
                "preconditions": [],
                "expected_outcomes": ["message_delivered"]
            })
        
        elif intent_type == "QUERY":
            primitives.append({
                "action_type": "scan_environment",
                "parameters": {"mode": "full"},
                "preconditions": ["perception_enabled"],
                "expected_outcomes": ["objects_detected"]
            })
            primitives.append({
                "action_type": "say",
                "parameters": {"text": f"I can see several objects in the area.", "priority": "normal"},
                "preconditions": [],
                "expected_outcomes": ["message_delivered"]
            })
        
        elif intent_type == "CANCEL":
            primitives.append({
                "action_type": "cancel",
                "parameters": {},
                "preconditions": [],
                "expected_outcomes": ["action_cancelled"]
            })
        
        else:
            # Generic handling
            primitives.append({
                "action_type": "say",
                "parameters": {"text": "I'm not sure how to help with that. Could you rephrase?", "priority": "normal"},
                "preconditions": [],
                "expected_outcomes": ["message_delivered"]
            })
        
        return {
            "plan_id": str(uuid.uuid4()),
            "explanation": f"Simulated plan for {intent_type} command",
            "primitives": primitives,
            "estimated_duration_s": len(primitives) * 10.0
        }


def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
