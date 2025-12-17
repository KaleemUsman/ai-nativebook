#!/usr/bin/env python3
"""
Fetch Object Scenario for VLA Capstone

Demonstrates the complete VLA pipeline for a "fetch object" task:
- Voice command: "Go to the desk and pick up the phone"
- Navigation to target location
- Object detection and localization
- Grasp execution
- Status reporting

Usage:
    python3 fetch_object.py
    
    # With custom parameters
    python3 fetch_object.py --location kitchen --object cup
"""

import argparse
import json
import time
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src'))

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS 2 not available - running in standalone simulation mode")

try:
    from vla.llm_client import LLMClient, create_llm_client
except ImportError:
    LLMClient = None

from examples.vla.common.action_primitives import (
    navigate, say, pick_up, scan, find,
    validate_primitive, get_primitive
)
from examples.vla.common.speech_synthesis import say as speak, get_synthesizer
from examples.vla.common.error_handling import (
    ErrorHandler, create_error, ErrorCode, RecoveryAction
)


class FetchObjectScenario:
    """
    Fetch Object Scenario - Navigate to location, find object, and pick it up.
    
    This scenario demonstrates:
    1. Voice command processing
    2. LLM-based plan generation
    3. Sequential primitive execution
    4. Error handling and recovery
    5. User feedback via speech
    """
    
    def __init__(
        self,
        target_location: str = "desk",
        target_object: str = "phone",
        use_llm: bool = True,
        simulate: bool = True
    ):
        """
        Initialize the scenario.
        
        Args:
            target_location: Where to navigate
            target_object: What to pick up
            use_llm: Use LLM for plan generation
            simulate: Run in simulation mode (no actual robot)
        """
        self.target_location = target_location
        self.target_object = target_object
        self.use_llm = use_llm
        self.simulate = simulate
        
        # State
        self.robot_pose = {"x": 0, "y": 0, "z": 0, "yaw": 0}
        self.gripper_state = "OPEN"
        self.held_object = None
        self.detected_objects = []
        
        # LLM client
        self.llm_client = None
        if use_llm and LLMClient:
            try:
                self.llm_client = create_llm_client()
            except Exception as e:
                print(f"LLM client initialization failed: {e}")
        
        # Error handler
        self.error_handler = ErrorHandler(
            max_retries=3,
            user_callback=self._user_notification
        )
        
        # Synthesizer for speech feedback
        try:
            self.synthesizer = get_synthesizer()
        except:
            self.synthesizer = None
    
    def run(self) -> bool:
        """
        Execute the fetch object scenario.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print(f"FETCH OBJECT SCENARIO")
        print(f"Target: {self.target_object} at {self.target_location}")
        print("="*60 + "\n")
        
        # Step 1: Acknowledge command
        self._say(f"Understood. I'll get the {self.target_object} from the {self.target_location}.")
        
        # Step 2: Generate plan
        plan = self._generate_plan()
        if not plan:
            self._say("I'm sorry, I couldn't figure out how to do that.")
            return False
        
        print(f"\nGenerated plan with {len(plan)} steps:")
        for i, step in enumerate(plan):
            print(f"  {i+1}. {step['action_type']}: {step.get('parameters', {})}")
        print()
        
        # Step 3: Execute plan
        success = self._execute_plan(plan)
        
        # Step 4: Report result
        if success:
            self._say(f"I've got the {self.target_object}. What should I do with it?")
        else:
            self._say(f"I'm sorry, I couldn't complete the task.")
        
        return success
    
    def _generate_plan(self) -> list:
        """Generate action plan using LLM or fallback"""
        
        # Try LLM-based planning
        if self.llm_client:
            try:
                context = {
                    "robot_pose": self.robot_pose,
                    "gripper_state": self.gripper_state,
                    "held_object": self.held_object,
                    "detected_objects": self.detected_objects,
                    "known_locations": ["home", "desk", "kitchen", "living_room"]
                }
                
                plan = self.llm_client.generate_plan(
                    command=f"Go to the {self.target_location} and pick up the {self.target_object}",
                    intent_type="MANIPULATION",
                    target_object=self.target_object,
                    target_location=self.target_location,
                    context=context
                )
                
                if plan and plan.primitives:
                    return [
                        {
                            "action_type": p.action_type,
                            "parameters": p.parameters,
                            "preconditions": p.preconditions,
                            "expected_outcomes": p.expected_outcomes
                        }
                        for p in plan.primitives
                    ]
            except Exception as e:
                print(f"LLM planning failed: {e}")
        
        # Fallback: hardcoded plan
        return self._generate_fallback_plan()
    
    def _generate_fallback_plan(self) -> list:
        """Generate a fallback plan without LLM"""
        return [
            say(f"I'm heading to the {self.target_location} to find the {self.target_object}."),
            navigate(location=self.target_location),
            scan(mode="full"),
            find(object_name=self.target_object),
            pick_up(object_id=f"{self.target_object}_001"),
            say(f"I've picked up the {self.target_object}.")
        ]
    
    def _execute_plan(self, plan: list) -> bool:
        """Execute the action plan"""
        for i, step in enumerate(plan):
            action_type = step['action_type']
            parameters = step.get('parameters', {})
            
            print(f"\n[Step {i+1}/{len(plan)}] Executing: {action_type}")
            print(f"  Parameters: {json.dumps(parameters, indent=2)}")
            
            success, error_msg = self._execute_primitive(action_type, parameters)
            
            if not success:
                print(f"  ❌ FAILED: {error_msg}")
                
                # Try error recovery
                error = create_error(
                    self._get_error_code(action_type),
                    {"action_type": action_type, "error": error_msg}
                )
                
                recovery_action, recovery_params = self.error_handler.handle(error)
                
                if recovery_action == RecoveryAction.RETRY:
                    print(f"  Retrying...")
                    success, error_msg = self._execute_primitive(action_type, parameters)
                    if not success:
                        return False
                elif recovery_action == RecoveryAction.ABORT:
                    return False
            else:
                print(f"  ✓ SUCCESS")
        
        return True
    
    def _execute_primitive(self, action_type: str, parameters: dict) -> tuple:
        """Execute a single primitive (simulated)"""
        
        if self.simulate:
            # Simulated execution
            time.sleep(0.5)  # Simulate action time
            
            # Update state based on action
            if action_type == "navigate_to":
                self.robot_pose = {"x": 5, "y": 2, "z": 0, "yaw": 0}
                return True, ""
            
            elif action_type == "scan_environment":
                self.detected_objects = [
                    {"id": f"{self.target_object}_001", "name": self.target_object, "confidence": 0.92}
                ]
                return True, ""
            
            elif action_type == "identify_object":
                # 90% success rate
                import random
                if random.random() < 0.9:
                    return True, ""
                else:
                    return False, "Object not found"
            
            elif action_type == "pick_up":
                self.gripper_state = "HOLDING"
                self.held_object = parameters.get('object_id')
                return True, ""
            
            elif action_type == "say":
                text = parameters.get('text', '')
                self._say(text)
                return True, ""
            
            else:
                return True, ""
        
        # Real execution would go here
        return True, ""
    
    def _say(self, text: str):
        """Speak text to user"""
        print(f"[ROBOT]: {text}")
        if self.synthesizer:
            try:
                self.synthesizer.say(text, blocking=False)
            except:
                pass
    
    def _user_notification(self, message: str, error):
        """Callback for user notifications"""
        self._say(message)
    
    def _get_error_code(self, action_type: str) -> str:
        """Map action type to error code"""
        error_map = {
            "navigate_to": ErrorCode.NAV_GOAL_UNREACHABLE,
            "identify_object": ErrorCode.PERC_OBJECT_NOT_FOUND,
            "pick_up": ErrorCode.MANIP_GRASP_FAILED,
            "scan_environment": ErrorCode.PERC_DETECTION_FAILED,
        }
        return error_map.get(action_type, "UNKNOWN")


def main():
    parser = argparse.ArgumentParser(description='Fetch Object Scenario')
    parser.add_argument('--location', type=str, default='desk',
                      help='Target location')
    parser.add_argument('--object', type=str, default='phone',
                      help='Target object')
    parser.add_argument('--use-llm', action='store_true', default=True,
                      help='Use LLM for planning')
    parser.add_argument('--no-llm', action='store_false', dest='use_llm',
                      help='Skip LLM, use fallback plan')
    
    args = parser.parse_args()
    
    # Create and run scenario
    scenario = FetchObjectScenario(
        target_location=args.location,
        target_object=args.object,
        use_llm=args.use_llm,
        simulate=True
    )
    
    success = scenario.run()
    
    print("\n" + "="*60)
    print(f"SCENARIO RESULT: {'SUCCESS ✓' if success else 'FAILED ✗'}")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
