#!/usr/bin/env python3
"""
Action Primitives Library for VLA Pipeline

Defines the action primitive types, schemas, and validation utilities
used throughout the VLA system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import json


class ActionCategory(Enum):
    """Categories of action primitives"""
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MANIPULATION = "manipulation"
    COMMUNICATION = "communication"
    CONTROL = "control"


class GraspType(Enum):
    """Types of grasps for manipulation"""
    POWER = "power"
    PRECISION = "precision"
    PINCH = "pinch"


class PlaceStyle(Enum):
    """Styles for placing objects"""
    DROP = "drop"
    GENTLE = "gentle"
    PRECISE = "precise"


class ScanMode(Enum):
    """Modes for environment scanning"""
    QUICK = "quick"
    FULL = "full"
    DETAILED = "detailed"


class SpeechPriority(Enum):
    """Priority levels for speech output"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class PrimitiveParameter:
    """Definition of a primitive parameter"""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class PrimitiveDefinition:
    """Complete definition of an action primitive"""
    name: str
    description: str
    category: ActionCategory
    parameters: List[PrimitiveParameter]
    preconditions: List[str]
    expected_outcomes: List[str]
    default_timeout_s: float = 30.0
    interruptible: bool = True


# Primitive Definitions
PRIMITIVES: Dict[str, PrimitiveDefinition] = {
    'navigate_to': PrimitiveDefinition(
        name='navigate_to',
        description='Navigate the robot to a specified location',
        category=ActionCategory.NAVIGATION,
        parameters=[
            PrimitiveParameter('location', 'string', 'Named location', False),
            PrimitiveParameter('pose', 'object', 'Target pose {x, y, z, yaw}', False),
            PrimitiveParameter('approach_distance', 'float', 'Stop distance from target', False, 0.0),
        ],
        preconditions=['robot_mobile', 'navigation_enabled'],
        expected_outcomes=['robot_at_location'],
        default_timeout_s=60.0
    ),
    
    'look_at': PrimitiveDefinition(
        name='look_at',
        description='Orient sensors toward a target',
        category=ActionCategory.PERCEPTION,
        parameters=[
            PrimitiveParameter('target', 'string', 'Object name to look at', False),
            PrimitiveParameter('target_id', 'string', 'Specific object ID', False),
            PrimitiveParameter('direction', 'object', 'Direction vector {x, y, z}', False),
            PrimitiveParameter('duration_s', 'float', 'How long to maintain gaze', False, 2.0),
        ],
        preconditions=['head_controllable'],
        expected_outcomes=['target_in_view'],
        default_timeout_s=10.0
    ),
    
    'scan_environment': PrimitiveDefinition(
        name='scan_environment',
        description='Perform an environmental scan to detect objects',
        category=ActionCategory.PERCEPTION,
        parameters=[
            PrimitiveParameter('area', 'string', 'Named area to scan', False),
            PrimitiveParameter('mode', 'string', 'Scan mode', False, 'full', ['quick', 'full', 'detailed']),
            PrimitiveParameter('object_classes', 'array', 'Specific classes to find', False),
        ],
        preconditions=['perception_enabled'],
        expected_outcomes=['objects_detected'],
        default_timeout_s=30.0
    ),
    
    'identify_object': PrimitiveDefinition(
        name='identify_object',
        description='Detect and localize a specific object',
        category=ActionCategory.PERCEPTION,
        parameters=[
            PrimitiveParameter('object_name', 'string', 'Object class name', True),
            PrimitiveParameter('color', 'string', 'Optional color filter', False),
            PrimitiveParameter('location_hint', 'string', 'Where to look first', False),
        ],
        preconditions=['perception_enabled'],
        expected_outcomes=['object_found', 'object_localized'],
        default_timeout_s=15.0
    ),
    
    'pick_up': PrimitiveDefinition(
        name='pick_up',
        description='Grasp and lift an object',
        category=ActionCategory.MANIPULATION,
        parameters=[
            PrimitiveParameter('object_id', 'string', 'ID of object to pick up', True),
            PrimitiveParameter('grasp_type', 'string', 'Type of grasp', False, 'power', ['power', 'precision', 'pinch']),
            PrimitiveParameter('approach_direction', 'string', 'Direction to approach', False, 'top', ['top', 'side', 'front']),
        ],
        preconditions=['gripper_empty', 'object_reachable', 'manipulation_enabled'],
        expected_outcomes=['object_held'],
        default_timeout_s=45.0,
        interruptible=False
    ),
    
    'place': PrimitiveDefinition(
        name='place',
        description='Place a held object at a location',
        category=ActionCategory.MANIPULATION,
        parameters=[
            PrimitiveParameter('location', 'string', 'Named location', False),
            PrimitiveParameter('pose', 'object', 'Placement pose {x, y, z, yaw}', False),
            PrimitiveParameter('place_style', 'string', 'How to place', False, 'gentle', ['drop', 'gentle', 'precise']),
        ],
        preconditions=['gripper_holding', 'location_reachable', 'manipulation_enabled'],
        expected_outcomes=['object_placed', 'gripper_empty'],
        default_timeout_s=30.0,
        interruptible=False
    ),
    
    'say': PrimitiveDefinition(
        name='say',
        description='Speak text to the user via TTS',
        category=ActionCategory.COMMUNICATION,
        parameters=[
            PrimitiveParameter('text', 'string', 'Text to speak', True),
            PrimitiveParameter('priority', 'string', 'Speech priority', False, 'normal', ['low', 'normal', 'high']),
            PrimitiveParameter('wait_for_completion', 'boolean', 'Wait for speech to finish', False, True),
        ],
        preconditions=['tts_enabled'],
        expected_outcomes=['message_delivered'],
        default_timeout_s=30.0
    ),
    
    'wait': PrimitiveDefinition(
        name='wait',
        description='Pause execution',
        category=ActionCategory.CONTROL,
        parameters=[
            PrimitiveParameter('duration', 'float', 'Seconds to wait', False),
            PrimitiveParameter('condition', 'string', 'Condition to wait for', False, enum=['object_stable', 'user_ready', 'path_clear']),
        ],
        preconditions=[],
        expected_outcomes=['wait_complete'],
        default_timeout_s=60.0
    ),
    
    'cancel': PrimitiveDefinition(
        name='cancel',
        description='Cancel the current action or plan',
        category=ActionCategory.CONTROL,
        parameters=[],
        preconditions=[],
        expected_outcomes=['action_cancelled'],
        default_timeout_s=5.0,
        interruptible=False
    ),
}


def get_primitive(name: str) -> Optional[PrimitiveDefinition]:
    """Get primitive definition by name"""
    return PRIMITIVES.get(name)


def get_all_primitives() -> List[PrimitiveDefinition]:
    """Get all primitive definitions"""
    return list(PRIMITIVES.values())


def get_primitives_by_category(category: ActionCategory) -> List[PrimitiveDefinition]:
    """Get primitives by category"""
    return [p for p in PRIMITIVES.values() if p.category == category]


def validate_primitive(action_type: str, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate primitive parameters.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    primitive = get_primitive(action_type)
    if not primitive:
        return False, [f"Unknown primitive: {action_type}"]
    
    # Check required parameters
    for param in primitive.parameters:
        if param.required and param.name not in parameters:
            errors.append(f"Missing required parameter: {param.name}")
    
    # Check parameter types and enum values
    for param in primitive.parameters:
        if param.name in parameters:
            value = parameters[param.name]
            
            # Check enum values
            if param.enum and value not in param.enum:
                errors.append(
                    f"Invalid value for {param.name}: '{value}'. "
                    f"Must be one of: {param.enum}"
                )
    
    return len(errors) == 0, errors


def create_primitive(
    action_type: str,
    parameters: Dict[str, Any],
    preconditions: Optional[List[str]] = None,
    expected_outcomes: Optional[List[str]] = None
) -> Optional[Dict]:
    """
    Create a primitive dictionary with defaults filled in.
    
    Returns:
        Primitive dictionary or None if invalid
    """
    primitive = get_primitive(action_type)
    if not primitive:
        return None
    
    # Validate
    is_valid, errors = validate_primitive(action_type, parameters)
    if not is_valid:
        return None
    
    # Fill defaults
    filled_params = {}
    for param in primitive.parameters:
        if param.name in parameters:
            filled_params[param.name] = parameters[param.name]
        elif param.default is not None:
            filled_params[param.name] = param.default
    
    return {
        'action_type': action_type,
        'parameters': filled_params,
        'preconditions': preconditions or list(primitive.preconditions),
        'expected_outcomes': expected_outcomes or list(primitive.expected_outcomes),
        'timeout_s': primitive.default_timeout_s
    }


def get_function_schema() -> Dict:
    """Get the LLM function calling schema for all primitives"""
    return {
        "type": "function",
        "function": {
            "name": "create_action_plan",
            "description": "Create a sequence of robot actions",
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
                                "action_type": {
                                    "type": "string",
                                    "enum": list(PRIMITIVES.keys())
                                },
                                "parameters": {"type": "object"},
                                "preconditions": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "expected_outcomes": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["action_type", "parameters"]
                        }
                    }
                },
                "required": ["explanation", "primitives"]
            }
        }
    }


# Convenience functions for creating common primitives
def navigate(location: str = None, pose: Dict = None) -> Dict:
    """Create a navigate_to primitive"""
    params = {}
    if location:
        params['location'] = location
    if pose:
        params['pose'] = pose
    return create_primitive('navigate_to', params)


def say(text: str, priority: str = 'normal') -> Dict:
    """Create a say primitive"""
    return create_primitive('say', {'text': text, 'priority': priority})


def pick_up(object_id: str, grasp_type: str = 'power') -> Dict:
    """Create a pick_up primitive"""
    return create_primitive('pick_up', {
        'object_id': object_id,
        'grasp_type': grasp_type
    })


def place(location: str = None, pose: Dict = None, style: str = 'gentle') -> Dict:
    """Create a place primitive"""
    params = {'place_style': style}
    if location:
        params['location'] = location
    if pose:
        params['pose'] = pose
    return create_primitive('place', params)


def scan(mode: str = 'full', area: str = None) -> Dict:
    """Create a scan_environment primitive"""
    params = {'mode': mode}
    if area:
        params['area'] = area
    return create_primitive('scan_environment', params)


def find(object_name: str, color: str = None) -> Dict:
    """Create an identify_object primitive"""
    params = {'object_name': object_name}
    if color:
        params['color'] = color
    return create_primitive('identify_object', params)
