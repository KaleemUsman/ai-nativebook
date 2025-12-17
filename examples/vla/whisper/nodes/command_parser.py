#!/usr/bin/env python3
"""
Command Parser Node for VLA Pipeline

This node receives transcriptions from the Whisper node and parses them into
structured intents that can be processed by the LLM planner.

Subscribes to:
    - /vla/transcription (String with JSON transcription)
    
Publishes to:
    - /vla/parsed_intent (String with JSON parsed intent)
    - /vla/whisper_status (std_msgs/String)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

import json
import re
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from enum import Enum


class IntentType(Enum):
    """Types of user intents"""
    NAVIGATION = "NAVIGATION"
    MANIPULATION = "MANIPULATION"
    QUERY = "QUERY"
    CANCEL = "CANCEL"
    UNKNOWN = "UNKNOWN"


class Urgency(Enum):
    """Urgency levels for commands"""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


@dataclass
class ParsedIntent:
    """Represents a parsed user intent"""
    intent_id: str
    source_command_id: str
    intent_type: str
    action_verb: str
    target_object: Optional[str]
    target_location: Optional[str]
    modifiers: List[str]
    urgency: str
    raw_transcription: str
    parse_confidence: float
    timestamp: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class IntentParser:
    """
    Parses natural language transcriptions into structured intents.
    
    Uses pattern matching and keyword extraction for intent classification
    and entity extraction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Intent patterns
        self.intent_patterns = self.config.get('intent_patterns', {
            'navigation': [
                r'\b(go|navigate|move|walk|head|travel|come|return)\b.*\b(to|towards|into)\b',
                r'\b(go|get)\b.*\b(there|here)\b',
            ],
            'manipulation': [
                r'\b(pick up|grab|take|get|fetch|bring|carry)\b',
                r'\b(put|place|drop|set|move)\b.*\b(on|in|at|down)\b',
                r'\b(open|close|turn on|turn off|switch)\b',
            ],
            'query': [
                r'\b(where is|where are|find|locate|search|look for)\b',
                r'\b(what is|what are|tell me|describe|show me)\b',
                r'\b(can you see|do you see|is there)\b',
            ],
            'cancel': [
                r'\b(stop|cancel|abort|halt|quit|nevermind|never mind)\b',
            ],
        })
        
        # Compile patterns
        self.compiled_patterns = {}
        for intent_type, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent_type] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]
        
        # Action verb extraction patterns
        self.action_verbs = {
            'navigation': ['go', 'navigate', 'move', 'walk', 'head', 'travel', 'come', 'return'],
            'manipulation': ['pick', 'grab', 'take', 'get', 'fetch', 'bring', 'carry',
                           'put', 'place', 'drop', 'set', 'move', 'open', 'close'],
            'query': ['find', 'locate', 'search', 'look', 'tell', 'describe', 'show', 'see'],
            'cancel': ['stop', 'cancel', 'abort', 'halt', 'quit'],
        }
        
        # Location aliases
        self.location_aliases = self.config.get('location_aliases', {
            'kitchen': ['kitchen', 'cooking area', 'food area'],
            'living_room': ['living room', 'lounge', 'sitting room', 'living'],
            'bedroom': ['bedroom', 'sleeping room', 'bed'],
            'bathroom': ['bathroom', 'restroom', 'toilet', 'washroom'],
            'office': ['office', 'study', 'work room', 'desk area'],
            'garage': ['garage', 'car park'],
            'entrance': ['entrance', 'front door', 'doorway', 'entry', 'door'],
            'table': ['table', 'dining table', 'desk'],
            'shelf': ['shelf', 'bookshelf', 'shelves'],
            'couch': ['couch', 'sofa', 'loveseat'],
            'window': ['window', 'windows'],
            'counter': ['counter', 'countertop', 'kitchen counter'],
        })
        
        # Object aliases
        self.object_aliases = self.config.get('object_aliases', {
            'cup': ['cup', 'mug', 'glass', 'cups'],
            'phone': ['phone', 'mobile', 'cellphone', 'smartphone', 'iphone', 'android'],
            'book': ['book', 'novel', 'textbook', 'magazine'],
            'keys': ['keys', 'key', 'keychain', 'car keys'],
            'bottle': ['bottle', 'water bottle', 'drink'],
            'remote': ['remote', 'remote control', 'tv remote', 'controller'],
            'laptop': ['laptop', 'computer', 'macbook', 'notebook'],
            'bag': ['bag', 'backpack', 'purse', 'handbag'],
            'ball': ['ball', 'basketball', 'football', 'soccer ball'],
            'pen': ['pen', 'pencil', 'marker'],
            'paper': ['paper', 'document', 'papers', 'documents'],
        })
        
        # Urgency indicators
        self.urgency_high = ['immediately', 'now', 'quickly', 'fast', 'hurry', 'urgent', 'asap']
        self.urgency_low = ['slowly', 'carefully', 'gently', 'when you can', 'eventually']
        
        # Modifier patterns
        self.color_patterns = re.compile(
            r'\b(red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey)\b',
            re.IGNORECASE
        )
        self.size_patterns = re.compile(
            r'\b(big|large|small|tiny|huge|little|medium)\b',
            re.IGNORECASE
        )
        self.position_patterns = re.compile(
            r'\b(left|right|front|back|top|bottom|near|far|closest|nearest)\b',
            re.IGNORECASE
        )
    
    def parse(self, transcription: str, command_id: str) -> ParsedIntent:
        """
        Parse a transcription into a structured intent.
        
        Args:
            transcription: The raw transcription text
            command_id: ID of the source voice command
            
        Returns:
            ParsedIntent object
        """
        text = transcription.lower().strip()
        
        # Determine intent type
        intent_type = self._classify_intent(text)
        
        # Extract action verb
        action_verb = self._extract_action_verb(text, intent_type)
        
        # Extract target object and location
        target_object = self._extract_object(text)
        target_location = self._extract_location(text)
        
        # Extract modifiers
        modifiers = self._extract_modifiers(text)
        
        # Determine urgency
        urgency = self._determine_urgency(text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            intent_type, action_verb, target_object, target_location
        )
        
        return ParsedIntent(
            intent_id=str(uuid.uuid4()),
            source_command_id=command_id,
            intent_type=intent_type.value,
            action_verb=action_verb or "",
            target_object=target_object,
            target_location=target_location,
            modifiers=modifiers,
            urgency=urgency.value,
            raw_transcription=transcription,
            parse_confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
    
    def _classify_intent(self, text: str) -> IntentType:
        """Classify the intent type based on pattern matching"""
        scores = {intent_type: 0 for intent_type in IntentType}
        
        # Check each intent type's patterns
        for intent_name, patterns in self.compiled_patterns.items():
            intent_enum = IntentType[intent_name.upper()]
            for pattern in patterns:
                if pattern.search(text):
                    scores[intent_enum] += 1
        
        # Find the best match
        best_intent = max(scores.items(), key=lambda x: x[1])
        
        if best_intent[1] > 0:
            return best_intent[0]
        
        return IntentType.UNKNOWN
    
    def _extract_action_verb(self, text: str, intent_type: IntentType) -> Optional[str]:
        """Extract the primary action verb"""
        if intent_type == IntentType.UNKNOWN:
            return None
        
        intent_name = intent_type.name.lower()
        if intent_name not in self.action_verbs:
            return None
        
        for verb in self.action_verbs[intent_name]:
            # Match verb as whole word
            pattern = rf'\b{verb}\b'
            if re.search(pattern, text, re.IGNORECASE):
                return verb
        
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract the target location from text"""
        # Look for location indicators
        location_patterns = [
            r'(?:go|navigate|move|walk|head|travel|come|return)\s+(?:to|towards|into)\s+(?:the\s+)?(.+?)(?:\s+and|\s+then|$)',
            r'(?:in|at|on|near|by)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)',
            r'(?:from|to)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location_text = match.group(1).strip()
                # Check against known location aliases
                for canonical, aliases in self.location_aliases.items():
                    for alias in aliases:
                        if alias in location_text:
                            return canonical
                # Return the raw location if no alias match
                if location_text and len(location_text) < 30:
                    return location_text
        
        return None
    
    def _extract_object(self, text: str) -> Optional[str]:
        """Extract the target object from text"""
        # Look for object patterns
        object_patterns = [
            r'(?:pick up|grab|take|get|fetch|bring|find|locate|look for)\s+(?:the\s+)?(?:a\s+)?(.+?)(?:\s+from|\s+on|\s+in|\s+and|$)',
            r'(?:the\s+)?([a-z]+)\s+(?:on|in|at|by)\s+(?:the\s+)?',
            r'my\s+([a-z]+)',
        ]
        
        for pattern in object_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                object_text = match.group(1).strip()
                # Check against known object aliases
                for canonical, aliases in self.object_aliases.items():
                    for alias in aliases:
                        if alias in object_text:
                            return canonical
                # Return the raw object if no alias match
                if object_text and len(object_text) < 20:
                    return object_text
        
        return None
    
    def _extract_modifiers(self, text: str) -> List[str]:
        """Extract modifiers (colors, sizes, positions)"""
        modifiers = []
        
        # Colors
        colors = self.color_patterns.findall(text)
        modifiers.extend([c.lower() for c in colors])
        
        # Sizes
        sizes = self.size_patterns.findall(text)
        modifiers.extend([s.lower() for s in sizes])
        
        # Positions
        positions = self.position_patterns.findall(text)
        modifiers.extend([p.lower() for p in positions])
        
        return list(set(modifiers))  # Remove duplicates
    
    def _determine_urgency(self, text: str) -> Urgency:
        """Determine command urgency"""
        text_lower = text.lower()
        
        for indicator in self.urgency_high:
            if indicator in text_lower:
                return Urgency.HIGH
        
        for indicator in self.urgency_low:
            if indicator in text_lower:
                return Urgency.LOW
        
        return Urgency.NORMAL
    
    def _calculate_confidence(
        self,
        intent_type: IntentType,
        action_verb: Optional[str],
        target_object: Optional[str],
        target_location: Optional[str]
    ) -> float:
        """Calculate parsing confidence score"""
        confidence = 0.0
        
        # Intent classification confidence
        if intent_type != IntentType.UNKNOWN:
            confidence += 0.4
        
        # Action verb confidence
        if action_verb:
            confidence += 0.2
        
        # Target extraction confidence
        if intent_type in [IntentType.NAVIGATION]:
            if target_location:
                confidence += 0.3
            else:
                confidence += 0.1  # Navigation without location is partially valid
        elif intent_type in [IntentType.MANIPULATION]:
            if target_object:
                confidence += 0.3
            if target_location:
                confidence += 0.1
        elif intent_type == IntentType.QUERY:
            if target_object or target_location:
                confidence += 0.3
            else:
                confidence += 0.2  # General queries are valid
        elif intent_type == IntentType.CANCEL:
            confidence += 0.3  # Cancel commands don't need targets
        
        return min(confidence, 1.0)


class CommandParserNode(Node):
    """
    ROS 2 node for parsing voice command transcriptions into structured intents.
    """
    
    def __init__(self):
        super().__init__('command_parser')
        
        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('unknown_intent_handling', 'ask_clarification')
        
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.unknown_handling = self.get_parameter('unknown_intent_handling').value
        
        # Initialize parser
        self.parser = IntentParser()
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.transcription_sub = self.create_subscription(
            String,
            '/vla/transcription',
            self._transcription_callback,
            qos
        )
        
        # Publishers
        self.intent_pub = self.create_publisher(
            String,
            '/vla/parsed_intent',
            qos
        )
        self.status_pub = self.create_publisher(
            String,
            '/vla/whisper_status',
            qos
        )
        
        self._publish_status("Command parser ready")
    
    def _publish_status(self, message: str):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
        self.get_logger().info(message)
    
    def _transcription_callback(self, msg: String):
        """Handle incoming transcription"""
        try:
            # Parse JSON transcription
            data = json.loads(msg.data)
            transcription = data.get('transcription', '')
            command_id = data.get('command_id', str(uuid.uuid4()))
            whisper_confidence = data.get('confidence', 1.0)
            
            if not transcription.strip():
                self.get_logger().debug("Empty transcription, skipping")
                return
            
            # Parse the transcription
            intent = self.parser.parse(transcription, command_id)
            
            # Adjust confidence based on Whisper confidence
            intent.parse_confidence *= whisper_confidence
            
            # Check confidence threshold
            if intent.parse_confidence < self.confidence_threshold:
                self.get_logger().warn(
                    f"Low confidence intent: {intent.parse_confidence:.2f} "
                    f"(threshold: {self.confidence_threshold})"
                )
                
                if self.unknown_handling == 'ignore':
                    return
            
            # Handle unknown intents
            if intent.intent_type == IntentType.UNKNOWN.value:
                if self.unknown_handling == 'ask_clarification':
                    self.get_logger().info(
                        f"Unknown intent for: '{transcription}' - requesting clarification"
                    )
                elif self.unknown_handling == 'ignore':
                    self.get_logger().debug(f"Ignoring unknown intent: '{transcription}'")
                    return
            
            # Publish parsed intent
            intent_msg = String()
            intent_msg.data = intent.to_json()
            self.intent_pub.publish(intent_msg)
            
            self.get_logger().info(
                f"Parsed: '{transcription}' -> "
                f"type={intent.intent_type}, verb={intent.action_verb}, "
                f"object={intent.target_object}, location={intent.target_location}, "
                f"confidence={intent.parse_confidence:.2f}"
            )
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in transcription: {e}")
        except Exception as e:
            self.get_logger().error(f"Error parsing transcription: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CommandParserNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
