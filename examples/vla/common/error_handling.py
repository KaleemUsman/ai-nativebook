#!/usr/bin/env python3
"""
Error Handling Utilities for VLA Pipeline

Provides tiered error recovery, safety procedures, and error reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors"""
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MANIPULATION = "manipulation"
    PLANNING = "planning"
    COMMUNICATION = "communication"
    HARDWARE = "hardware"
    NETWORK = "network"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Possible recovery actions"""
    RETRY = "retry"
    RETRY_MODIFIED = "retry_modified"
    SKIP = "skip"
    REPLAN = "replan"
    ASK_USER = "ask_user"
    ABORT = "abort"
    SAFETY_STOP = "safety_stop"


@dataclass
class VLAError:
    """Represents an error in the VLA system"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recoverable: bool = True
    suggested_recovery: RecoveryAction = RecoveryAction.RETRY
    
    def to_dict(self) -> Dict:
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
            "suggested_recovery": self.suggested_recovery.value
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# Standard error codes
class ErrorCode:
    """Standard error codes for VLA system"""
    
    # Navigation errors (1xxx)
    NAV_GOAL_UNREACHABLE = "NAV_1001"
    NAV_PATH_BLOCKED = "NAV_1002"
    NAV_LOCALIZATION_LOST = "NAV_1003"
    NAV_TIMEOUT = "NAV_1004"
    NAV_COLLISION_RISK = "NAV_1005"
    
    # Perception errors (2xxx)
    PERC_OBJECT_NOT_FOUND = "PERC_2001"
    PERC_DETECTION_FAILED = "PERC_2002"
    PERC_SENSOR_FAILURE = "PERC_2003"
    PERC_LOW_CONFIDENCE = "PERC_2004"
    PERC_OCCLUSION = "PERC_2005"
    
    # Manipulation errors (3xxx)
    MANIP_GRASP_FAILED = "MANIP_3001"
    MANIP_OBJECT_DROPPED = "MANIP_3002"
    MANIP_UNREACHABLE = "MANIP_3003"
    MANIP_COLLISION = "MANIP_3004"
    MANIP_JOINT_LIMIT = "MANIP_3005"
    
    # Planning errors (4xxx)
    PLAN_GENERATION_FAILED = "PLAN_4001"
    PLAN_INVALID = "PLAN_4002"
    PLAN_PRECONDITION_FAILED = "PLAN_4003"
    PLAN_IMPOSSIBLE = "PLAN_4004"
    
    # Communication errors (5xxx)
    COMM_TTS_FAILED = "COMM_5001"
    COMM_ASR_FAILED = "COMM_5002"
    COMM_LLM_TIMEOUT = "COMM_5003"
    COMM_API_ERROR = "COMM_5004"
    
    # Hardware errors (6xxx)
    HW_MOTOR_FAILURE = "HW_6001"
    HW_SENSOR_FAILURE = "HW_6002"
    HW_BATTERY_LOW = "HW_6003"
    HW_EMERGENCY_STOP = "HW_6004"


# Error templates
ERROR_TEMPLATES: Dict[str, Dict] = {
    ErrorCode.NAV_GOAL_UNREACHABLE: {
        "category": ErrorCategory.NAVIGATION,
        "severity": ErrorSeverity.ERROR,
        "message": "Cannot reach the specified goal",
        "recoverable": True,
        "suggested_recovery": RecoveryAction.REPLAN
    },
    ErrorCode.NAV_PATH_BLOCKED: {
        "category": ErrorCategory.NAVIGATION,
        "severity": ErrorSeverity.WARNING,
        "message": "Path is blocked by an obstacle",
        "recoverable": True,
        "suggested_recovery": RecoveryAction.RETRY_MODIFIED
    },
    ErrorCode.PERC_OBJECT_NOT_FOUND: {
        "category": ErrorCategory.PERCEPTION,
        "severity": ErrorSeverity.WARNING,
        "message": "Target object was not found",
        "recoverable": True,
        "suggested_recovery": RecoveryAction.RETRY
    },
    ErrorCode.MANIP_GRASP_FAILED: {
        "category": ErrorCategory.MANIPULATION,
        "severity": ErrorSeverity.ERROR,
        "message": "Failed to grasp the object",
        "recoverable": True,
        "suggested_recovery": RecoveryAction.RETRY_MODIFIED
    },
    ErrorCode.PLAN_GENERATION_FAILED: {
        "category": ErrorCategory.PLANNING,
        "severity": ErrorSeverity.ERROR,
        "message": "Failed to generate action plan",
        "recoverable": True,
        "suggested_recovery": RecoveryAction.ASK_USER
    },
    ErrorCode.COMM_LLM_TIMEOUT: {
        "category": ErrorCategory.COMMUNICATION,
        "severity": ErrorSeverity.WARNING,
        "message": "LLM API request timed out",
        "recoverable": True,
        "suggested_recovery": RecoveryAction.RETRY
    },
    ErrorCode.HW_EMERGENCY_STOP: {
        "category": ErrorCategory.HARDWARE,
        "severity": ErrorSeverity.CRITICAL,
        "message": "Emergency stop activated",
        "recoverable": False,
        "suggested_recovery": RecoveryAction.SAFETY_STOP
    },
}


def create_error(
    code: str,
    details: Optional[Dict] = None,
    override_message: Optional[str] = None
) -> VLAError:
    """
    Create a VLAError from an error code.
    
    Args:
        code: Error code from ErrorCode class
        details: Additional error details
        override_message: Custom message to override template
        
    Returns:
        VLAError instance
    """
    import uuid
    
    template = ERROR_TEMPLATES.get(code, {
        "category": ErrorCategory.UNKNOWN,
        "severity": ErrorSeverity.ERROR,
        "message": "Unknown error occurred",
        "recoverable": True,
        "suggested_recovery": RecoveryAction.ABORT
    })
    
    return VLAError(
        error_id=str(uuid.uuid4()),
        category=template["category"],
        severity=template["severity"],
        code=code,
        message=override_message or template["message"],
        details=details or {},
        recoverable=template["recoverable"],
        suggested_recovery=template["suggested_recovery"]
    )


class ErrorHandler:
    """
    Tiered error handling with recovery strategies.
    
    Implements three-tier recovery:
    1. Primitive-level: Retry with adjusted parameters
    2. Plan-level: Request replanning from LLM
    3. User-level: Report to user and request guidance
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        replan_callback: Optional[Callable] = None,
        user_callback: Optional[Callable] = None,
        safety_callback: Optional[Callable] = None
    ):
        """
        Initialize the error handler.
        
        Args:
            max_retries: Maximum retry attempts per error
            replan_callback: Function to call for replanning
            user_callback: Function to call for user notification
            safety_callback: Function to call for safety stop
        """
        self.max_retries = max_retries
        self.replan_callback = replan_callback
        self.user_callback = user_callback
        self.safety_callback = safety_callback
        
        # Error history
        self.error_history: List[VLAError] = []
        self.retry_counts: Dict[str, int] = {}
    
    def handle(
        self,
        error: VLAError,
        context: Optional[Dict] = None
    ) -> tuple[RecoveryAction, Dict]:
        """
        Handle an error and determine recovery action.
        
        Args:
            error: The error to handle
            context: Additional context for recovery
            
        Returns:
            Tuple of (recovery_action, recovery_params)
        """
        # Log error
        self._log_error(error)
        
        # Record in history
        self.error_history.append(error)
        
        # Handle critical errors immediately
        if error.severity == ErrorSeverity.CRITICAL:
            return self._handle_critical(error)
        
        # Check if recoverable
        if not error.recoverable:
            return RecoveryAction.ABORT, {"reason": error.message}
        
        # Get retry count for this error code
        retry_key = f"{error.code}:{error.details.get('action_type', 'unknown')}"
        retries = self.retry_counts.get(retry_key, 0)
        
        # Tier 1: Retry at primitive level
        if retries < self.max_retries:
            if error.suggested_recovery in [RecoveryAction.RETRY, RecoveryAction.RETRY_MODIFIED]:
                self.retry_counts[retry_key] = retries + 1
                return self._tier1_recovery(error, retries)
        
        # Tier 2: Request replanning
        if error.suggested_recovery in [RecoveryAction.REPLAN, RecoveryAction.RETRY_MODIFIED]:
            return self._tier2_recovery(error, context)
        
        # Tier 3: Ask user
        return self._tier3_recovery(error)
    
    def _handle_critical(self, error: VLAError) -> tuple[RecoveryAction, Dict]:
        """Handle critical errors requiring immediate action"""
        logger.critical(f"Critical error: {error.code} - {error.message}")
        
        # Call safety callback if available
        if self.safety_callback:
            try:
                self.safety_callback(error)
            except Exception as e:
                logger.error(f"Safety callback failed: {e}")
        
        return RecoveryAction.SAFETY_STOP, {
            "reason": error.message,
            "error": error.to_dict()
        }
    
    def _tier1_recovery(
        self,
        error: VLAError,
        retry_count: int
    ) -> tuple[RecoveryAction, Dict]:
        """Tier 1: Primitive-level retry with parameter adjustment"""
        logger.info(f"Tier 1 recovery: Retry {retry_count + 1}/{self.max_retries}")
        
        params = {
            "retry_count": retry_count + 1,
            "adjustments": {}
        }
        
        # Adjust parameters based on error type
        if error.code == ErrorCode.MANIP_GRASP_FAILED:
            # Try different grasp approach
            params["adjustments"]["grasp_type"] = "precision" if retry_count == 0 else "pinch"
            params["adjustments"]["approach_direction"] = "side" if retry_count > 0 else "top"
        
        elif error.code == ErrorCode.NAV_PATH_BLOCKED:
            # Increase clearance
            params["adjustments"]["clearance_margin"] = 0.1 * (retry_count + 1)
        
        elif error.code == ErrorCode.PERC_OBJECT_NOT_FOUND:
            # Expand search area
            params["adjustments"]["scan_mode"] = "detailed"
        
        return RecoveryAction.RETRY_MODIFIED, params
    
    def _tier2_recovery(
        self,
        error: VLAError,
        context: Optional[Dict]
    ) -> tuple[RecoveryAction, Dict]:
        """Tier 2: Plan-level replanning"""
        logger.info("Tier 2 recovery: Requesting replan")
        
        # Reset retry counts for this action
        self.retry_counts.clear()
        
        # Call replan callback if available
        if self.replan_callback:
            try:
                replan_result = self.replan_callback(error, context)
                return RecoveryAction.REPLAN, {
                    "failure_reason": error.message,
                    "context": context,
                    "replan_result": replan_result
                }
            except Exception as e:
                logger.error(f"Replan callback failed: {e}")
        
        return RecoveryAction.REPLAN, {
            "failure_reason": error.message,
            "context": context
        }
    
    def _tier3_recovery(self, error: VLAError) -> tuple[RecoveryAction, Dict]:
        """Tier 3: User-level notification"""
        logger.info("Tier 3 recovery: Asking user for help")
        
        # Format user message
        user_message = self._format_user_message(error)
        
        # Call user callback if available
        if self.user_callback:
            try:
                self.user_callback(user_message, error)
            except Exception as e:
                logger.error(f"User callback failed: {e}")
        
        return RecoveryAction.ASK_USER, {
            "message": user_message,
            "error": error.to_dict()
        }
    
    def _format_user_message(self, error: VLAError) -> str:
        """Format error message for user display"""
        messages = {
            ErrorCode.NAV_GOAL_UNREACHABLE: 
                "I can't reach that location. Could you suggest an alternative?",
            ErrorCode.PERC_OBJECT_NOT_FOUND:
                "I couldn't find the object you asked for. Can you point it out or describe it differently?",
            ErrorCode.MANIP_GRASP_FAILED:
                "I wasn't able to pick up the object. Would you like me to try a different approach?",
            ErrorCode.PLAN_GENERATION_FAILED:
                "I'm not sure how to do that. Could you rephrase your request?",
        }
        
        return messages.get(error.code, f"I encountered an issue: {error.message}")
    
    def _log_error(self, error: VLAError):
        """Log error with appropriate level"""
        log_levels = {
            ErrorSeverity.INFO: logger.info,
            ErrorSeverity.WARNING: logger.warning,
            ErrorSeverity.ERROR: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }
        
        log_func = log_levels.get(error.severity, logger.error)
        log_func(f"[{error.code}] {error.message} - {error.details}")
    
    def get_error_summary(self) -> Dict:
        """Get summary of error history"""
        summary = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "recoverable": 0,
            "non_recoverable": 0
        }
        
        for error in self.error_history:
            # By category
            cat = error.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            
            # By severity
            sev = error.severity.value
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            
            # Recoverable
            if error.recoverable:
                summary["recoverable"] += 1
            else:
                summary["non_recoverable"] += 1
        
        return summary
    
    def clear_history(self):
        """Clear error history and retry counts"""
        self.error_history.clear()
        self.retry_counts.clear()


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(
    code: str,
    details: Optional[Dict] = None,
    context: Optional[Dict] = None
) -> tuple[RecoveryAction, Dict]:
    """Convenience function to handle an error"""
    error = create_error(code, details)
    return get_error_handler().handle(error, context)
