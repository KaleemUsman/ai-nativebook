#!/usr/bin/env python3
"""
Capstone Launch File - Autonomous Humanoid Pipeline

Launches the complete VLA system including:
1. Voice capture and Whisper transcription
2. LLM-based cognitive planning
3. Plan execution with Module 2/3 integration
4. Error handling and speech feedback

Usage:
    ros2 launch examples/vla/capstone/launch/autonomous_humanoid.launch.py
    
    # With custom parameters:
    ros2 launch examples/vla/capstone/launch/autonomous_humanoid.launch.py \
        llm_model:=gpt-4-turbo simulation:=true
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    GroupAction
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    """Generate launch description for the complete VLA pipeline."""
    
    # Base paths
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    whisper_launch_dir = os.path.join(pkg_dir, 'whisper', 'launch')
    
    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            'whisper_model',
            default_value='small',
            description='Whisper model size'
        ),
        DeclareLaunchArgument(
            'llm_model',
            default_value='gpt-4',
            description='LLM model for planning'
        ),
        DeclareLaunchArgument(
            'llm_provider',
            default_value='openai',
            description='LLM provider (openai, ollama)'
        ),
        DeclareLaunchArgument(
            'simulation',
            default_value='true',
            description='Run in simulation mode'
        ),
        DeclareLaunchArgument(
            'use_gpu',
            default_value='true',
            description='Use GPU acceleration'
        ),
        DeclareLaunchArgument(
            'enable_speech',
            default_value='true',
            description='Enable speech synthesis'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='INFO',
            description='Logging level'
        ),
    ]
    
    # ===== Voice Pipeline Nodes =====
    
    # Audio Capture
    audio_capture_node = Node(
        package='vla_nodes',
        executable='audio_capture',
        name='audio_capture',
        output='screen',
        parameters=[{
            'device_id': -1,
            'sample_rate': 16000,
            'channels': 1,
            'vad.enabled': True,
            'vad.energy_threshold': -40.0,
            'vad.min_speech_duration_ms': 500,
        }],
    )
    
    # Whisper Transcriber
    whisper_transcriber_node = Node(
        package='vla_nodes',
        executable='whisper_transcriber',
        name='whisper_transcriber',
        output='screen',
        parameters=[{
            'model_size': LaunchConfiguration('whisper_model'),
            'language': 'en',
            'device': 'cuda' if LaunchConfiguration('use_gpu') else 'cpu',
            'local_model': True,
        }],
    )
    
    # Command Parser
    command_parser_node = Node(
        package='vla_nodes',
        executable='command_parser',
        name='command_parser',
        output='screen',
        parameters=[{
            'confidence_threshold': 0.5,
            'unknown_intent_handling': 'ask_clarification',
        }],
    )
    
    # ===== LLM Planning Nodes =====
    
    # Context Manager
    context_manager_node = Node(
        package='vla_nodes',
        executable='context_manager',
        name='context_manager',
        output='screen',
        parameters=[{
            'object_timeout_s': 60.0,
            'max_history_items': 10,
            'publish_rate_hz': 2.0,
        }],
    )
    
    # LLM Planner
    llm_planner_node = Node(
        package='vla_nodes',
        executable='llm_planner',
        name='llm_planner',
        output='screen',
        parameters=[{
            'llm.provider': LaunchConfiguration('llm_provider'),
            'llm.model': LaunchConfiguration('llm_model'),
            'llm.temperature': 0.2,
            'llm.timeout_s': 15.0,
            'llm.fallback_enabled': True,
            'llm.fallback_provider': 'ollama',
            'llm.fallback_model': 'llama3',
            'planning.validate_plans': True,
            'planning.max_plan_length': 15,
        }],
    )
    
    # ===== Execution Nodes =====
    
    # Plan Executor
    plan_executor_node = Node(
        package='vla_nodes',
        executable='plan_executor',
        name='plan_executor',
        output='screen',
        parameters=[{
            'execution.max_retries': 3,
            'execution.primitive_timeout_s': 60.0,
            'execution.abort_on_failure': False,
            'feedback.publish_rate_hz': 10.0,
        }],
    )
    
    # ===== Support Nodes =====
    
    # Speech Synthesis (conditional)
    speech_synthesis_node = Node(
        package='vla_nodes',
        executable='speech_synthesis',
        name='speech_synthesis',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_speech'))
    )
    
    # ===== Logging =====
    
    log_startup = LogInfo(
        msg=['Starting Autonomous Humanoid VLA Pipeline: ',
             'LLM=', LaunchConfiguration('llm_model'),
             ', Whisper=', LaunchConfiguration('whisper_model')]
    )
    
    # Return launch description
    return LaunchDescription(
        declared_arguments + [
            log_startup,
            
            # Voice Pipeline
            audio_capture_node,
            whisper_transcriber_node,
            command_parser_node,
            
            # LLM Planning
            context_manager_node,
            llm_planner_node,
            
            # Execution
            plan_executor_node,
            
            # Support
            speech_synthesis_node,
        ]
    )
