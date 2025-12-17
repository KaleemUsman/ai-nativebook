#!/usr/bin/env python3
"""
Whisper Pipeline Launch File

Launches the complete voice-to-action pipeline:
1. audio_capture - Captures audio from microphone with VAD
2. whisper_transcriber - Transcribes audio using Whisper
3. command_parser - Parses transcriptions into structured intents

Usage:
    ros2 launch examples/vla/whisper/launch/whisper_pipeline.launch.py
    
    # With custom parameters:
    ros2 launch examples/vla/whisper/launch/whisper_pipeline.launch.py \
        model_size:=medium device:=cuda
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for the Whisper pipeline."""
    
    # Get the path to the config file
    # Note: In a real ROS 2 package, use get_package_share_directory
    # For now, we use a relative path from this file
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'config'
    )
    default_config = os.path.join(config_dir, 'whisper_config.yaml')
    
    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to the whisper configuration file'
        ),
        DeclareLaunchArgument(
            'model_size',
            default_value='small',
            description='Whisper model size: tiny, base, small, medium, large-v3'
        ),
        DeclareLaunchArgument(
            'device',
            default_value='auto',
            description='Compute device: cuda, cpu, or auto'
        ),
        DeclareLaunchArgument(
            'language',
            default_value='en',
            description='Target language for transcription'
        ),
        DeclareLaunchArgument(
            'vad_enabled',
            default_value='true',
            description='Enable Voice Activity Detection'
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.5',
            description='Minimum confidence for intent parsing'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='INFO',
            description='Logging level: DEBUG, INFO, WARNING, ERROR'
        ),
    ]
    
    # Audio Capture Node
    audio_capture_node = Node(
        package='vla_nodes',
        executable='audio_capture',
        name='audio_capture',
        output='screen',
        parameters=[{
            'device_id': -1,  # Default device
            'sample_rate': 16000,
            'channels': 1,
            'chunk_duration_ms': 100,
            'vad.enabled': LaunchConfiguration('vad_enabled'),
            'vad.energy_threshold': -40.0,
            'vad.speech_pad_ms': 300,
            'vad.min_speech_duration_ms': 500,
            'vad.max_speech_duration_ms': 15000,
            'vad.silence_duration_ms': 700,
        }],
        remappings=[
            ('/vla/audio_chunk', '/vla/audio_chunk'),
            ('/vla/audio_level', '/vla/audio_level'),
            ('/vla/whisper_status', '/vla/whisper_status'),
        ],
    )
    
    # Whisper Transcriber Node
    whisper_transcriber_node = Node(
        package='vla_nodes',
        executable='whisper_transcriber',
        name='whisper_transcriber',
        output='screen',
        parameters=[{
            'model_size': LaunchConfiguration('model_size'),
            'language': LaunchConfiguration('language'),
            'device': LaunchConfiguration('device'),
            'local_model': True,
            'temperature': 0.0,
            'word_timestamps': False,
            'no_speech_threshold': 0.6,
        }],
        remappings=[
            ('/vla/audio_chunk', '/vla/audio_chunk'),
            ('/vla/transcription', '/vla/transcription'),
            ('/vla/whisper_status', '/vla/whisper_status'),
        ],
    )
    
    # Command Parser Node
    command_parser_node = Node(
        package='vla_nodes',
        executable='command_parser',
        name='command_parser',
        output='screen',
        parameters=[{
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'unknown_intent_handling': 'ask_clarification',
        }],
        remappings=[
            ('/vla/transcription', '/vla/transcription'),
            ('/vla/parsed_intent', '/vla/parsed_intent'),
            ('/vla/whisper_status', '/vla/whisper_status'),
        ],
    )
    
    # Log startup info
    log_startup = LogInfo(
        msg=['Launching VLA Whisper Pipeline with model=', LaunchConfiguration('model_size')]
    )
    
    return LaunchDescription(
        declared_arguments + [
            log_startup,
            audio_capture_node,
            whisper_transcriber_node,
            command_parser_node,
        ]
    )
