# ROS 2 Terminology Standards

## Overview
This document defines consistent terminology for ROS 2 concepts across all chapters of the book module.

## Core Concepts

### Node
- **Definition**: An executable that uses ROS 2 client library to communicate with other nodes
- **Usage**: Always refer to as "ROS 2 node" or simply "node"
- **Variants**: Avoid "ROS node", "process", or "component" when referring to nodes

### Topic
- **Definition**: A named bus over which nodes exchange messages
- **Usage**: Always refer to as "topic" (not "channel", "bus", or "connection")
- **Related**: Publisher-subscriber communication pattern

### Publisher
- **Definition**: A node that sends messages to a topic
- **Usage**: "publisher node" or "publisher"
- **Code context**: Created using `create_publisher()`

### Subscriber
- **Definition**: A node that receives messages from a topic
- **Usage**: "subscriber node" or "subscriber"
- **Code context**: Created using `create_subscription()`

### Service
- **Definition**: A synchronous request-response communication pattern between nodes
- **Usage**: "service" (not "RPC" or "function call")
- **Related**: Client-service communication pattern

### Client
- **Definition**: A node that makes requests to a service
- **Usage**: "client node" or "client"
- **Code context**: Created using `create_client()`

### Service Server
- **Definition**: A node that responds to service requests
- **Usage**: "service server" (not "service provider" or "server")
- **Code context**: Created using `create_service()`

### DDS
- **Definition**: Data Distribution Service - the underlying middleware in ROS 2
- **Usage**: Always "DDS" (not "middleware" unless in broader context)
- **Full form**: When first used in a chapter: "Data Distribution Service (DDS)"

### Quality of Service (QoS)
- **Definition**: Configurable policies that govern delivery of messages
- **Usage**: "QoS" or "Quality of Service" (never "QOS" or "qos")
- **Full form**: When first used in a chapter: "Quality of Service (QoS)"

### Executor
- **Definition**: Manages the execution of callbacks from subscriptions, services, and timers
- **Usage**: "executor" (not "scheduler" or "handler")
- **Types**: Single-threaded executor, multi-threaded executor

### URDF
- **Definition**: Unified Robot Description Format
- **Usage**: Always "URDF" (not "robot description" unless in broader context)
- **Full form**: When first used in a chapter: "Unified Robot Description Format (URDF)"

### Link
- **Definition**: A rigid body in a URDF robot model
- **Usage**: "link" (not "body" or "part" in URDF context)
- **Related**: Connected to other links via joints

### Joint
- **Definition**: A connection between two links in a URDF robot model
- **Usage**: "joint" (not "connection" or "hinge" in URDF context)
- **Types**: revolute, continuous, prismatic, fixed, etc.

## Code-Related Terms

### rclpy
- **Definition**: Python client library for ROS 2
- **Usage**: Always "rclpy" (not "ROS 2 Python library" or "Python ROS library")
- **Context**: Python-specific ROS 2 client library

### Message
- **Definition**: Data structure sent between nodes via topics
- **Usage**: "message" (not "data packet" or "payload")
- **Related**: Service requests and responses are also messages

### Action
- **Definition**: A goal-request-feedback-result communication pattern
- **Usage**: "action" (not "task" or "process" in ROS 2 context)
- **Components**: Goal, feedback, and result

## Writing Guidelines

1. Always use the defined terms consistently across all chapters
2. When introducing a term for the first time in a chapter, provide the full form followed by the abbreviation in parentheses
3. Use the shorter form for subsequent references in the same chapter
4. Maintain consistency with official ROS 2 documentation terminology
5. Avoid colloquialisms or informal terms that might confuse readers