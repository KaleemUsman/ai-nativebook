# Quickstart: ROS 2 Book Module

## Overview

This quickstart guide helps you get up and running with the ROS 2 Book Module, including how to run the example code and understand the documentation structure.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Python 3.8 or higher
- Basic Python programming knowledge
- Familiarity with command line tools

## Setup

### 1. Environment Setup

```bash
# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Create a workspace for examples
mkdir -p ~/ros2_book_examples/src
cd ~/ros2_book_examples

# The example code can be found in the examples/ros2 directory of this repository
```

### 2. Verify ROS 2 Installation

```bash
# Check ROS 2 version
ros2 --version

# Verify rclpy is available
python3 -c "import rclpy; print('rclpy available')"
```

## Running Examples

### Publisher/Subscriber Example

1. Navigate to the publisher example:
```bash
cd examples/ros2/publisher_subscriber
```

2. Source ROS 2 environment:
```bash
source /opt/ros/humble/setup.bash
```

3. Run the publisher:
```bash
python3 minimal_publisher.py
```

4. In a new terminal, run the subscriber:
```bash
source /opt/ros/humble/setup.bash
python3 minimal_subscriber.py
```

### Service/Client Example

1. Navigate to the service example:
```bash
cd examples/ros2/service_client
```

2. Run the service server:
```bash
python3 minimal_service.py
```

3. In a new terminal, call the service:
```bash
python3 minimal_client.py
```

## Understanding the Content Structure

### Chapter Organization

The ROS 2 module is organized into 4 chapters:

1. **Architecture** - Core ROS 2 concepts including nodes, executors, and DDS
2. **Communication** - Patterns like topics, services, and QoS settings
3. **AI Integration** - Connecting AI agents to ROS controllers
4. **URDF Modeling** - Creating robot descriptions for humanoid robots

### Code Example Format

Each code example follows this structure:
- Clear problem statement
- Minimal implementation
- Explanation of key concepts
- Connection to broader ROS 2 architecture

## Development Workflow

### For Content Contributors

1. Create MDX files in the docs/modules/ros2/ directory
2. Follow the frontmatter template:
```md
---
title: Your Chapter Title
sidebar_position: X
---
```

3. Include runnable code examples with syntax highlighting:
```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    # Your code here
```

### For Code Example Testing

1. Ensure all examples run in a clean ROS 2 Humble environment
2. Test with different QoS profiles where applicable
3. Verify URDF examples with `check_urdf` tool
4. Document any dependencies or special setup requirements

## Troubleshooting

### Common Issues

- **rclpy import errors**: Ensure ROS 2 environment is sourced
- **Node communication failures**: Check that nodes are on the same ROS domain
- **URDF validation errors**: Use `check_urdf <urdf_file>` to identify issues

### Verification Steps

1. Confirm all examples run without errors
2. Verify all technical claims against official ROS 2 documentation
3. Test URDF models with ROS 2 tools
4. Ensure MDX content renders correctly in Docusaurus