# Quickstart Guide: Gazebo & Unity Simulation Module

## Overview
This guide provides a quick path to get started with the Digital Twin simulation environment combining Gazebo physics simulation and Unity high-fidelity rendering for humanoid robotics.

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS or Windows 10/11 with WSL2
- Python 3.8+ with pip
- At least 8GB RAM (16GB recommended)
- NVIDIA GPU recommended for Unity rendering (optional but improves performance)

### Software Dependencies
1. **ROS 2 Humble Hawksbill**
   - Install via: https://docs.ros.org/en/humble/Installation.html
   - Source the ROS 2 environment: `source /opt/ros/humble/setup.bash`

2. **Gazebo Garden/Fortress**
   - Install via: `sudo apt install ros-humble-gazebo-*` (on Ubuntu)
   - Or download from: https://gazebosim.org/docs/garden/install/

3. **Unity 2022.3+ LTS**
   - Download from: https://unity.com/releases/editor/whats-new/2022.3.0
   - Install with Universal Render Pipeline (URP) package

4. **Python Dependencies**
   ```bash
   pip3 install rclpy transforms3d numpy matplotlib
   ```

## Setup Steps

### 1. Clone and Prepare Repository
```bash
# Navigate to your workspace
cd ~/ros2_workspace/src
git clone [your-repo-url]
cd [repo-name]
```

### 2. Install ROS Packages
```bash
# From the repository root
cd ~/ros2_workspace
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select gazebo_ros_pkgs
source install/setup.bash
```

### 3. Set Up Unity Project
1. Open Unity Hub
2. Create a new 3D project named "HumanoidSimulation"
3. Import the Unity Robotics Package via Package Manager
4. Add the example scripts from `examples/gazebo-unity/unity/scripts` to your project

### 4. Verify Gazebo Installation
```bash
# Test basic Gazebo functionality
gz sim --version
# Should show Gazebo Garden or Fortress version

# Launch simple test environment
gz sim -r examples/worlds/shapes.sdf
```

## Running the Basic Example

### 1. Start ROS 2 Environment
```bash
# Terminal 1: Source ROS 2
source /opt/ros/humble/setup.bash
source ~/ros2_workspace/install/setup.bash

# Launch Gazebo with humanoid robot
ros2 launch examples/gazebo-unity/gazebo/launch/humanoid_sim.launch.py
```

### 2. Start Unity Visualization (Conceptual)
In a real implementation, you would:
1. Open the Unity project
2. Load the humanoid scene
3. Configure the ROS bridge connection
4. Run the simulation to visualize the robot

### 3. Launch Sensor Data Publisher
```bash
# Terminal 2: Run sensor simulation example
source /opt/ros/humble/setup.bash
source ~/ros2_workspace/install/setup.bash
python3 examples/gazebo-unity/ros-bridge/python-examples/sensor_publisher.py
```

### 4. Connect AI Agent (Example)
```bash
# Terminal 3: Run basic AI agent example
source /opt/ros/humble/setup.bash
source ~/ros2_workspace/install/setup.bash
python3 examples/gazebo-unity/ros-bridge/python-examples/ai_agent_example.py
```

## Key Commands

### Gazebo Commands
```bash
# Launch with specific world
gz sim -r path/to/world.sdf

# List active topics
ros2 topic list

# Echo sensor data
ros2 topic echo /humanoid/laser_scan sensor_msgs/msg/LaserScan
```

### ROS 2 Commands for Simulation
```bash
# Check active nodes
ros2 node list

# Get node information
ros2 node info /humanoid_controller

# Send joint commands
ros2 action send_goal /humanoid/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{trajectory: ...}"
```

## Troubleshooting

### Common Issues

1. **Gazebo fails to start**
   - Check X11 forwarding if using WSL2
   - Ensure proper graphics drivers are installed
   - Try running with `--verbose` flag

2. **ROS bridge connection fails**
   - Verify ROS 2 domain ID matches between components
   - Check network connectivity between systems
   - Ensure ROS_MASTER_URI is properly set

3. **Unity-ROS communication issues**
   - Verify WebSocket connections are allowed
   - Check firewall settings
   - Confirm ROS bridge server is running

### Verification Steps
```bash
# Verify ROS 2 installation
ros2 topic list | grep -E "(scan|imu|camera|joint)"

# Check Gazebo simulation status
gz topic -l

# Verify Unity can connect to ROS bridge
# (Implementation-specific based on chosen bridge solution)
```

## Next Steps

1. Follow the detailed chapters in the documentation:
   - Gazebo Physics Simulation
   - Unity High-Fidelity Rendering
   - Sensor Integration
   - Digital Twin Integration

2. Experiment with the example environments in `examples/gazebo-unity/gazebo/environments/`

3. Try importing your own URDF models following the examples in `examples/gazebo-unity/humanoid-models/`

4. Explore AI agent integration examples in `examples/gazebo-unity/ros-bridge/python-examples/`