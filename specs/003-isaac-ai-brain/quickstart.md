# Quickstart Guide: The AI-Robot Brain (NVIDIA Isaac™)

## Prerequisites

- Ubuntu 22.04 LTS (recommended) or Windows 10/11
- NVIDIA GPU with Compute Capability 6.0+ (GTX 1060 or better)
- Isaac Sim 2023.1+ installed and licensed
- ROS 2 Humble Hawksbill installed
- Isaac ROS 3.0+ installed
- Nav2 installed and configured

## Setup Isaac Sim Environment

1. Launch Isaac Sim:
   ```bash
   cd /path/to/isaac-sim
   python3 -m omni.isaac.kit
   ```

2. Import the humanoid robot model:
   ```bash
   # In Isaac Sim, go to File -> Import -> USD/URDF
   # Select the humanoid robot model file from examples/gazebo-unity/isaac-sim/models/
   ```

3. Configure a photorealistic environment:
   ```bash
   # In Isaac Sim, load a scene from examples/gazebo-unity/isaac-sim/environments/
   # Adjust lighting and environmental settings as needed
   ```

## Configure Isaac ROS Perception Pipelines

1. Source ROS 2 Humble:
   ```bash
   source /opt/ros/humble/setup.bash
   source /path/to/isaac-ros-workspace/install/setup.bash
   ```

2. Launch the perception pipeline:
   ```bash
   ros2 launch examples/gazebo-unity/isaac-ros/launch/perception_pipeline.launch.py
   ```

3. Verify sensor data is flowing:
   ```bash
   ros2 topic list | grep sensor
   ros2 topic echo /camera/rgb/image_raw
   ros2 topic echo /lidar/points
   ros2 topic echo /imu/data
   ```

## Set up Humanoid Navigation with Nav2

1. Configure Nav2 for humanoid-specific parameters:
   ```bash
   # Navigate to the nav2 config directory
   cd examples/gazebo-unity/nav2/config/

   # Review and adjust parameters in humanoid_nav_params.yaml
   # Pay special attention to bipedal-specific settings
   ```

2. Launch the navigation stack:
   ```bash
   ros2 launch examples/gazebo-unity/nav2/launch/navigation.launch.py
   ```

3. Send a navigation goal:
   ```bash
   ros2 run nav2_msgs navigation_goal_sender --goal-x 1.0 --goal-y 1.0
   ```

## Generate Synthetic Datasets

1. Configure the data generation pipeline:
   ```bash
   cd examples/gazebo-unity/isaac-sim/scripts/
   python3 configure_data_generation.py --environment office --lighting varied
   ```

2. Run the synthetic data generation:
   ```bash
   python3 generate_synthetic_data.py --output-path /path/to/dataset --duration 3600
   ```

## Verify Implementation

1. Test the complete pipeline:
   ```bash
   # In Isaac Sim, ensure the humanoid robot is properly simulated
   # Verify perception data is being processed by Isaac ROS nodes
   # Confirm navigation goals are being executed successfully
   ```

2. Validate against success criteria:
   ```bash
   # Run the validation script
   python3 validate_implementation.py

   # Expected results:
   # - Isaac Sim running at 60+ FPS
   # - VSLAM position error < 5cm
   # - Navigation success rate > 90%
   ```

## Troubleshooting

- **Isaac Sim performance issues**: Check GPU compute capability and driver compatibility
- **Sensor data not flowing**: Verify Isaac ROS bridge configuration
- **Navigation failing**: Check Nav2 parameter configuration for humanoid constraints
- **Dataset generation errors**: Validate Isaac Sim scene setup and lighting conditions

## Next Steps

- Follow the detailed documentation in docs/modules/isaac-ai-brain/
- Experiment with different environments and lighting conditions
- Test synthetic-to-real transfer with your specific AI models
- Explore advanced perception and navigation features