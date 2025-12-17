# Data Model: The AI-Robot Brain (NVIDIA Isaac™)

## Key Entities

### IsaacSimEnvironment
- **name**: string - Unique identifier for the simulation environment
- **description**: string - Brief description of the environment
- **scene_file**: string - Path to the Isaac Sim scene file (.usd/.sdf)
- **lighting_conditions**: array - List of supported lighting conditions for synthetic data
- **physics_properties**: object - Physics configuration (gravity, friction, etc.)
- **interactive_objects**: array - List of objects that can interact with the robot

### HumanoidRobotModel
- **model_name**: string - Name of the humanoid robot model
- **urdf_path**: string - Path to the URDF file describing the robot
- **sdf_path**: string - Path to the SDF file for simulation
- **kinematic_properties**: object - Joint limits, link masses, etc.
- **dynamic_properties**: object - Inertial properties, friction coefficients
- **sensor_configurations**: array - List of sensor types and mounting positions
- **actuator_configurations**: array - List of actuator types and control parameters

### PerceptionPipeline
- **pipeline_name**: string - Name of the perception pipeline
- **sensor_inputs**: array - List of sensor types the pipeline accepts
- **vslam_enabled**: boolean - Whether VSLAM is active in the pipeline
- **sensor_fusion_enabled**: boolean - Whether sensor fusion is active
- **processing_rate**: number - Rate at which the pipeline processes data (Hz)
- **hardware_acceleration**: boolean - Whether hardware acceleration is used
- **output_format**: string - Format of the processed perception data

### NavigationPlan
- **plan_id**: string - Unique identifier for the navigation plan
- **start_pose**: object - Starting position and orientation of the robot
- **goal_pose**: object - Target position and orientation
- **path_waypoints**: array - List of waypoints in the planned path
- **obstacle_avoidance**: boolean - Whether obstacle avoidance is active
- **bipedal_constraints**: object - Constraints specific to humanoid locomotion
- **trajectory_profile**: string - Profile for the movement trajectory

### SyntheticDataset
- **dataset_id**: string - Unique identifier for the dataset
- **name**: string - Name of the dataset
- **source_environment**: string - Environment where data was generated
- **sensor_config**: object - Configuration of sensors during data capture
- **lighting_conditions**: array - Lighting conditions during data capture
- **annotation_format**: string - Format of the annotations
- **size**: number - Number of samples in the dataset
- **creation_date**: string - Date when the dataset was created

## State Transitions

### HumanoidRobotModelState
- **UNCONFIGURED** → **CONFIGURING** → **ACTIVE** → **STOPPED**
- Transitions triggered by Isaac Sim simulation events and ROS 2 lifecycle management

### PerceptionPipelineState
- **UNCONFIGURED** → **CONFIGURING** → **PROCESSING** → **PAUSED** → **ERROR**
- Transitions based on sensor data availability and processing status

### NavigationState
- **IDLE** → **PLANNING** → **EXECUTING** → **COMPLETING** → **IDLE** or **ABORTED**
- Transitions based on goal achievement and obstacle detection

## Validation Rules

### IsaacSimEnvironment
- **name** must be unique within the simulation workspace
- **scene_file** must exist and be accessible to Isaac Sim
- **physics_properties** must be valid for the Isaac Sim physics engine

### HumanoidRobotModel
- **urdf_path** and **sdf_path** must point to valid robot description files
- **kinematic_properties** must allow for stable bipedal locomotion
- **sensor_configurations** must match actual sensor specifications

### PerceptionPipeline
- **processing_rate** must not exceed sensor data rate
- **sensor_inputs** must match available sensor data in the simulation
- **hardware_acceleration** requires compatible GPU hardware

### NavigationPlan
- **start_pose** and **goal_pose** must be within the same environment
- **bipedal_constraints** must be physically feasible for the robot model
- **path_waypoints** must not result in collisions with known obstacles