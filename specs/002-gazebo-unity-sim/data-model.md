# Data Model: Gazebo & Unity Simulation Module

## Overview
This document defines the key data entities and their relationships for the Digital Twin simulation module. Since this is primarily a documentation module with simulation examples, the "data model" focuses on the conceptual entities and data flows rather than traditional database schemas.

## Key Entities

### 1. Gazebo Simulation Environment
- **Name**: String identifier for the environment
- **Description**: Text description of the environment purpose
- **Physics Parameters**:
  - Gravity vector (x, y, z)
  - Air density
  - Time step settings
- **Objects**: Collection of models/entities in the environment
- **Sensors**: Collection of sensor configurations
- **Lighting**: Environmental lighting properties

### 2. Unity Visualization Scene
- **Name**: String identifier for the scene
- **Description**: Text description of the scene purpose
- **Camera Settings**: Position, angle, and rendering parameters
- **Lighting**: Scene lighting configuration
- **Materials**: Collection of material properties
- **Models**: Collection of 3D model references

### 3. Humanoid Robot Model
- **Name**: String identifier for the robot
- **URDF Path**: File path to the URDF definition
- **Links**: Collection of rigid body components with properties:
  - Mass
  - Inertia tensor
  - Visual geometry
  - Collision geometry
- **Joints**: Collection of joint connections with properties:
  - Type (revolute, prismatic, fixed, etc.)
  - Limits (position, velocity, effort)
  - Axis of rotation/translation
- **Transmissions**: Actuator-to-joint mappings

### 4. Sensor Configuration
- **Type**: Sensor type (LiDAR, Depth Camera, IMU, etc.)
- **Name**: String identifier
- **Parent Link**: Which robot link the sensor is attached to
- **Position**: 3D offset from parent link
- **Orientation**: 3D rotation from parent link
- **Parameters**: Type-specific configuration (range, resolution, etc.)
- **ROS Topic**: Topic name for sensor data publication

### 5. ROS Bridge Connection
- **Connection Type**: WebSocket, TCP/IP, etc.
- **Host Address**: Network address of the ROS master
- **Port**: Communication port
- **Topics**: Collection of subscribed/published topics
- **Message Types**: ROS message types for each topic
- **QoS Settings**: Quality of Service configuration

### 6. Simulation Session
- **ID**: Unique identifier for the session
- **Start Time**: Timestamp of session start
- **Environment**: Reference to the Gazebo environment
- **Scene**: Reference to the Unity scene
- **Robot Models**: Collection of robots in the simulation
- **Sensor Data Streams**: Active sensor data being collected
- **AI Agent Connection**: Reference to connected AI agent (if applicable)

## Relationships

### Environment-Model Relationships
- A Gazebo Simulation Environment contains multiple Humanoid Robot Models
- A Unity Visualization Scene contains multiple Humanoid Robot Models
- A Humanoid Robot Model can exist in multiple environments/scenes

### Model-Sensor Relationships
- A Humanoid Robot Model contains multiple Sensor Configurations
- A Sensor Configuration belongs to one Humanoid Robot Model
- A Sensor Configuration connects to ROS Bridge Connection

### Simulation Data Flow
- Gazebo Simulation Environment produces Sensor Data
- Unity Visualization Scene consumes/visualizes Sensor Data
- ROS Bridge Connection facilitates data exchange between components
- Simulation Session orchestrates the entire data flow

## Data Flow Patterns

### Physics Simulation Data Flow
1. Gazebo Environment calculates physics
2. Robot models respond to forces and collisions
3. Sensors generate data based on physics state
4. Data published to ROS topics via bridge

### Visualization Data Flow
1. Unity Scene receives state updates
2. Models are positioned and animated
3. Camera and lighting applied
4. Rendered output displayed

### AI Integration Data Flow
1. Sensor data collected from simulation
2. AI agent processes sensor information
3. Control commands sent to robot models
4. Robot behavior affects simulation state

## Constraints and Validation Rules

- All URDF models must be valid XML with proper structure
- Sensor configurations must match available ROS message types
- Physics parameters must be within realistic ranges
- Unity scenes must maintain performance targets (30+ FPS)
- ROS communication must maintain real-time constraints
- Sensor data must be published at appropriate frequencies for the sensor type