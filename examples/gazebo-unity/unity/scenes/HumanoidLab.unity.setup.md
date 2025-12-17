# HumanoidLab Unity Scene Setup Guide

This document provides instructions for creating the HumanoidLab.unity scene file in Unity for humanoid robot visualization.

## Scene Setup Instructions

### 1. Create New Scene
- Open Unity and create a new 3D project or open an existing one
- Go to File → New Scene
- Save the scene as "HumanoidLab.unity" in the examples/gazebo-unity/unity/scenes/ directory

### 2. Set Up Environment
- Add a Plane as the floor (GameObject → 3D Object → Plane)
  - Scale: (5, 1, 5) for a 5x5 meter area
  - Position: (0, 0, 0)
  - Apply a suitable material (e.g., concrete or laboratory floor)

- Add boundary walls:
  - Create 4 cubes as walls (GameObject → 3D Object → Cube)
  - Position them around the perimeter:
    - North wall: Position (0, 0.5, 2.5), Scale (5, 1, 0.1)
    - South wall: Position (0, 0.5, -2.5), Scale (5, 1, 0.1)
    - East wall: Position (2.5, 0.5, 0), Scale (0.1, 1, 5)
    - West wall: Position (-2.5, 0.5, 0), Scale (0.1, 1, 5)

### 3. Configure Lighting
- Select the default Directional Light
- Set the following properties:
  - Direction: (0.2, -1, -0.2) to simulate overhead lighting
  - Color: Warm white (255, 255, 255, 1)
  - Intensity: 1.0
  - Shadow Type: Cascaded

- Add additional Point Lights if needed:
  - Position them at (2, 2, 2) and (-2, 2, -2)
  - Range: 10
  - Intensity: 0.5

### 4. Add Reflection Probes (Optional)
- GameObject → Light → Reflection Probe
- Set Type to "Box"
- Position at (0, 1, 0)
- Set Size to (6, 3, 6)

### 5. Import Humanoid Robot
- Use the URDF Importer (GameObject → URDF Importer → From File Path)
- Select your humanoid URDF file
- Configure import settings as needed
- Position the robot appropriately in the scene (e.g., at (0, 0.5, 0))

### 6. Add Cameras
- Main Camera:
  - Position: (0, 3, 5)
  - Rotation: (0, 0, 0)
  - Field of View: 60

- Overhead Camera:
  - Create new camera (GameObject → Camera)
  - Position: (0, 8, 0)
  - Rotation: (-90, 0, 0)
  - Set Projection to Orthographic
  - Orthographic Size: 4

### 7. Configure Physics Materials
- Create new Physic Material in Project window
- Name it "RobotFeetMaterial"
- Set Dynamic Friction: 0.6, Static Friction: 0.6, Bounciness: 0
- Apply to robot foot colliders if present

### 8. Add Humanoid Controller Script
- Attach the HumanoidController.cs script to the robot's root object
- Configure the joint mappings in the Inspector:
  - Assign each joint's ArticulationBody to the corresponding joint in the script
  - Set appropriate joint limits and types

### 9. Set Up Lighting Settings (Optional)
- Go to Window → Rendering → Lighting Settings
- Configure Environment Lighting:
  - Source: Skybox
  - Intensity: 1.0
- Configure Realtime Lighting:
  - Bounce Scale: 1
  - Indirect Intensity: 1
  - Albedo Boost: 1
- Configure Mixed Lighting:
  - Technique: Baked Indirect
  - Directional Mode: Non-Directional

### 10. Final Organization
- Create the following empty GameObjects as organizational parents:
  - "Environment" (parent to floor and walls)
  - "Lighting" (parent to lights)
  - "Cameras" (parent to cameras)
  - "Robot" (parent to the imported robot)

### 11. Save Scene
- Save the scene (Ctrl+S) as "HumanoidLab.unity"
- Ensure all assets are properly referenced

## Scene Configuration Notes

### Coordinate System
- Unity uses Y-up coordinate system
- The imported URDF should be converted from Z-up (ROS) to Y-up (Unity)
- Verify that the robot stands upright after import

### Scale Considerations
- Ensure 1 Unity unit equals 1 meter for proper ROS integration
- Check that robot dimensions match real-world values

### Performance Optimization
- Use appropriate Level of Detail (LOD) for complex robot models
- Consider occlusion culling for complex scenes
- Optimize draw calls by combining materials where possible

## Testing the Scene

Once the scene is set up:

1. Enter Play mode to test basic functionality
2. Verify that the HumanoidController script properly controls the robot joints
3. Check that cameras provide appropriate views
4. Test lighting and material appearance
5. Ensure physics interactions work correctly

## Troubleshooting

### Robot Appears Upside Down
- Check the URDF import coordinate conversion settings
- Verify joint axes are aligned properly

### Joints Don't Move
- Verify ArticulationBody components are properly assigned in the HumanoidController script
- Check joint limits and drive settings on ArticulationBody components

### Lighting Issues
- Ensure the scene has proper lighting for PBR materials
- Check that reflection probes are appropriately placed if using reflective materials

This setup provides a complete humanoid robot visualization environment that integrates with ROS systems for real-time control and data exchange.