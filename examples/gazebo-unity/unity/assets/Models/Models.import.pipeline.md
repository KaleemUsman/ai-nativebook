# Unity Asset Import Pipeline for Humanoid Robot Models

This document provides instructions for setting up an efficient asset import pipeline for humanoid robot models in Unity, focusing on optimization for real-time robotics visualization.

## Recommended Asset Structure

```
Assets/
├── Models/
│   ├── HumanoidRobot/
│   │   ├── Meshes/
│   │   │   ├── Visual/
│   │   │   │   ├── Head.fbx
│   │   │   │   ├── Torso.fbx
│   │   │   │   ├── Arm_Left.fbx
│   │   │   │   ├── Arm_Right.fbx
│   │   │   │   ├── Leg_Left.fbx
│   │   │   │   ├── Leg_Right.fbx
│   │   │   │   └── Feet_Left.fbx
│   │   │   │   └── Feet_Right.fbx
│   │   │   └── Collision/
│   │   │       ├── Head_collision.fbx
│   │   │       ├── Torso_collision.fbx
│   │   │       └── ...
│   │   ├── Materials/
│   │   │   ├── Plastic_Black.mat
│   │   │   ├── Plastic_White.mat
│   │   │   ├── Metal_Joints.mat
│   │   │   └── Rubber_Feet.mat
│   │   ├── Prefabs/
│   │   │   └── HumanoidRobot.prefab
│   │   └── Textures/
│   │       ├── Robot_Diffuse.png
│   │       ├── Robot_Normal.png
│   │       └── Robot_Metallic.png
│   └── Environment/
│       ├── Floor.fbx
│       ├── Walls.fbx
│       └── Obstacles.fbx
```

## Import Pipeline Configuration

### 1. Mesh Import Settings

For each robot part mesh:

#### Visual Meshes
- **File Type**: FBX, OBJ, or DAE
- **Scale Factor**: 1 (ensure URDF units are in meters)
- **Mesh Compression**: Off (for robotics accuracy)
- **Read/Write Enabled**: True (for dynamic modification)
- **Optimize Mesh**: Checked
- **Import Visibility**: False
- **Import Cameras**: False
- **Import Lights**: False
- **Import Animations**: False (for static robot parts)
- **Generate Colliders**: False (separate collision meshes)

#### Collision Meshes
- **File Type**: FBX, OBJ, or simple primitives
- **Scale Factor**: 1 (ensure URDF units are in meters)
- **Mesh Compression**: Off
- **Read/Write Enabled**: False (for performance)
- **Optimize Mesh**: Checked
- **Import Visibility**: False
- **Generate Colliders**: True (or manually create)

### 2. Model Import Optimization

#### For High-Detail Robot Parts
1. **LOD Generation**:
   - Create 3-4 LOD levels
   - Use 50%, 25%, 12.5%, 6.25% reduction
   - Set appropriate screen percentages (50%, 25%, 12%, 5%)

2. **Mesh Simplification**:
   - Use Unity's built-in mesh simplification
   - Or use external tools like Simplygon
   - Preserve important details for robotic components

#### For Performance-Critical Applications
1. **Polygon Count Limits**:
   - Head: < 5,000 triangles
   - Torso: < 8,000 triangles
   - Arms: < 4,000 triangles each
   - Legs: < 4,000 triangles each
   - Feet: < 2,000 triangles each

2. **Texture Atlas**:
   - Combine multiple small textures into larger atlases
   - Aim for 2048x2048 maximum size
   - Use appropriate padding (4-8 pixels)

### 3. Material Assignment Pipeline

#### Automatic Material Assignment
1. Create a naming convention that matches URDF material names
2. Use Unity's Material Switch component for runtime switching
3. Implement a Material Manager script for centralized control

#### Material Property Optimization
1. **Metallic Smoothness Workflow**:
   - Metallic: 0.0-0.2 for plastics
   - Metallic: 0.7-1.0 for metals
   - Smoothness: 0.2-0.8 depending on surface finish

2. **Texture Setup**:
   - Use Linear color space for PBR materials
   - Set appropriate texture import types
   - Compress textures appropriately

### 4. Articulation Body Setup

For each robot joint:

1. **Select Appropriate Joint Type**:
   - `JointType.Revolute` for rotational joints
   - `JointType.Prismatic` for linear joints
   - `JointType.Fixed` for static connections

2. **Configure Joint Limits**:
   - Set `xDrive`, `yDrive`, `zDrive` limits based on URDF specifications
   - Configure `lowerLimit` and `upperLimit` appropriately
   - Set `forceLimit` and `damping` values from URDF

3. **Set Mass Properties**:
   - Configure `mass` based on URDF inertial properties
   - Set `centerOfMass` appropriately
   - Configure `inertiaTensor` if needed

### 5. Robot Assembly Pipeline

#### Creating the Robot Prefab
1. **Hierarchy Structure**:
   ```
   HumanoidRobot (Prefab Root)
   ├── BaseLink
   ├── Head
   ├── Torso
   ├── LeftArm
   │   ├── Shoulder
   │   ├── Elbow
   │   └── Wrist
   ├── RightArm
   │   ├── Shoulder
   │   ├── Elbow
   │   └── Wrist
   ├── LeftLeg
   │   ├── Hip
   │   ├── Knee
   │   └── Ankle
   └── RightLeg
       ├── Hip
       ├── Knee
       └── Ankle
   ```

2. **Component Setup**:
   - Add `HumanoidController` script to root
   - Add `ArticulationBody` components to each joint
   - Add colliders where appropriate
   - Configure joint mappings in the controller

#### Automated Assembly Script
Create an assembly script to automatically configure the robot:

```csharp
using UnityEngine;

public class RobotAssembly : MonoBehaviour
{
    [System.Serializable]
    public class JointConfig
    {
        public string jointName;
        public Transform jointTransform;
        public HumanoidController.JointType jointType;
        public float minAngle;
        public float maxAngle;
    }

    public JointConfig[] jointConfigs;

    void Start()
    {
        // Automatically configure the HumanoidController with joint mappings
        HumanoidController controller = GetComponent<HumanoidController>();
        if (controller != null)
        {
            controller.jointMappings = new HumanoidController.JointMapping[jointConfigs.Length];

            for (int i = 0; i < jointConfigs.Length; i++)
            {
                var mapping = new HumanoidController.JointMapping();
                mapping.jointName = jointConfigs[i].jointName;
                mapping.jointArticulationBody = jointConfigs[i].jointTransform.GetComponent<ArticulationBody>();
                mapping.jointType = jointConfigs[i].jointType;
                mapping.minAngle = jointConfigs[i].minAngle;
                mapping.maxAngle = jointConfigs[i].maxAngle;

                controller.jointMappings[i] = mapping;
            }
        }
    }
}
```

### 6. Quality Assurance Pipeline

#### Pre-Import Checks
1. **URDF Validation**:
   - Use `check_urdf` to validate URDF files
   - Verify joint limits and types
   - Check for proper mesh file references

2. **Mesh Quality**:
   - Validate mesh topology (no non-manifold edges)
   - Check for proper normals
   - Ensure appropriate scale (meters)

#### Post-Import Validation
1. **Kinematic Chain Verification**:
   - Verify joint hierarchy matches URDF
   - Check joint limits are properly configured
   - Test range of motion

2. **Visual Verification**:
   - Ensure materials are properly applied
   - Check for texture seams or artifacts
   - Verify proper lighting response

### 7. Performance Optimization Pipeline

#### Draw Call Reduction
1. **Static Batching**:
   - Mark static environment objects as Static
   - Combine static meshes where possible

2. **Dynamic Batching**:
   - Use similar materials for multiple instances
   - Keep mesh vertex count under 900

#### Memory Optimization
1. **Texture Streaming**:
   - Enable texture streaming in Quality Settings
   - Use appropriate texture compression
   - Implement LOD for textures

2. **Mesh Optimization**:
   - Use Level of Detail (LOD) groups
   - Implement occlusion culling
   - Use object pooling for multiple robots

### 8. Automated Import Process

#### Using AssetPostprocessor
Create an AssetPostprocessor to automatically configure robot assets:

```csharp
using UnityEngine;
using UnityEditor;

public class RobotModelPostprocessor : AssetPostprocessor
{
    void OnPostprocessModel(GameObject gameObject)
    {
        // Automatically configure ArticulationBodies if the model has joint-like names
        ArticulationBody[] articulationBodies = gameObject.GetComponentsInChildren<ArticulationBody>();

        foreach (ArticulationBody body in articulationBodies)
        {
            // Set default values for robotics applications
            body.mass = body.mass == 0 ? 1.0f : body.mass; // Set default mass if zero
            body.linearDamping = 0.05f;
            body.angularDamping = 0.05f;
        }
    }

    void OnPostprocessTexture(Texture2D texture)
    {
        // Automatically configure textures for robotics visualization
        TextureImporter importer = (TextureImporter)assetImporter;
        importer.sRGBTexture = !importer.name.Contains("Metallic") &&
                              !importer.name.Contains("Normal");
    }
}
```

### 9. Version Control for Assets

#### Git LFS Configuration
For large model and texture files:
1. Install Git LFS
2. Track large files: `git lfs track "*.fbx" "*.dae" "*.obj" "*.png" "*.jpg"`
3. Add .gitattributes file to repository
4. Commit and push large files with LFS

#### Asset Bundles for Distribution
For distributing robot models:
1. Create asset bundles for each robot model
2. Implement asset bundle loading system
3. Handle versioning and updates

This import pipeline ensures that humanoid robot models are properly configured for high-fidelity visualization in Unity while maintaining good performance characteristics for real-time robotics applications.