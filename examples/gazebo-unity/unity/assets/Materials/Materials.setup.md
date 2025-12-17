# Unity Materials Setup for Humanoid Robotics

This document provides instructions for creating and configuring materials for humanoid robot visualization in Unity.

## Material Configuration for Robot Parts

### Metallic Surfaces (Joints, Actuators, Metal Components)

#### Metal_Joints Material
- **Shader**: Standard
- **Albedo**: Dark Gray (RGB: 50, 50, 50)
- **Metallic**: 0.9
- **Smoothness**: 0.7
- **Normal Map**: Optional brushed metal normal map
- **Occlusion**: Default

#### Metal_Actuators Material
- **Shader**: Standard
- **Albedo**: Silver (RGB: 192, 192, 192)
- **Metallic**: 0.8
- **Smoothness**: 0.6
- **Normal Map**: Fine surface detail normal map
- **Occlusion**: Default

### Plastic Surfaces (Body Panels, Covers)

#### Plastic_White Material
- **Shader**: Standard
- **Albedo**: Pure White (RGB: 255, 255, 255)
- **Metallic**: 0.0
- **Smoothness**: 0.3
- **Normal Map**: Subtle plastic texture
- **Occlusion**: Default

#### Plastic_Black Material
- **Shader**: Standard
- **Albedo**: Pure Black (RGB: 25, 25, 25)
- **Metallic**: 0.0
- **Smoothness**: 0.2
- **Normal Map**: Subtle plastic texture
- **Occlusion**: Default

#### Plastic_Gray Material
- **Shader**: Standard
- **Albedo**: Medium Gray (RGB: 100, 100, 100)
- **Metallic**: 0.0
- **Smoothness**: 0.25
- **Normal Map**: Subtle plastic texture
- **Occlusion**: Default

### Rubber Surfaces (Feet, Grippers, Bumpers)

#### Rubber_Black Material
- **Shader**: Standard
- **Albedo**: Black (RGB: 20, 20, 20)
- **Metallic**: 0.0
- **Smoothness**: 0.1
- **Normal Map**: Subtle rubber texture
- **Occlusion**: Default

#### Rubber_Grip Material
- **Shader**: Standard
- **Albedo**: Dark Gray (RGB: 40, 40, 40)
- **Metallic**: 0.0
- **Smoothness**: 0.05
- **Normal Map**: Strong grip pattern normal map
- **Occlusion**: Default

### Sensor Materials

#### Camera_Lens Material
- **Shader**: Standard (Specular setup)
- **Albedo**: Dark Gray (RGB: 30, 30, 30)
- **Specular**: White (RGB: 200, 200, 200)
- **Smoothness**: 0.9
- **Normal Map**: Clean lens surface

#### LiDAR_Cover Material
- **Shader**: Standard
- **Albedo**: Translucent Gray (RGB: 50, 50, 50, A: 150)
- **Metallic**: 0.1
- **Smoothness**: 0.5
- **Rendering Mode**: Transparent

## Lighting Setup Configuration

### Directional Light (Main Sun/Sky)
- **Type**: Directional
- **Color**: (255, 255, 250) - Slightly warm white
- **Intensity**: 1.0
- **Shadows**: Cascaded
- **Shadow Strength**: 1.0
- **Shadow Resolution**: High
- **Shadow Distance**: 50
- **Shadow Near Plane**: 0.2

### Fill Lights (Indoor Environment)
- **Type**: Point or Spot
- **Color**: (240, 240, 255) - Cool white
- **Intensity**: 0.3-0.5
- **Range**: 10-20 units
- **Shadows**: Off or Soft
- **Baking**: Mixed (for performance)

### Accent Lights (Robot Detailing)
- **Type**: Spot
- **Color**: (255, 250, 240) - Warm white
- **Intensity**: 0.7
- **Spot Angle**: 45 degrees
- **Range**: 5 units
- **Shadows**: Soft

## Material Optimization for Performance

### Texture Settings
- **Resolution**: 1024x1024 for most robot parts
- **Format**: ASTC (Android) / DXT (PC) / Metal (iOS)
- **Mip Maps**: Enabled
- **Compression**: High quality
- **Aniso Level**: 1-4 (depending on importance)

### Shader Considerations
- Use Standard shader for most surfaces
- Consider mobile-friendly shaders for performance
- Use texture atlasing to reduce draw calls
- Implement Level of Detail (LOD) for materials

## Environment Materials

### Floor Material (Laboratory)
- **Shader**: Standard
- **Albedo**: Light Gray (RGB: 200, 200, 205)
- **Metallic**: 0.0
- **Smoothness**: 0.2
- **Normal Map**: Subtle concrete texture
- **Tiling**: (4, 4) for large areas

### Wall Material
- **Shader**: Standard
- **Albedo**: White (RGB: 240, 240, 240)
- **Metallic**: 0.0
- **Smoothness**: 0.1
- **Normal Map**: Subtle wall texture
- **Tiling**: (2, 2)

## Material Creation Process

### Step 1: Create Base Materials
1. Right-click in Project window → Create → Material
2. Name the material appropriately
3. Select Standard shader
4. Configure base properties (Albedo, Metallic, Smoothness)

### Step 2: Assign Textures
1. Click the circle next to the property to assign a texture
2. Select your texture asset
3. Adjust tiling and offset as needed

### Step 3: Fine-tune Appearance
1. Use the Scene view to preview materials under different lighting
2. Adjust properties to achieve desired appearance
3. Test on different robot parts to ensure consistency

### Step 4: Apply to Robot Parts
1. Select the robot part in the Scene or Prefab
2. In the Inspector, drag the material to the Mesh Renderer's material slot
3. Verify the appearance looks correct

## Quality Settings for Different Platforms

### High Quality (Development)
- **Shadows**: All shadow types enabled
- **Anti-aliasing**: 4x MSAA
- **Texture Quality**: Full size
- **Anisotropic Filtering**: 4x

### Medium Quality (Testing)
- **Shadows**: Cascaded shadows on directional lights only
- **Anti-aliasing**: FXAA
- **Texture Quality**: Half size
- **Anisotropic Filtering**: 2x

### Low Quality (Performance Critical)
- **Shadows**: Off or minimal
- **Anti-aliasing**: Off
- **Texture Quality**: Quarter size
- **Anisotropic Filtering**: Off

## Best Practices

1. **Reuse Materials**: Use the same material for similar surfaces to reduce draw calls
2. **Texture Atlasing**: Combine multiple small textures into larger atlases
3. **LOD Materials**: Create simplified materials for distant robots
4. **Consistent Look**: Maintain visual consistency across all robot parts
5. **Performance Testing**: Regularly profile material performance
6. **Version Control**: Keep material configurations documented

## Troubleshooting Common Issues

### Materials Appear Too Dark
- Check lighting setup in the scene
- Verify material Metallic and Smoothness values
- Ensure textures are properly assigned

### Materials Appear Too Shiny
- Reduce Smoothness value
- Lower Metallic value for non-metallic surfaces
- Check for proper texture assignment

### Performance Issues
- Reduce texture resolution
- Simplify shader complexity
- Use fewer materials overall

This material setup provides a realistic and performant visualization system for humanoid robots in Unity, with appropriate materials for different robot components and optimized lighting configurations.