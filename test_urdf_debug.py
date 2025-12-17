#!/usr/bin/env python3
"""Debug script to check URDF parsing"""

import xml.etree.ElementTree as ET

def debug_urdf_parsing():
    urdf_path = "examples/ros2/urdf/humanoid.urdf"

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        print(f"Root tag: {root.tag}")
        print(f"Root name: {root.get('name')}")

        joints = root.findall('.//joint')
        print(f"Found {len(joints)} joints")

        for i, joint in enumerate(joints):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            print(f"Joint {i+1}: name='{joint_name}', type='{joint_type}'")

            if joint_name == 'left_shoulder_joint':
                print(f"  DEBUG: left_shoulder_joint attributes: {joint.attrib}")
                print(f"  DEBUG: left_shoulder_joint text: '{joint.text}'")
                print(f"  DEBUG: left_shoulder_joint children: {[child.tag for child in joint]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_urdf_parsing()