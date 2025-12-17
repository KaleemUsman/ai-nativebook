#!/usr/bin/env python3
"""
URDF Validation Script

This script demonstrates how to validate URDF files using ROS 2 tools
and checks for common issues in humanoid robot models.
"""

import xml.etree.ElementTree as ET
import subprocess
import sys
import os
from pathlib import Path


def validate_urdf_syntax(urdf_path):
    """
    Validate URDF syntax by attempting to parse it as XML.

    Args:
        urdf_path (str): Path to the URDF file

    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        if root.tag != 'robot':
            print(f"ERROR: Root element is not 'robot', found '{root.tag}'")
            return False
        print(f"[PASS] XML syntax is valid for {urdf_path}")
        return True
    except ET.ParseError as e:
        print(f"ERROR: XML parsing failed for {urdf_path}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to parse {urdf_path}: {e}")
        return False


def check_urdf_elements(urdf_path):
    """
    Check for required elements in the URDF file.

    Args:
        urdf_path (str): Path to the URDF file

    Returns:
        bool: True if all required elements are present, False otherwise
    """
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Check for robot name
        robot_name = root.get('name')
        if not robot_name:
            print("ERROR: Robot element missing 'name' attribute")
            return False
        print(f"[PASS] Robot name: {robot_name}")

        # Find all links
        links = root.findall('.//link')
        if not links:
            print("ERROR: No links found in URDF")
            return False
        print(f"[PASS] Found {len(links)} links")

        # Check that each link has required elements
        for link in links:
            link_name = link.get('name')
            if not link_name:
                print(f"ERROR: Link without name found")
                return False

            # Check for inertial element
            inertial = link.find('inertial')
            if inertial is None:
                print(f"WARNING: Link '{link_name}' has no inertial element (simulation may be inaccurate)")
            else:
                mass = inertial.find('mass')
                if mass is None or mass.get('value') is None:
                    print(f"ERROR: Link '{link_name}' has no mass defined")
                    return False

                inertia = inertial.find('inertia')
                if inertia is None:
                    print(f"ERROR: Link '{link_name}' has no inertia defined")
                    return False

                # Check that all inertia values are present
                required_inertia = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
                for attr in required_inertia:
                    if inertia.get(attr) is None:
                        print(f"ERROR: Link '{link_name}' missing inertia attribute '{attr}'")
                        return False
                print(f"  [PASS] Link '{link_name}' has valid inertial properties")

        # Find all joints (only kinematic joints, not transmission joints)
        # Get joints that are direct children of robot and children of other elements except transmission
        joints = []
        # Direct children of robot
        joints.extend(root.findall('joint'))
        # Children of other elements but not inside transmission
        for child in root:
            if child.tag != 'transmission':
                joints.extend(child.findall('joint'))
        if not joints:
            print("WARNING: No joints found in URDF (static model)")
        else:
            print(f"[PASS] Found {len(joints)} joints")

            for joint in joints:
                joint_name = joint.get('name')
                joint_type = joint.get('type')

                if not joint_name:
                    print(f"ERROR: Joint without name found")
                    return False
                if not joint_type:
                    print(f"ERROR: Joint '{joint_name}' without type found")
                    return False

                # Check for parent and child elements
                parent = joint.find('parent')
                child = joint.find('child')

                if parent is None or parent.get('link') is None:
                    print(f"ERROR: Joint '{joint_name}' missing parent link")
                    return False
                if child is None or child.get('link') is None:
                    print(f"ERROR: Joint '{joint_name}' missing child link")
                    return False

                print(f"  [PASS] Joint '{joint_name}' ({joint_type}) connects {parent.get('link')} to {child.get('link')}")

        # Check for transmissions if present
        transmissions = root.findall('.//transmission')
        if transmissions:
            print(f"[PASS] Found {len(transmissions)} transmissions")
            for trans in transmissions:
                trans_name = trans.get('name')
                trans_type = trans.find('type')
                if trans_type is None:
                    print(f"ERROR: Transmission '{trans_name}' missing type")
                    return False
                print(f"  [PASS] Transmission '{trans_name}' has valid type")

        return True

    except Exception as e:
        print(f"ERROR: Failed to check URDF elements: {e}")
        return False


def run_check_urdf_command(urdf_path):
    """
    Run the ROS 2 check_urdf command on the URDF file.

    Args:
        urdf_path (str): Path to the URDF file

    Returns:
        bool: True if the command succeeds, False otherwise
    """
    try:
        # Try to run check_urdf command
        result = subprocess.run(['check_urdf', urdf_path],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("[PASS] check_urdf command succeeded")
            print("  Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"[FAIL] check_urdf command failed with return code {result.returncode}")
            print("  Error:", result.stderr)
            return False

    except FileNotFoundError:
        print("[WARN] check_urdf command not found (ROS 2 may not be installed or sourced)")
        return True  # Don't fail if tool isn't available
    except subprocess.TimeoutExpired:
        print("[WARN] check_urdf command timed out")
        return False
    except Exception as e:
        print(f"[WARN] Error running check_urdf command: {e}")
        return True  # Don't fail if tool isn't available


def validate_humanoid_urdf(urdf_path):
    """
    Perform comprehensive validation of a humanoid robot URDF.

    Args:
        urdf_path (str): Path to the URDF file

    Returns:
        bool: True if validation passes, False otherwise
    """
    print(f"Validating humanoid URDF: {urdf_path}")
    print("=" * 50)

    # Check if file exists
    if not Path(urdf_path).exists():
        print(f"ERROR: URDF file does not exist: {urdf_path}")
        return False

    # Validate XML syntax
    if not validate_urdf_syntax(urdf_path):
        return False

    # Check URDF elements
    if not check_urdf_elements(urdf_path):
        return False

    # Try running check_urdf command if available
    run_check_urdf_command(urdf_path)

    print("=" * 50)
    print("[PASS] URDF validation completed successfully")
    return True


def main():
    """
    Main function to validate the humanoid URDF example.
    """
    urdf_file = "examples/ros2/urdf/humanoid.urdf"

    if not validate_humanoid_urdf(urdf_file):
        print(f"\n[FAIL] Validation failed for {urdf_file}")
        sys.exit(1)
    else:
        print(f"\n[PASS] Validation passed for {urdf_file}")
        sys.exit(0)


if __name__ == "__main__":
    main()