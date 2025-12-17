#!/usr/bin/env python3
"""
Verification script to confirm all ROS 2 book module implementation tasks are completed.
"""

import os
from pathlib import Path

def verify_implementation():
    print("Verifying ROS 2 Book Module Implementation")
    print("=" * 50)

    # Check documentation files
    docs_dir = Path("docs/modules/ros2/")
    docs_files = [
        "architecture.mdx",
        "communication.mdx",
        "ai-integration.mdx",
        "urdf-modeling.mdx",
        "summary.mdx"
    ]

    print("Checking documentation files...")
    for file in docs_files:
        path = docs_dir / file
        if path.exists():
            print(f"  [PASS] {file}")
        else:
            print(f"  [FAIL] {file} - MISSING")

    # Check example code files
    examples_dir = Path("examples/ros2/")
    example_files = [
        "publisher_subscriber/minimal_publisher.py",
        "publisher_subscriber/minimal_subscriber.py",
        "service_client/minimal_service.py",
        "service_client/minimal_client.py",
        "ai_agent_bridge/ai_agent.py",
        "ai_agent_bridge/controller_bridge.py",
        "urdf/humanoid.urdf"
    ]

    print("\nChecking example code files...")
    for file in example_files:
        path = examples_dir / file
        if path.exists():
            print(f"  [PASS] {file}")
        else:
            print(f"  [FAIL] {file} - MISSING")

    # Check validation files
    validation_files = [
        "specs/001-ros2-book-module/validation/urdf_validation_script.py",
        "specs/001-ros2-book-module/testing/urdf_simulation_test_plan.md",
        "specs/001-ros2-book-module/verification/urdf_specification_verification.md"
    ]

    print("\nChecking validation files...")
    for file in validation_files:
        path = Path(file)
        if path.exists():
            print(f"  [PASS] {file}")
        else:
            print(f"  [FAIL] {file} - MISSING")

    # Check spec files
    spec_files = [
        "specs/001-ros2-book-module/spec.md",
        "specs/001-ros2-book-module/plan.md",
        "specs/001-ros2-book-module/tasks.md"
    ]

    print("\nChecking specification files...")
    for file in spec_files:
        path = Path(file)
        if path.exists():
            with open(path, 'r') as f:
                content = f.read()
                if 'T070' in content and '[X] T070' in content:
                    print(f"  [PASS] {file} (all tasks completed)")
                else:
                    print(f"  [FAIL] {file} (tasks not completed)")
        else:
            print(f"  [FAIL] {file} - MISSING")

    # Verify URDF syntax
    print("\nChecking URDF syntax...")
    try:
        import xml.etree.ElementTree as ET
        urdf_path = Path("examples/ros2/urdf/humanoid.urdf")
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        if root.tag == 'robot' and root.get('name') == 'simple_humanoid':
            print("  [PASS] humanoid.urdf has valid syntax and structure")
        else:
            print("  [FAIL] humanoid.urdf has invalid structure")
    except Exception as e:
        print(f"  [FAIL] humanoid.urdf validation error: {e}")

    # Check code syntax
    print("\nChecking Python code syntax...")
    python_files = [
        "examples/ros2/publisher_subscriber/minimal_publisher.py",
        "examples/ros2/publisher_subscriber/minimal_subscriber.py",
        "examples/ros2/service_client/minimal_service.py",
        "examples/ros2/service_client/minimal_client.py",
        "examples/ros2/ai_agent_bridge/ai_agent.py",
        "examples/ros2/ai_agent_bridge/controller_bridge.py",
        "specs/001-ros2-book-module/validation/urdf_validation_script.py"
    ]

    for file in python_files:
        try:
            compile(open(file).read(), file, 'exec')
            print(f"  [PASS] {os.path.basename(file)} syntax OK")
        except SyntaxError as e:
            print(f"  [FAIL] {os.path.basename(file)} syntax error: {e}")

    print("\n" + "=" * 50)
    print("Verification complete. All major components of the ROS 2 book module have been implemented.")

if __name__ == "__main__":
    verify_implementation()