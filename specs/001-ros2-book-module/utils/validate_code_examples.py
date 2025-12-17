#!/usr/bin/env python3
"""
Utility script to validate code examples in the ROS 2 book module
"""

import ast
import os
import sys
from pathlib import Path


def validate_python_syntax(file_path):
    """
    Validate Python syntax of a file

    Args:
        file_path (str): Path to the Python file to validate

    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def validate_rclpy_imports(file_path):
    """
    Check if a Python file has proper rclpy imports

    Args:
        file_path (str): Path to the Python file to check

    Returns:
        bool: True if rclpy imports are present and correct, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Parse the AST to check imports
        tree = ast.parse(content)

        has_rclpy = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('rclpy'):
                        has_rclpy = True
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('rclpy'):
                    has_rclpy = True

        return has_rclpy
    except Exception as e:
        print(f"Error checking imports in {file_path}: {e}")
        return False


def validate_ros2_example_structure(file_path):
    """
    Validate that a ROS 2 example has proper structure (init, node creation, spin, shutdown)

    Args:
        file_path (str): Path to the Python file to check

    Returns:
        bool: True if structure is valid, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        required_patterns = [
            'rclpy.init',
            'Node',
            'rclpy.spin',
            'rclpy.shutdown'
        ]

        for pattern in required_patterns:
            if pattern not in content:
                print(f"Missing required pattern '{pattern}' in {file_path}")
                return False

        return True
    except Exception as e:
        print(f"Error checking structure in {file_path}: {e}")
        return False


def validate_all_examples():
    """
    Validate all Python examples in the examples/ros2 directory

    Returns:
        bool: True if all examples pass validation, False otherwise
    """
    examples_dir = Path("examples/ros2")
    if not examples_dir.exists():
        print(f"Examples directory {examples_dir} does not exist")
        return False

    all_passed = True
    python_files = list(examples_dir.rglob("*.py"))

    print(f"Validating {len(python_files)} Python examples...")

    for py_file in python_files:
        if py_file.name != "validate_code_examples.py":  # Skip this validation script
            print(f"Validating {py_file}...")

            # Check syntax
            if not validate_python_syntax(str(py_file)):
                print(f"  ✗ Syntax validation failed for {py_file}")
                all_passed = False
                continue

            # Check rclpy imports
            if not validate_rclpy_imports(str(py_file)):
                print(f"  ✗ rclpy import validation failed for {py_file}")
                all_passed = False
                continue

            # Check ROS 2 structure
            if not validate_ros2_example_structure(str(py_file)):
                print(f"  ✗ ROS 2 structure validation failed for {py_file}")
                all_passed = False
                continue

            print(f"  ✓ {py_file} passed all validations")

    return all_passed


def validate_mdx_content():
    """
    Validate MDX content for proper structure and formatting

    Returns:
        bool: True if all MDX files pass validation, False otherwise
    """
    docs_dir = Path("docs/modules/ros2")
    if not docs_dir.exists():
        print(f"Documentation directory {docs_dir} does not exist")
        return False

    all_passed = True
    mdx_files = list(docs_dir.rglob("*.mdx"))

    print(f"Validating {len(mdx_files)} MDX files...")

    for mdx_file in mdx_files:
        print(f"Validating {mdx_file}...")

        try:
            with open(mdx_file, 'r', encoding='utf-8') as file:
                content = file.read()

            # Check for proper frontmatter
            if not content.startswith('---'):
                print(f"  ✗ Missing frontmatter in {mdx_file}")
                all_passed = False
                continue

            # Find end of frontmatter
            lines = content.split('\n')
            frontmatter_end = -1
            for i, line in enumerate(lines[1:], 1):  # Skip first '---'
                if line.strip() == '---' and i > 0:
                    frontmatter_end = i
                    break

            if frontmatter_end == -1:
                print(f"  ✗ Malformed frontmatter in {mdx_file}")
                all_passed = False
                continue

            print(f"  ✓ {mdx_file} passed MDX validation")

        except Exception as e:
            print(f"  ✗ Error validating {mdx_file}: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("Starting code example validation...")

    examples_valid = validate_all_examples()
    print()

    mdx_valid = validate_mdx_content()
    print()

    if examples_valid and mdx_valid:
        print("✓ All validations passed!")
        sys.exit(0)
    else:
        print("✗ Some validations failed!")
        sys.exit(1)