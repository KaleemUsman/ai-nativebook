#!/usr/bin/env python3
"""
Sensor Validation Test Script
This script validates sensor readings from Gazebo simulation against expected values
in various simulated scenarios to ensure accuracy and reliability.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String, Float64MultiArray
from nav_msgs.msg import Odometry
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import threading
from collections import deque
import statistics
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json


@dataclass
class ValidationResult:
    """Result of a sensor validation test"""
    sensor_type: str
    test_name: str
    passed: bool
    expected_value: Any
    actual_value: Any
    tolerance: float
    error_message: str = ""


@dataclass
class TestScenario:
    """Definition of a test scenario"""
    name: str
    description: str
    expected_values: Dict[str, Any]
    validation_function: callable


class SensorValidator(Node):
    """
    Validates sensor readings from Gazebo simulation against expected values
    in various simulated scenarios.
    """

    def __init__(self):
        super().__init__('sensor_validator')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=10),
            'imu': deque(maxlen=10),
            'joint': deque(maxlen=10),
            'camera': deque(maxlen=5),
            'odometry': deque(maxlen=10)
        }

        self.validation_results = []
        self.current_scenario = None
        self.test_results = []

        # Initialize ROS interfaces
        self.bridge = CvBridge()

        # Subscribers for sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid/imu/data',
            self.imu_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/humanoid/camera/color/image_raw',
            self.camera_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        # Publishers for test control and results
        self.test_control_pub = self.create_publisher(
            String,
            '/sensor_validator/test_control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/sensor_validator/test_results',
            10
        )

        # Timer for validation tests
        self.validation_timer = self.create_timer(0.1, self.run_validation_cycle)

        # Define test scenarios
        self.define_test_scenarios()

        # Statistics for validation
        self.stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'validation_cycles': 0
        }

        self.get_logger().info("Sensor Validator initialized and ready for testing")

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.sensor_data['laser'].append(msg)

    def imu_callback(self, msg):
        """Process IMU data"""
        self.sensor_data['imu'].append(msg)

    def joint_callback(self, msg):
        """Process joint state data"""
        self.sensor_data['joint'].append(msg)

    def camera_callback(self, msg):
        """Process camera image data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.sensor_data['camera'].append(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error processing camera data: {e}")

    def odom_callback(self, msg):
        """Process odometry data"""
        self.sensor_data['odometry'].append(msg)

    def define_test_scenarios(self):
        """Define various test scenarios for sensor validation"""
        self.test_scenarios = [
            TestScenario(
                name="empty_room",
                description="Robot in empty room - LiDAR should show maximum range in most directions",
                expected_values={
                    'laser_min_range': 9.5,  # Should be close to max range of 10m
                    'laser_avg_range': 9.0,
                    'imu_orientation_w': 1.0  # No rotation expected
                },
                validation_function=self.validate_empty_room_scenario
            ),
            TestScenario(
                name="wall_front",
                description="Robot facing wall 1m ahead - LiDAR should detect wall in front",
                expected_values={
                    'laser_front_range': 1.0,
                    'laser_min_range': 0.8  # Allow some tolerance
                },
                validation_function=self.validate_wall_front_scenario
            ),
            TestScenario(
                name="obstacle_avoidance",
                description="Robot navigating around obstacle - sensors should detect and respond",
                expected_values={
                    'laser_left_range': 2.0,
                    'laser_right_range': 0.5
                },
                validation_function=self.validate_obstacle_scenario
            ),
            TestScenario(
                name="balance_test",
                description="Robot maintaining balance - IMU should show stable orientation",
                expected_values={
                    'imu_roll_stable': True,
                    'imu_pitch_stable': True
                },
                validation_function=self.validate_balance_scenario
            )
        ]

    def run_validation_cycle(self):
        """Run a cycle of validation tests"""
        try:
            if len(self.sensor_data['laser']) > 0 and len(self.sensor_data['imu']) > 0:
                # Run all validation tests
                for scenario in self.test_scenarios:
                    result = scenario.validation_function()
                    if result:
                        self.test_results.append(result)
                        self.stats['total_tests'] += 1
                        if result.passed:
                            self.stats['passed_tests'] += 1
                        else:
                            self.stats['failed_tests'] += 1

                # Update statistics
                self.stats['validation_cycles'] += 1

                # Log periodic summary
                if self.stats['validation_cycles'] % 50 == 0:
                    self.log_validation_summary()

        except Exception as e:
            self.get_logger().error(f"Error in validation cycle: {e}")

    def validate_empty_room_scenario(self) -> Optional[ValidationResult]:
        """Validate sensor readings in empty room scenario"""
        if len(self.sensor_data['laser']) == 0:
            return None

        latest_scan = self.sensor_data['laser'][-1]
        ranges = np.array(latest_scan.ranges)

        # Filter out invalid ranges
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) == 0:
            return ValidationResult(
                sensor_type='laser',
                test_name='empty_room_min_range',
                passed=False,
                expected_value=9.5,
                actual_value=0.0,
                tolerance=0.5,
                error_message="No valid laser ranges found"
            )

        min_range = np.min(valid_ranges)
        avg_range = np.mean(valid_ranges)

        # Check if minimum range is within expected bounds (empty room should have long ranges)
        passed = min_range >= 9.0  # Expect most ranges to be long in empty room

        return ValidationResult(
            sensor_type='laser',
            test_name='empty_room_min_range',
            passed=passed,
            expected_value=9.5,
            actual_value=min_range,
            tolerance=0.5,
            error_message="" if passed else f"Expected min range >= 9.0, got {min_range:.2f}"
        )

    def validate_wall_front_scenario(self) -> Optional[ValidationResult]:
        """Validate sensor readings when facing a wall"""
        if len(self.sensor_data['laser']) == 0:
            return None

        latest_scan = self.sensor_data['laser'][-1]
        ranges = np.array(latest_scan.ranges)

        # Get front-facing ranges (approx 60 degrees in front)
        front_ranges = ranges[150:210]
        valid_front_ranges = front_ranges[np.isfinite(front_ranges)]

        if len(valid_front_ranges) == 0:
            return ValidationResult(
                sensor_type='laser',
                test_name='wall_front_detection',
                passed=False,
                expected_value=1.0,
                actual_value=0.0,
                tolerance=0.5,
                error_message="No valid front-facing laser ranges found"
            )

        # Get the minimum front range
        min_front_range = np.min(valid_front_ranges)

        # Check if there's an obstacle in front (wall at ~1m)
        passed = 0.5 <= min_front_range <= 1.5  # Expect wall between 0.5 and 1.5m

        return ValidationResult(
            sensor_type='laser',
            test_name='wall_front_detection',
            passed=passed,
            expected_value=1.0,
            actual_value=min_front_range,
            tolerance=0.5,
            error_message="" if passed else f"Expected front range ~1.0m, got {min_front_range:.2f}m"
        )

    def validate_obstacle_scenario(self) -> Optional[ValidationResult]:
        """Validate sensor readings in obstacle avoidance scenario"""
        if len(self.sensor_data['laser']) == 0:
            return None

        latest_scan = self.sensor_data['laser'][-1]
        ranges = np.array(latest_scan.ranges)

        # Get left and right side ranges
        left_ranges = ranges[80:120]  # Left side
        right_ranges = ranges[280:320]  # Right side

        valid_left = left_ranges[np.isfinite(left_ranges)]
        valid_right = right_ranges[np.isfinite(right_ranges)]

        if len(valid_left) == 0 or len(valid_right) == 0:
            return ValidationResult(
                sensor_type='laser',
                test_name='obstacle_differentiation',
                passed=False,
                expected_value={'left': 2.0, 'right': 0.5},
                actual_value={'left': 0.0, 'right': 0.0},
                tolerance=0.5,
                error_message="No valid left or right laser ranges found"
            )

        avg_left = np.mean(valid_left) if len(valid_left) > 0 else 0.0
        avg_right = np.mean(valid_right) if len(valid_right) > 0 else 0.0

        # Check if left side is clearer than right (obstacle on right)
        passed = avg_left > avg_right + 1.0  # Left should be significantly clearer

        return ValidationResult(
            sensor_type='laser',
            test_name='obstacle_differentiation',
            passed=passed,
            expected_value={'left': 2.0, 'right': 0.5},
            actual_value={'left': avg_left, 'right': avg_right},
            tolerance=0.5,
            error_message="" if passed else f"Expected left ({avg_left:.2f}) > right ({avg_right:.2f}) by 1m+"
        )

    def validate_balance_scenario(self) -> Optional[ValidationResult]:
        """Validate IMU readings for balance"""
        if len(self.sensor_data['imu']) == 0:
            return None

        latest_imu = self.sensor_data['imu'][-1]

        # Extract orientation
        quat = np.array([
            latest_imu.orientation.x,
            latest_imu.orientation.y,
            latest_imu.orientation.z,
            latest_imu.orientation.w
        ])

        # Convert to Euler angles to check stability
        roll, pitch, yaw = self.quaternion_to_euler(quat)

        # Check if robot is reasonably upright (within 10 degrees)
        max_stable_angle = 0.1745  # ~10 degrees in radians
        roll_stable = abs(roll) <= max_stable_angle
        pitch_stable = abs(pitch) <= max_stable_angle

        passed = roll_stable and pitch_stable

        return ValidationResult(
            sensor_type='imu',
            test_name='balance_stability',
            passed=passed,
            expected_value={'roll_stable': True, 'pitch_stable': True},
            actual_value={'roll': roll, 'pitch': pitch},
            tolerance=max_stable_angle,
            error_message="" if passed else f"Robot tilted: roll={roll:.3f}, pitch={pitch:.3f}"
        )

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def validate_joint_ranges(self) -> List[ValidationResult]:
        """Validate that joint positions are within expected ranges"""
        results = []

        if len(self.sensor_data['joint']) == 0:
            return results

        latest_joints = self.sensor_data['joint'][-1]

        # Define expected ranges for humanoid joints (example values)
        joint_limits = {
            'head_pan': (-1.57, 1.57),      # -90 to 90 degrees
            'head_tilt': (-0.785, 0.785),   # -45 to 45 degrees
            'shoulder_left': (-2.0, 2.0),   # -114 to 114 degrees
            'elbow_left': (0.0, 2.5),       # 0 to 143 degrees
            'shoulder_right': (-2.0, 2.0),  # -114 to 114 degrees
            'elbow_right': (0.0, 2.5),      # 0 to 143 degrees
            'hip_left': (-1.57, 0.785),     # -90 to 45 degrees
            'knee_left': (0.0, 2.356),      # 0 to 135 degrees
            'ankle_left': (-0.785, 0.785),  # -45 to 45 degrees
            'hip_right': (-1.57, 0.785),    # -90 to 45 degrees
            'knee_right': (0.0, 2.356),     # 0 to 135 degrees
            'ankle_right': (-0.785, 0.785)  # -45 to 45 degrees
        }

        for i, joint_name in enumerate(latest_joints.name):
            if i < len(latest_joints.position):
                position = latest_joints.position[i]

                if joint_name in joint_limits:
                    min_limit, max_limit = joint_limits[joint_name]
                    passed = min_limit <= position <= max_limit

                    results.append(ValidationResult(
                        sensor_type='joint',
                        test_name=f'joint_range_{joint_name}',
                        passed=passed,
                        expected_value={'min': min_limit, 'max': max_limit},
                        actual_value=position,
                        tolerance=0.0,
                        error_message="" if passed else f"Joint {joint_name} out of range: {position:.3f} (limits: {min_limit:.3f} to {max_limit:.3f})"
                    ))

        return results

    def validate_sensor_consistency(self) -> List[ValidationResult]:
        """Validate consistency between different sensor readings"""
        results = []

        if (len(self.sensor_data['laser']) > 0 and
            len(self.sensor_data['odometry']) > 0):

            latest_scan = self.sensor_data['laser'][-1]
            latest_odom = self.sensor_data['odometry'][-1]

            # Check if robot position is consistent with expected environment
            robot_x = latest_odom.pose.pose.position.x
            robot_y = latest_odom.pose.pose.position.y

            # Example: if robot is near a wall, laser should detect it
            ranges = np.array(latest_scan.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]

            if len(valid_ranges) > 0:
                min_range = np.min(valid_ranges)

                # If robot is near environment boundaries, expect some obstacles
                near_boundary = abs(robot_x) > 4.5 or abs(robot_y) > 4.5
                expects_obstacle = near_boundary and min_range < 1.0

                passed = not (near_boundary and min_range > 1.5)  # Should detect walls when near boundary

                results.append(ValidationResult(
                    sensor_type='multi_sensor',
                    test_name='environment_consistency',
                    passed=passed,
                    expected_value={'near_boundary': near_boundary},
                    actual_value={'min_laser_range': min_range, 'robot_pos': (robot_x, robot_y)},
                    tolerance=1.0,
                    error_message="" if passed else f"Robot position {robot_x:.2f},{robot_y:.2f} inconsistent with laser range {min_range:.2f}"
                ))

        return results

    def run_comprehensive_validation(self):
        """Run all validation tests and return comprehensive results"""
        all_results = []

        # Run scenario-based validations
        for scenario in self.test_scenarios:
            result = scenario.validation_function()
            if result:
                all_results.append(result)

        # Run joint range validations
        joint_results = self.validate_joint_ranges()
        all_results.extend(joint_results)

        # Run consistency validations
        consistency_results = self.validate_sensor_consistency()
        all_results.extend(consistency_results)

        return all_results

    def get_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        all_results = self.run_comprehensive_validation()

        passed_count = sum(1 for r in all_results if r.passed)
        total_count = len(all_results)

        report = {
            'timestamp': time.time(),
            'total_tests': total_count,
            'passed_tests': passed_count,
            'failed_tests': total_count - passed_count,
            'success_rate': passed_count / total_count if total_count > 0 else 0,
            'individual_results': [
                {
                    'sensor_type': r.sensor_type,
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'expected': r.expected_value,
                    'actual': r.actual_value,
                    'tolerance': r.tolerance,
                    'error': r.error_message
                } for r in all_results
            ],
            'statistics': self.stats
        }

        return report

    def log_validation_summary(self):
        """Log a summary of validation results"""
        if self.stats['total_tests'] > 0:
            success_rate = (self.stats['passed_tests'] / self.stats['total_tests']) * 100
            self.get_logger().info(
                f"Validation Summary - Cycles: {self.stats['validation_cycles']}, "
                f"Tests: {self.stats['total_tests']}, "
                f"Passed: {self.stats['passed_tests']} ({success_rate:.1f}%), "
                f"Failed: {self.stats['failed_tests']}"
            )

    def print_detailed_report(self):
        """Print a detailed validation report"""
        report = self.get_validation_report()

        print("\n" + "="*60)
        print("SENSOR VALIDATION DETAILED REPORT")
        print("="*60)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.2f}%")
        print("\nIndividual Test Results:")
        print("-" * 60)

        for result in report['individual_results']:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{status:4} | {result['sensor_type']:8} | {result['test_name']:20} | {result['error'] or 'OK'}")

        print("="*60)


class SensorCalibrator:
    """
    Utility class for sensor calibration and validation
    """

    def __init__(self):
        self.calibration_data = {}
        self.calibration_offsets = {}

    def calibrate_sensor(self, sensor_type: str, reference_data: np.ndarray) -> Dict[str, float]:
        """Calibrate a sensor based on reference data"""
        if sensor_type == 'laser':
            # Calculate statistical properties for laser calibration
            mean_val = np.mean(reference_data)
            std_val = np.std(reference_data)

            self.calibration_offsets[sensor_type] = {
                'mean_offset': 0.0,  # We want to keep the mean as reference
                'std_threshold': std_val * 2  # 2-sigma threshold for anomalies
            }

            return self.calibration_offsets[sensor_type]

        elif sensor_type == 'imu':
            # For IMU, we might calibrate bias
            return {
                'bias_x': 0.0,
                'bias_y': 0.0,
                'bias_z': 0.0,
                'scale_factor': 1.0
            }

        return {}

    def validate_calibration(self, sensor_type: str, test_data: np.ndarray) -> bool:
        """Validate if test data is within calibrated parameters"""
        if sensor_type not in self.calibration_offsets:
            return True  # No calibration data, assume valid

        offsets = self.calibration_offsets[sensor_type]

        if sensor_type == 'laser':
            mean_val = np.mean(test_data)
            # Check if mean is within calibrated range
            return abs(mean_val) < offsets['std_threshold']

        return True


def main(args=None):
    """Main function to run sensor validation tests"""
    rclpy.init(args=args)

    validator = SensorValidator()

    try:
        # Run for a period of time to collect validation data
        start_time = time.time()
        run_duration = 10  # Run for 10 seconds

        while time.time() - start_time < run_duration:
            rclpy.spin_once(validator, timeout_sec=0.1)

        # Print detailed report
        validator.print_detailed_report()

    except KeyboardInterrupt:
        validator.get_logger().info("Sensor validation interrupted by user")
    finally:
        # Print final report
        validator.print_detailed_report()
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()