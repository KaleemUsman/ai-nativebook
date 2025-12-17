#!/usr/bin/env python3
"""
Sensor Validation Test Suite
This script validates sensor outputs from Gazebo simulation against expected
values in various scenarios to ensure realistic and reliable sensor data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import threading
from collections import deque
import statistics
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class SensorValidationResult:
    """Result of a sensor validation test"""
    test_name: str
    sensor_type: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class SensorValidationTester(Node):
    """
    Comprehensive tester for validating sensor outputs from Gazebo simulation
    """

    def __init__(self):
        super().__init__('sensor_validation_tester')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=100),
            'imu': deque(maxlen=100),
            'joint': deque(maxlen=100),
            'camera': deque(maxlen=20)
        }

        self.validation_metrics = {
            'laser_accuracy': 0.0,
            'imu_stability': 0.0,
            'joint_precision': 0.0,
            'camera_quality': 0.0,
            'sensor_consistency': 0.0
        }

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

        # Publishers for test control and visualization
        self.test_control_pub = self.create_publisher(
            String,
            '/sensor_validation/control',
            10
        )

        self.validation_results_pub = self.create_publisher(
            String,
            '/sensor_validation/results',
            10
        )

        self.debug_marker_pub = self.create_publisher(
            Marker,
            '/sensor_validation/debug_markers',
            10
        )

        # Timer for validation monitoring
        self.monitor_timer = self.create_timer(0.1, self.validation_monitoring)

        # Test parameters
        self.test_scenarios = [
            'laser_scan_validation',
            'imu_data_validation',
            'joint_state_validation',
            'camera_feed_validation',
            'multi_sensor_consistency'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        self.get_logger().info("Sensor Validation Tester initialized and ready for testing")

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.sensor_data['laser'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def imu_callback(self, msg):
        """Process IMU data"""
        self.sensor_data['imu'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def joint_callback(self, msg):
        """Process joint state data"""
        self.sensor_data['joint'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def camera_callback(self, msg):
        """Process camera image data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.sensor_data['camera'].append({
                'data': cv_image,
                'header': msg.header,
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f"Error processing camera data: {e}")

    def validation_monitoring(self):
        """Monitor sensor data quality and consistency"""
        try:
            # Calculate validation metrics based on sensor data
            self.update_laser_metrics()
            self.update_imu_metrics()
            self.update_joint_metrics()
            self.update_camera_metrics()
            self.update_consistency_metrics()

        except Exception as e:
            self.get_logger().error(f"Error in validation monitoring: {e}")

    def update_laser_metrics(self):
        """Update metrics for LiDAR sensor validation"""
        if len(self.sensor_data['laser']) > 0:
            latest_scan = self.sensor_data['laser'][-1]['data']
            ranges = np.array(latest_scan.ranges)

            # Calculate range statistics
            valid_ranges = ranges[np.isfinite(ranges)]
            if len(valid_ranges) > 0:
                range_mean = np.mean(valid_ranges)
                range_std = np.std(valid_ranges)

                # Calculate accuracy score based on expected range characteristics
                # In free space, expect most ranges to be close to max range
                # Near obstacles, expect some ranges to be shorter
                max_range_ratio = np.sum(valid_ranges > (latest_scan.range_max * 0.8)) / len(valid_ranges)
                self.validation_metrics['laser_accuracy'] = max_range_ratio

    def update_imu_metrics(self):
        """Update metrics for IMU sensor validation"""
        if len(self.sensor_data['imu']) > 10:
            # Calculate stability based on variation in readings
            orientations = []
            angular_velocities = []
            linear_accelerations = []

            for data in list(self.sensor_data['imu'])[-10:]:
                imu = data['data']
                orientations.append([
                    imu.orientation.x, imu.orientation.y,
                    imu.orientation.z, imu.orientation.w
                ])
                angular_velocities.append([
                    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z
                ])
                linear_accelerations.append([
                    imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z
                ])

            orient_array = np.array(orientations)
            ang_vel_array = np.array(angular_velocities)
            lin_acc_array = np.array(linear_accelerations)

            # Calculate variance (lower variance = more stable)
            orient_variance = np.mean(np.var(orient_array, axis=0))
            ang_vel_variance = np.mean(np.var(ang_vel_array, axis=0))
            lin_acc_variance = np.mean(np.var(lin_acc_array, axis=0))

            # Convert to stability score (lower variance = higher stability)
            stability_score = max(0, min(1, 1 - (orient_variance + ang_vel_variance + lin_acc_variance) * 5))
            self.validation_metrics['imu_stability'] = stability_score

    def update_joint_metrics(self):
        """Update metrics for joint state sensor validation"""
        if len(self.sensor_data['joint']) > 5:
            # Calculate joint precision based on consistency
            positions = []
            velocities = []

            for data in list(self.sensor_data['joint'])[-5:]:
                joint = data['data']
                if len(joint.position) > 0:
                    positions.append(joint.position[:min(5, len(joint.position))])  # Take first 5 joints
                if len(joint.velocity) > 0:
                    velocities.append(joint.velocity[:min(5, len(joint.velocity))])

            if len(positions) > 1:
                pos_array = np.array(positions)
                # Calculate variance in joint positions (lower variance = more precise)
                if pos_array.ndim > 1:
                    pos_variance = np.mean(np.var(pos_array, axis=0))
                    self.validation_metrics['joint_precision'] = max(0, min(1, 1 - pos_variance))

    def update_camera_metrics(self):
        """Update metrics for camera sensor validation"""
        if len(self.sensor_data['camera']) > 0:
            latest_img = self.sensor_data['camera'][-1]['data']

            # Calculate image quality metrics
            # Brightness
            gray = cv2.cvtColor(latest_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            # Contrast
            contrast = np.std(gray)

            # Sharpness (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Calculate quality score based on these metrics
            brightness_score = max(0, min(1, abs(brightness - 128) / 128))  # 0-255 range
            contrast_score = max(0, min(1, contrast / 100))  # Normalized
            sharpness_score = max(0, min(1, laplacian_var / 1000))  # Normalized

            quality_score = (brightness_score + contrast_score + sharpness_score) / 3
            self.validation_metrics['camera_quality'] = quality_score

    def update_consistency_metrics(self):
        """Update metrics for multi-sensor consistency"""
        # Check if data from different sensors is temporally consistent
        if (len(self.sensor_data['laser']) > 0 and
            len(self.sensor_data['imu']) > 0 and
            len(self.sensor_data['joint']) > 0):

            laser_time = self.sensor_data['laser'][-1]['timestamp']
            imu_time = self.sensor_data['imu'][-1]['timestamp']
            joint_time = self.sensor_data['joint'][-1]['timestamp']

            # Calculate time differences
            max_time_diff = max(
                abs(laser_time - imu_time),
                abs(laser_time - joint_time),
                abs(imu_time - joint_time)
            )

            # Calculate consistency score (lower time difference = higher consistency)
            consistency_score = max(0, min(1, 1 - max_time_diff))  # Assuming 1s max acceptable delay
            self.validation_metrics['sensor_consistency'] = consistency_score

    def run_comprehensive_test(self):
        """Run comprehensive sensor validation tests"""
        self.get_logger().info("Starting comprehensive sensor validation tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running sensor validation test scenario: {scenario}")
            start_time = time.time()

            result = self.run_specific_test(scenario)
            duration = time.time() - start_time

            result.duration = duration
            self.test_results.append(result)

            self.test_statistics['total_tests'] += 1
            if result.passed:
                self.test_statistics['passed_tests'] += 1
            else:
                self.test_statistics['failed_tests'] += 1

        # Calculate average duration
        if self.test_statistics['total_tests'] > 0:
            total_duration = sum(r.duration for r in self.test_results)
            self.test_statistics['average_duration'] = total_duration / self.test_statistics['total_tests']

        # Publish results
        self.publish_test_results()

        return self.test_results

    def run_specific_test(self, scenario: str) -> SensorValidationResult:
        """Run a specific sensor validation test scenario"""
        if scenario == 'laser_scan_validation':
            return self.run_laser_validation()
        elif scenario == 'imu_data_validation':
            return self.run_imu_validation()
        elif scenario == 'joint_state_validation':
            return self.run_joint_validation()
        elif scenario == 'camera_feed_validation':
            return self.run_camera_validation()
        elif scenario == 'multi_sensor_consistency':
            return self.run_consistency_validation()
        else:
            return SensorValidationResult(
                test_name=scenario,
                sensor_type='unknown',
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_laser_validation(self) -> SensorValidationResult:
        """Validate LiDAR sensor data quality"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(3)

        if len(self.sensor_data['laser']) > 0:
            latest_scan = self.sensor_data['laser'][-1]['data']
            ranges = np.array(latest_scan.ranges)

            # Validate range values
            valid_ranges = ranges[np.isfinite(ranges)]
            range_accuracy = self.validation_metrics['laser_accuracy']

            # Check for expected range characteristics
            range_values_valid = len(valid_ranges) > 0
            range_bounds_valid = np.all((valid_ranges >= latest_scan.range_min) &
                                       (valid_ranges <= latest_scan.range_max))

            # Check for reasonable distribution of ranges
            if len(valid_ranges) > 10:
                # In a typical environment, expect some variety in ranges
                range_variance = np.var(valid_ranges)
                range_distribution_reasonable = range_variance > 0.01

            metrics = {
                'range_accuracy_score': range_accuracy,
                'valid_range_count': len(valid_ranges),
                'range_values_valid': range_values_valid,
                'range_bounds_valid': range_bounds_valid,
                'range_distribution_reasonable': range_distribution_reasonable if 'range_distribution_reasonable' in locals() else True,
                'duration': time.time() - start_time
            }

            # Pass if all validations pass
            passed = range_values_valid and range_bounds_valid

        else:
            metrics = {
                'error': 'No laser data available',
                'duration': time.time() - start_time
            }
            passed = False

        return SensorValidationResult(
            test_name='laser_scan_validation',
            sensor_type='laser',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "LiDAR validation failed"
        )

    def run_imu_validation(self) -> SensorValidationResult:
        """Validate IMU sensor data quality"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(2)

        if len(self.sensor_data['imu']) > 10:
            latest_imu = self.sensor_data['imu'][-1]['data']

            # Validate quaternion normalization
            quat_norm = math.sqrt(
                latest_imu.orientation.x**2 +
                latest_imu.orientation.y**2 +
                latest_imu.orientation.z**2 +
                latest_imu.orientation.w**2
            )

            quat_normalized = abs(quat_norm - 1.0) < 0.01  # Within 1% tolerance

            # Validate angular velocity magnitudes (reasonable for humanoid)
            ang_vel_mag = math.sqrt(
                latest_imu.angular_velocity.x**2 +
                latest_imu.angular_velocity.y**2 +
                latest_imu.angular_velocity.z**2
            )
            ang_vel_reasonable = ang_vel_mag < 10.0  # 10 rad/s max

            # Validate linear acceleration (should include gravity ~9.8 m/s^2)
            lin_acc_mag = math.sqrt(
                latest_imu.linear_acceleration.x**2 +
                latest_imu.linear_acceleration.y**2 +
                latest_imu.linear_acceleration.z**2
            )
            # For a standing humanoid, expect ~9.8 m/s^2 (gravity)
            grav_aligned = abs(lin_acc_mag - 9.8) < 2.0  # Within 2 m/s^2 tolerance

            stability_score = self.validation_metrics['imu_stability']

            metrics = {
                'quaternion_normalized': quat_normalized,
                'angular_velocity_reasonable': ang_vel_reasonable,
                'gravity_aligned': grav_aligned,
                'stability_score': stability_score,
                'duration': time.time() - start_time
            }

            passed = quat_normalized and ang_vel_reasonable and stability_score > 0.5

        else:
            metrics = {
                'error': 'Insufficient IMU data for validation',
                'duration': time.time() - start_time
            }
            passed = False

        return SensorValidationResult(
            test_name='imu_data_validation',
            sensor_type='imu',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "IMU validation failed"
        )

    def run_joint_validation(self) -> SensorValidationResult:
        """Validate joint state sensor data quality"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(2)

        if len(self.sensor_data['joint']) > 5:
            latest_joint = self.sensor_data['joint'][-1]['data']

            # Validate joint positions are within reasonable ranges
            position_reasonable = True
            if len(latest_joint.position) > 0:
                # Check if joint positions are within expected physical limits
                for pos in latest_joint.position:
                    if abs(pos) > 10:  # Unreasonably large joint angle
                        position_reasonable = False
                        break

            # Validate joint velocities are reasonable
            velocity_reasonable = True
            if len(latest_joint.velocity) > 0:
                for vel in latest_joint.velocity:
                    if abs(vel) > 10:  # Unreasonably large joint velocity
                        velocity_reasonable = False
                        break

            precision_score = self.validation_metrics['joint_precision']

            metrics = {
                'position_reasonable': position_reasonable,
                'velocity_reasonable': velocity_reasonable,
                'precision_score': precision_score,
                'joint_count': len(latest_joint.position),
                'duration': time.time() - start_time
            }

            passed = position_reasonable and velocity_reasonable and precision_score > 0.5

        else:
            metrics = {
                'error': 'Insufficient joint data for validation',
                'duration': time.time() - start_time
            }
            passed = False

        return SensorValidationResult(
            test_name='joint_state_validation',
            sensor_type='joint',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "Joint validation failed"
        )

    def run_camera_validation(self) -> SensorValidationResult:
        """Validate camera sensor data quality"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(2)

        if len(self.sensor_data['camera']) > 0:
            latest_img = self.sensor_data['camera'][-1]['data']

            # Validate image dimensions and quality
            img_height, img_width = latest_img.shape[:2]
            img_dimensions_valid = img_width > 0 and img_height > 0

            # Validate image data type and range
            img_dtype_valid = latest_img.dtype == np.uint8
            img_range_valid = np.max(latest_img) <= 255 and np.min(latest_img) >= 0

            # Calculate image quality metrics
            quality_score = self.validation_metrics['camera_quality']

            metrics = {
                'dimensions_valid': img_dimensions_valid,
                'dtype_valid': img_dtype_valid,
                'range_valid': img_range_valid,
                'quality_score': quality_score,
                'image_size': f"{img_width}x{img_height}",
                'duration': time.time() - start_time
            }

            passed = img_dimensions_valid and img_dtype_valid and img_range_valid and quality_score > 0.3

        else:
            metrics = {
                'error': 'No camera data available',
                'duration': time.time() - start_time
            }
            passed = False

        return SensorValidationResult(
            test_name='camera_feed_validation',
            sensor_type='camera',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "Camera validation failed"
        )

    def run_consistency_validation(self) -> SensorValidationResult:
        """Validate consistency across multiple sensors"""
        start_time = time.time()

        # Check if we have data from all sensors
        sensors_have_data = (
            len(self.sensor_data['laser']) > 0 and
            len(self.sensor_data['imu']) > 0 and
            len(self.sensor_data['joint']) > 0
        )

        if sensors_have_data:
            consistency_score = self.validation_metrics['sensor_consistency']

            # Additional consistency checks
            # For example, check if IMU orientation is consistent with expected robot pose
            # Or if joint positions are consistent with forward kinematics

            metrics = {
                'temporal_consistency_score': consistency_score,
                'sensors_data_available': sensors_have_data,
                'duration': time.time() - start_time
            }

            passed = consistency_score > 0.7  # Require high consistency

        else:
            metrics = {
                'error': 'Insufficient sensor data for consistency check',
                'duration': time.time() - start_time
            }
            passed = False

        return SensorValidationResult(
            test_name='multi_sensor_consistency',
            sensor_type='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "Multi-sensor consistency validation failed"
        )

    def publish_test_results(self):
        """Publish sensor validation test results"""
        results_msg = String()
        results_msg.data = json.dumps({
            'timestamp': time.time(),
            'test_results': [
                {
                    'test_name': r.test_name,
                    'sensor_type': r.sensor_type,
                    'passed': r.passed,
                    'metrics': r.metrics,
                    'duration': r.duration,
                    'error_message': r.error_message
                } for r in self.test_results
            ],
            'statistics': self.test_statistics,
            'validation_metrics': self.validation_metrics
        })

        self.validation_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive sensor validation test report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'validation_metrics': self.validation_metrics.copy(),
            'individual_results': [
                {
                    'test_name': r.test_name,
                    'sensor_type': r.sensor_type,
                    'passed': r.passed,
                    'duration': r.duration,
                    'metrics': r.metrics,
                    'error': r.error_message
                } for r in self.test_results
            ],
            'recommendations': self.generate_recommendations()
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on sensor validation results"""
        recommendations = []

        if self.validation_metrics['laser_accuracy'] < 0.7:
            recommendations.append("LiDAR accuracy needs improvement - check sensor configuration and noise parameters")

        if self.validation_metrics['imu_stability'] < 0.7:
            recommendations.append("IMU stability is low - verify sensor mounting and noise models")

        if self.validation_metrics['joint_precision'] < 0.7:
            recommendations.append("Joint state precision is low - check joint limit configurations")

        if self.validation_metrics['camera_quality'] < 0.5:
            recommendations.append("Camera feed quality is poor - verify camera parameters and lighting")

        if self.validation_metrics['sensor_consistency'] < 0.8:
            recommendations.append("Multi-sensor consistency is low - check timing synchronization")

        if not recommendations:
            recommendations.append("All sensor validations passed - continue monitoring for optimization")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed sensor validation test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("SENSOR VALIDATION COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("VALIDATION METRICS:")
        print("-" * 40)
        metrics = report['validation_metrics']
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("\nINDIVIDUAL TEST RESULTS:")
        print("-" * 40)
        for result in report['individual_results']:
            status = "PASS" if result['passed'] else "FAIL"
            sensor_type = result['sensor_type']
            print(f"  {status:4} | {sensor_type:8} | {result['test_name']:25} | {result['duration']:.2f}s | {result['error'] or 'OK'}")

        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("="*80)


def main(args=None):
    """Main function to run sensor validation tests"""
    rclpy.init(args=args)

    tester = SensorValidationTester()

    try:
        # Run comprehensive sensor validation tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Sensor Validation Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Sensor validation testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()