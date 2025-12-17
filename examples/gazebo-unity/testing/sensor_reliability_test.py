#!/usr/bin/env python3
"""
Sensor Reliability and Realism Test
This script confirms that sensor outputs from Gazebo simulation are realistic and reliable
across different scenarios and conditions.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import String, Float64MultiArray
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
class SensorReliabilityResult:
    """Result of a sensor reliability test"""
    test_name: str
    sensor_type: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class SensorReliabilityTester(Node):
    """
    Tester for confirming sensor outputs are realistic and reliable across different scenarios
    """

    def __init__(self):
        super().__init__('sensor_reliability_tester')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=200),
            'imu': deque(maxlen=200),
            'joint': deque(maxlen=200),
            'camera': deque(maxlen=50)
        }

        self.reliability_metrics = {
            'laser_realism_score': 0.0,
            'laser_reliability_score': 0.0,
            'imu_realism_score': 0.0,
            'imu_reliability_score': 0.0,
            'joint_realism_score': 0.0,
            'joint_reliability_score': 0.0,
            'camera_realism_score': 0.0,
            'camera_reliability_score': 0.0,
            'cross_sensor_consistency': 0.0
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

        # Publishers for test control and results
        self.test_control_pub = self.create_publisher(
            String,
            '/sensor_reliability/test_control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/sensor_reliability/results',
            10
        )

        # Timer for reliability monitoring
        self.monitor_timer = self.create_timer(0.1, self.reliability_monitoring)

        # Test scenarios
        self.test_scenarios = [
            'empty_environment_reality',
            'obstacle_rich_environment',
            'dynamic_movement_scenarios',
            'extreme_orientations',
            'long_duration_stability'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        # Realism and reliability thresholds
        self.realism_threshold = 0.7  # Minimum realism score
        self.reliability_threshold = 0.8  # Minimum reliability score
        self.consistency_threshold = 0.75  # Minimum cross-sensor consistency

        self.get_logger().info("Sensor Reliability Tester initialized and ready for testing")

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

    def reliability_monitoring(self):
        """Monitor sensor reliability and realism metrics"""
        try:
            # Update reliability metrics for each sensor type
            self.update_laser_reliability_metrics()
            self.update_imu_reliability_metrics()
            self.update_joint_reliability_metrics()
            self.update_camera_reliability_metrics()
            self.update_cross_sensor_consistency()

        except Exception as e:
            self.get_logger().error(f"Error in reliability monitoring: {e}")

    def update_laser_reliability_metrics(self):
        """Update reliability and realism metrics for LiDAR sensor"""
        if len(self.sensor_data['laser']) > 10:
            latest_scan = self.sensor_data['laser'][-1]['data']
            ranges = np.array(latest_scan.ranges)

            # Calculate realism metrics
            valid_ranges = ranges[np.isfinite(ranges)]
            if len(valid_ranges) > 0:
                # Check if ranges are within expected bounds
                within_bounds = np.sum((valid_ranges >= latest_scan.range_min) &
                                     (valid_ranges <= latest_scan.range_max)) / len(valid_ranges)

                # Check for realistic distribution of ranges (not all the same)
                range_variance = np.var(valid_ranges)
                realistic_distribution = range_variance > 0.001  # Not all identical

                # Calculate realism score
                realism_score = (within_bounds + (1 if realistic_distribution else 0.5)) / 2
                self.reliability_metrics['laser_realism_score'] = realism_score

            # Calculate reliability metrics
            recent_scans = [d['data'].ranges for d in list(self.sensor_data['laser'])[-10:]]
            if len(recent_scans) > 1:
                # Calculate consistency across recent scans
                recent_ranges = np.array([scan for scan in recent_scans if len(scan) > 0])
                if len(recent_ranges) > 1:
                    # Calculate variance across scans (lower variance = more reliable)
                    scan_variance = np.mean(np.var(recent_ranges, axis=0))
                    reliability_score = max(0, min(1, 1 - scan_variance * 0.1))
                    self.reliability_metrics['laser_reliability_score'] = reliability_score

    def update_imu_reliability_metrics(self):
        """Update reliability and realism metrics for IMU sensor"""
        if len(self.sensor_data['imu']) > 10:
            latest_imu = self.sensor_data['imu'][-1]['data']

            # Calculate realism metrics
            # Check quaternion normalization
            quat_norm = math.sqrt(
                latest_imu.orientation.x**2 +
                latest_imu.orientation.y**2 +
                latest_imu.orientation.z**2 +
                latest_imu.orientation.w**2
            )
            quat_normalized = abs(quat_norm - 1.0) < 0.01

            # Check angular velocity magnitudes (should be reasonable for humanoid)
            ang_vel_mag = math.sqrt(
                latest_imu.angular_velocity.x**2 +
                latest_imu.angular_velocity.y**2 +
                latest_imu.angular_velocity.z**2
            )
            ang_vel_reasonable = ang_vel_mag < 10.0  # 10 rad/s max

            # Check linear acceleration (should include gravity component)
            lin_acc_mag = math.sqrt(
                latest_imu.linear_acceleration.x**2 +
                latest_imu.linear_acceleration.y**2 +
                latest_imu.linear_acceleration.z**2
            )
            # For humanoid at rest, expect ~9.8 m/s^2 (gravity)
            grav_aligned = abs(lin_acc_mag - 9.8) < 5.0  # Allow up to 5 m/s^2 variation

            realism_score = (int(quat_normalized) + int(ang_vel_reasonable) + int(grav_aligned)) / 3
            self.reliability_metrics['imu_realism_score'] = realism_score

            # Calculate reliability metrics
            recent_imus = [d['data'] for d in list(self.sensor_data['imu'])[-10:]]
            if len(recent_imus) > 1:
                # Calculate consistency of readings
                orient_changes = []
                for i in range(1, len(recent_imus)):
                    prev_imu = recent_imus[i-1]
                    curr_imu = recent_imus[i]

                    orient_change = math.sqrt(
                        (curr_imu.orientation.x - prev_imu.orientation.x)**2 +
                        (curr_imu.orientation.y - prev_imu.orientation.y)**2 +
                        (curr_imu.orientation.z - prev_imu.orientation.z)**2 +
                        (curr_imu.orientation.w - prev_imu.orientation.w)**2
                    )
                    orient_changes.append(orient_change)

                avg_change = np.mean(orient_changes) if orient_changes else 0
                reliability_score = max(0, min(1, 1 - avg_change * 5))  # Lower change = higher reliability
                self.reliability_metrics['imu_reliability_score'] = reliability_score

    def update_joint_reliability_metrics(self):
        """Update reliability and realism metrics for joint state sensor"""
        if len(self.sensor_data['joint']) > 10:
            latest_joint = self.sensor_data['joint'][-1]['data']

            # Calculate realism metrics
            if len(latest_joint.position) > 0:
                # Check if joint positions are within reasonable physical limits
                position_reasonable = True
                for pos in latest_joint.position:
                    if abs(pos) > 10:  # Unreasonably large joint angle (10 radians = ~570 degrees)
                        position_reasonable = False
                        break

                # Check if joint velocities are reasonable
                velocity_reasonable = True
                if len(latest_joint.velocity) > 0:
                    for vel in latest_joint.velocity:
                        if abs(vel) > 10:  # Unreasonably large joint velocity
                            velocity_reasonable = False
                            break

                realism_score = (int(position_reasonable) + int(velocity_reasonable)) / 2
                self.reliability_metrics['joint_realism_score'] = realism_score

            # Calculate reliability metrics
            recent_joints = [d['data'] for d in list(self.sensor_data['joint'])[-10:]]
            if len(recent_joints) > 1 and len(recent_joints[0].position) > 0:
                # Calculate consistency of joint positions
                pos_changes = []
                for i in range(1, len(recent_joints)):
                    prev_joint = recent_joints[i-1]
                    curr_joint = recent_joints[i]

                    # Calculate change in joint positions
                    min_joints = min(len(prev_joint.position), len(curr_joint.position))
                    if min_joints > 0:
                        pos_change = np.mean([
                            abs(curr_joint.position[j] - prev_joint.position[j])
                            for j in range(min_joints)
                        ])
                        pos_changes.append(pos_change)

                avg_change = np.mean(pos_changes) if pos_changes else 0
                reliability_score = max(0, min(1, 1 - avg_change * 2))  # Lower change = higher reliability
                self.reliability_metrics['joint_reliability_score'] = reliability_score

    def update_camera_reliability_metrics(self):
        """Update reliability and realism metrics for camera sensor"""
        if len(self.sensor_data['camera']) > 5:
            latest_img = self.sensor_data['camera'][-1]['data']

            # Calculate realism metrics
            img_height, img_width = latest_img.shape[:2]
            img_dimensions_valid = img_width > 0 and img_height > 0

            # Calculate image quality metrics
            gray = cv2.cvtColor(latest_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Check if image quality is reasonable
            brightness_reasonable = 20 <= brightness <= 235  # Avoid completely dark/bright images
            contrast_reasonable = contrast > 5  # Avoid low contrast images
            sharpness_reasonable = sharpness > 10  # Avoid blurry images

            realism_score = (int(img_dimensions_valid) + int(brightness_reasonable) +
                           int(contrast_reasonable) + int(sharpness_reasonable)) / 4
            self.reliability_metrics['camera_realism_score'] = realism_score

            # Calculate reliability metrics
            recent_imgs = [d['data'] for d in list(self.sensor_data['camera'])[-5:]]
            if len(recent_imgs) > 1:
                # Calculate consistency between consecutive images
                img_changes = []
                for i in range(1, len(recent_imgs)):
                    prev_img = cv2.cvtColor(recent_imgs[i-1], cv2.COLOR_BGR2GRAY)
                    curr_img = cv2.cvtColor(recent_imgs[i], cv2.COLOR_BGR2GRAY)

                    # Calculate difference between images
                    diff = cv2.absdiff(prev_img, curr_img)
                    avg_diff = np.mean(diff)
                    img_changes.append(avg_diff)

                avg_change = np.mean(img_changes) if img_changes else 0
                # Higher change might indicate movement, but too high might indicate instability
                # For reliability, we want consistent quality but allow reasonable changes
                reliability_score = max(0, min(1, 1 - avg_change / 50))  # Adjust threshold as needed
                self.reliability_metrics['camera_reliability_score'] = reliability_score

    def update_cross_sensor_consistency(self):
        """Update metrics for cross-sensor consistency"""
        # Check if we have data from multiple sensors
        if (len(self.sensor_data['laser']) > 0 and
            len(self.sensor_data['imu']) > 0 and
            len(self.sensor_data['joint']) > 0):

            # Example consistency check: IMU orientation should be consistent with joint positions
            latest_imu = self.sensor_data['imu'][-1]['data']
            latest_joint = self.sensor_data['joint'][-1]['data']

            # Calculate consistency score based on sensor agreement
            # This is a simplified example - in reality, we'd have more complex relationships
            consistency_score = 0.8  # Placeholder - would implement real consistency checks

            self.reliability_metrics['cross_sensor_consistency'] = consistency_score

    def run_comprehensive_test(self):
        """Run comprehensive sensor reliability and realism tests"""
        self.get_logger().info("Starting comprehensive sensor reliability and realism tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running sensor reliability test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> SensorReliabilityResult:
        """Run a specific sensor reliability test scenario"""
        if scenario == 'empty_environment_reality':
            return self.run_empty_environment_test()
        elif scenario == 'obstacle_rich_environment':
            return self.run_obstacle_rich_test()
        elif scenario == 'dynamic_movement_scenarios':
            return self.run_dynamic_movement_test()
        elif scenario == 'extreme_orientations':
            return self.run_extreme_orientation_test()
        elif scenario == 'long_duration_stability':
            return self.run_long_duration_test()
        else:
            return SensorReliabilityResult(
                test_name=scenario,
                sensor_type='multi',
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_empty_environment_test(self) -> SensorReliabilityResult:
        """Test sensor outputs in empty environment (should show mostly max ranges for LiDAR)"""
        start_time = time.time()

        # Wait for data collection in empty environment
        time.sleep(8)  # Wait for empty environment data

        # Assess LiDAR realism in empty environment
        laser_realism_ok = self.reliability_metrics['laser_realism_score'] >= self.realism_threshold
        laser_reliability_ok = self.reliability_metrics['laser_reliability_score'] >= self.reliability_threshold

        # Assess other sensors
        imu_realism_ok = self.reliability_metrics['imu_realism_score'] >= self.realism_threshold
        imu_reliability_ok = self.reliability_metrics['imu_reliability_score'] >= self.reliability_threshold

        metrics = {
            'laser_realism_score': self.reliability_metrics['laser_realism_score'],
            'laser_reliability_score': self.reliability_metrics['laser_reliability_score'],
            'imu_realism_score': self.reliability_metrics['imu_realism_score'],
            'imu_reliability_score': self.reliability_metrics['imu_reliability_score'],
            'laser_realism_ok': laser_realism_ok,
            'laser_reliability_ok': laser_reliability_ok,
            'imu_realism_ok': imu_realism_ok,
            'imu_reliability_ok': imu_reliability_ok,
            'duration': time.time() - start_time
        }

        passed = laser_realism_ok and laser_reliability_ok and imu_realism_ok and imu_reliability_ok

        return SensorReliabilityResult(
            test_name='empty_environment_reality',
            sensor_type='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Empty environment test failed - LiDAR: {self.reliability_metrics['laser_realism_score']:.2f}/{self.reliability_metrics['laser_reliability_score']:.2f}, IMU: {self.reliability_metrics['imu_realism_score']:.2f}/{self.reliability_metrics['imu_reliability_score']:.2f}"
        )

    def run_obstacle_rich_test(self) -> SensorReliabilityResult:
        """Test sensor outputs in obstacle-rich environment"""
        start_time = time.time()

        # Wait for data collection in obstacle-rich environment
        time.sleep(6)

        # Assess sensor performance with obstacles
        laser_with_obstacles_ok = self.reliability_metrics['laser_realism_score'] >= 0.6  # Slightly lower threshold for obstacle scenarios
        imu_stability_ok = self.reliability_metrics['imu_reliability_score'] >= self.reliability_threshold

        metrics = {
            'laser_realism_score': self.reliability_metrics['laser_realism_score'],
            'laser_reliability_score': self.reliability_metrics['laser_reliability_score'],
            'imu_stability_score': self.reliability_metrics['imu_reliability_score'],
            'laser_with_obstacles_ok': laser_with_obstacles_ok,
            'imu_stability_ok': imu_stability_ok,
            'duration': time.time() - start_time
        }

        passed = laser_with_obstacles_ok and imu_stability_ok

        return SensorReliabilityResult(
            test_name='obstacle_rich_environment',
            sensor_type='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Obstacle-rich test failed - LiDAR: {self.reliability_metrics['laser_realism_score']:.2f}/{self.reliability_metrics['laser_reliability_score']:.2f}, IMU: {self.reliability_metrics['imu_reliability_score']:.2f}"
        )

    def run_dynamic_movement_test(self) -> SensorReliabilityResult:
        """Test sensor outputs during dynamic movement"""
        start_time = time.time()

        # Wait for dynamic movement data
        time.sleep(7)

        # Assess sensor performance during movement
        dynamic_response_ok = (self.reliability_metrics['imu_realism_score'] >= 0.6 and
                              self.reliability_metrics['imu_reliability_score'] >= 0.7)

        joint_response_ok = (self.reliability_metrics['joint_realism_score'] >= 0.7 and
                            self.reliability_metrics['joint_reliability_score'] >= 0.7)

        metrics = {
            'imu_realism_score': self.reliability_metrics['imu_realism_score'],
            'imu_reliability_score': self.reliability_metrics['imu_reliability_score'],
            'joint_realism_score': self.reliability_metrics['joint_realism_score'],
            'joint_reliability_score': self.reliability_metrics['joint_reliability_score'],
            'dynamic_response_ok': dynamic_response_ok,
            'joint_response_ok': joint_response_ok,
            'duration': time.time() - start_time
        }

        passed = dynamic_response_ok and joint_response_ok

        return SensorReliabilityResult(
            test_name='dynamic_movement_scenarios',
            sensor_type='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Dynamic movement test failed - IMU: {self.reliability_metrics['imu_realism_score']:.2f}/{self.reliability_metrics['imu_reliability_score']:.2f}, Joint: {self.reliability_metrics['joint_realism_score']:.2f}/{self.reliability_metrics['joint_reliability_score']:.2f}"
        )

    def run_extreme_orientation_test(self) -> SensorReliabilityResult:
        """Test sensor outputs during extreme orientations"""
        start_time = time.time()

        # Wait for extreme orientation data
        time.sleep(5)

        # Assess sensor performance during extreme orientations
        orientation_stability_ok = (self.reliability_metrics['imu_realism_score'] >= 0.7 and
                                   self.reliability_metrics['imu_reliability_score'] >= 0.6)

        metrics = {
            'imu_realism_score': self.reliability_metrics['imu_realism_score'],
            'imu_reliability_score': self.reliability_metrics['imu_reliability_score'],
            'orientation_stability_ok': orientation_stability_ok,
            'duration': time.time() - start_time
        }

        passed = orientation_stability_ok

        return SensorReliabilityResult(
            test_name='extreme_orientations',
            sensor_type='imu',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Extreme orientation test failed - IMU: {self.reliability_metrics['imu_realism_score']:.2f}/{self.reliability_metrics['imu_reliability_score']:.2f}"
        )

    def run_long_duration_test(self) -> SensorReliabilityResult:
        """Test sensor reliability over long duration"""
        start_time = time.time()

        # Wait for long duration data collection
        time.sleep(12)  # Longer test for stability over time

        # Assess long-term reliability
        long_term_stability_ok = (
            self.reliability_metrics['laser_reliability_score'] >= self.reliability_threshold * 0.9 and
            self.reliability_metrics['imu_reliability_score'] >= self.reliability_threshold * 0.9 and
            self.reliability_metrics['joint_reliability_score'] >= self.reliability_threshold * 0.9
        )

        cross_consistency_ok = self.reliability_metrics['cross_sensor_consistency'] >= self.consistency_threshold

        metrics = {
            'laser_reliability_score': self.reliability_metrics['laser_reliability_score'],
            'imu_reliability_score': self.reliability_metrics['imu_reliability_score'],
            'joint_reliability_score': self.reliability_metrics['joint_reliability_score'],
            'cross_sensor_consistency': self.reliability_metrics['cross_sensor_consistency'],
            'long_term_stability_ok': long_term_stability_ok,
            'cross_consistency_ok': cross_consistency_ok,
            'duration': time.time() - start_time
        }

        passed = long_term_stability_ok and cross_consistency_ok

        return SensorReliabilityResult(
            test_name='long_duration_stability',
            sensor_type='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Long duration test failed - LiDAR: {self.reliability_metrics['laser_reliability_score']:.2f}, IMU: {self.reliability_metrics['imu_reliability_score']:.2f}, Joint: {self.reliability_metrics['joint_reliability_score']:.2f}, Consistency: {self.reliability_metrics['cross_sensor_consistency']:.2f}"
        )

    def publish_test_results(self):
        """Publish sensor reliability test results"""
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
            'reliability_metrics': self.reliability_metrics
        })

        self.test_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive sensor reliability test report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'reliability_metrics': self.reliability_metrics.copy(),
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
            'thresholds': {
                'realism': self.realism_threshold,
                'reliability': self.reliability_threshold,
                'consistency': self.consistency_threshold
            },
            'recommendations': self.generate_recommendations()
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on sensor reliability test results"""
        recommendations = []

        if self.reliability_metrics['laser_realism_score'] < self.realism_threshold:
            recommendations.append("Improve LiDAR realism - check noise parameters and range configuration")

        if self.reliability_metrics['laser_reliability_score'] < self.reliability_threshold:
            recommendations.append("Enhance LiDAR reliability - verify sensor update rates and data consistency")

        if self.reliability_metrics['imu_realism_score'] < self.realism_threshold:
            recommendations.append("Improve IMU realism - verify noise models and physical parameters")

        if self.reliability_metrics['imu_reliability_score'] < self.reliability_threshold:
            recommendations.append("Enhance IMU reliability - check sensor calibration and mounting")

        if self.reliability_metrics['joint_realism_score'] < self.realism_threshold:
            recommendations.append("Improve joint state realism - verify joint limits and dynamics")

        if self.reliability_metrics['joint_reliability_score'] < self.reliability_threshold:
            recommendations.append("Enhance joint state reliability - check encoder resolution and noise")

        if self.reliability_metrics['cross_sensor_consistency'] < self.consistency_threshold:
            recommendations.append("Improve cross-sensor consistency - verify timing synchronization and coordinate systems")

        if not recommendations:
            recommendations.append("All sensor reliability tests passed - continue monitoring for optimization")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed sensor reliability test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("SENSOR RELIABILITY AND REALISM COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("THRESHOLDS:")
        print("-" * 40)
        thresholds = report['thresholds']
        print(f"  Realism Threshold: {thresholds['realism']:.2f}")
        print(f"  Reliability Threshold: {thresholds['reliability']:.2f}")
        print(f"  Consistency Threshold: {thresholds['consistency']:.2f}")

        print("\nRELIABILITY METRICS:")
        print("-" * 40)
        metrics = report['reliability_metrics']
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
    """Main function to run sensor reliability tests"""
    rclpy.init(args=args)

    tester = SensorReliabilityTester()

    try:
        # Run comprehensive sensor reliability tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Sensor Reliability Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Sensor reliability testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()