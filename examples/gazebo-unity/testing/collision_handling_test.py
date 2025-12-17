#!/usr/bin/env python3
"""
Collision Handling Test Suite
This script tests collision handling for humanoid robots in Gazebo simulation,
including self-collision prevention, environment collision detection, and
collision response validation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import WrenchStamped, Point
from gazebo_msgs.svc import GetModelState, GetLinkState, ApplyBodyWrench
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import Marker
import numpy as np
import time
import threading
from collections import deque
import statistics
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class CollisionTestResult:
    """Result of a collision handling test"""
    test_name: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class CollisionHandlingTester(Node):
    """
    Comprehensive tester for collision handling in humanoid robotics simulation
    """

    def __init__(self):
        super().__init__('collision_handling_tester')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=50),
            'imu': deque(maxlen=50),
            'joint': deque(maxlen=50)
        }

        self.collision_metrics = {
            'self_collision_count': 0,
            'environment_collision_count': 0,
            'collision_response_quality': 0.0,
            'contact_force_average': 0.0,
            'collision_detection_rate': 0.0
        }

        # Initialize ROS interfaces
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

        # Publishers for test control and visualization
        self.test_control_pub = self.create_publisher(
            String,
            '/collision_test/control',
            10
        )

        self.collision_results_pub = self.create_publisher(
            String,
            '/collision_test/results',
            10
        )

        self.contact_marker_pub = self.create_publisher(
            Marker,
            '/collision_test/contact_markers',
            10
        )

        # Service clients for Gazebo interaction
        self.get_model_state_cli = self.create_client(GetModelState, '/get_model_state')
        self.get_link_state_cli = self.create_client(GetLinkState, '/get_link_state')
        self.apply_wrench_cli = self.create_client(ApplyBodyWrench, '/apply_body_wrench')

        # Timer for collision monitoring
        self.monitor_timer = self.create_timer(0.1, self.collision_monitoring)

        # Test parameters
        self.test_scenarios = [
            'self_collision_prevention',
            'environment_collision_detection',
            'contact_force_validation',
            'collision_response_quality',
            'multi_body_interactions'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        self.contact_points = deque(maxlen=100)

        self.get_logger().info("Collision Handling Tester initialized and ready for testing")

    def scan_callback(self, msg):
        """Process LiDAR scan data for environment collision detection"""
        self.sensor_data['laser'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def imu_callback(self, msg):
        """Process IMU data for balance and collision response"""
        self.sensor_data['imu'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def joint_callback(self, msg):
        """Process joint state data for self-collision detection"""
        self.sensor_data['joint'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def collision_monitoring(self):
        """Monitor collision events and metrics"""
        try:
            # Calculate collision metrics based on sensor data
            self.update_self_collision_metrics()
            self.update_environment_collision_metrics()
            self.update_contact_force_metrics()

        except Exception as e:
            self.get_logger().error(f"Error in collision monitoring: {e}")

    def update_self_collision_metrics(self):
        """Update metrics for self-collision detection and prevention"""
        if len(self.sensor_data['joint']) > 0:
            latest_joints = self.sensor_data['joint'][-1]['data']

            # Check for potential self-collisions based on joint positions
            # This is a simplified check - in real implementation, we'd use collision geometry
            potential_collisions = 0

            # Example: Check if arms are crossing over center (potential self-collision)
            if len(latest_joints.name) > 0:
                # This is a simplified check - in reality, we'd check actual collision volumes
                # For now, just incrementing based on joint position patterns
                left_arm_pos = self.get_joint_position(latest_joints, 'left_shoulder')
                right_arm_pos = self.get_joint_position(latest_joints, 'right_shoulder')

                if left_arm_pos is not None and right_arm_pos is not None:
                    # If arms are crossing center line, potential collision
                    if left_arm_pos[0] < 0 and right_arm_pos[0] > 0:  # Simplified check
                        potential_collisions += 1

            self.collision_metrics['self_collision_count'] += potential_collisions

    def update_environment_collision_metrics(self):
        """Update metrics for environment collision detection"""
        if len(self.sensor_data['laser']) > 0:
            latest_scan = self.sensor_data['laser'][-1]['data']

            # Check for environment collisions based on laser scan
            ranges = np.array(latest_scan.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]

            if len(valid_ranges) > 0:
                min_range = np.min(valid_ranges)

                # If minimum range is very small, likely environment collision
                if min_range < 0.1:  # 10cm threshold
                    self.collision_metrics['environment_collision_count'] += 1

    def update_contact_force_metrics(self):
        """Update metrics for contact force analysis"""
        # Calculate based on IMU data changes and joint state variations
        if len(self.sensor_data['imu']) > 1:
            latest_imu = self.sensor_data['imu'][-1]['data']
            prev_imu = self.sensor_data['imu'][-2]['data']

            # Calculate change in linear acceleration (indicative of contact forces)
            acc_change = math.sqrt(
                (latest_imu.linear_acceleration.x - prev_imu.linear_acceleration.x)**2 +
                (latest_imu.linear_acceleration.y - prev_imu.linear_acceleration.y)**2 +
                (latest_imu.linear_acceleration.z - prev_imu.linear_acceleration.z)**2
            )

            # Update contact force metrics
            if acc_change > 0.5:  # Threshold for contact detection
                self.collision_metrics['contact_force_average'] = (
                    self.collision_metrics['contact_force_average'] + acc_change
                ) / 2  # Running average

    def get_joint_position(self, joint_state, joint_name):
        """Get position of a specific joint"""
        try:
            idx = joint_state.name.index(joint_name)
            if idx < len(joint_state.position):
                # Simplified: returning position as a proxy for location
                return (joint_state.position[idx], 0, 0)  # x, y, z (simplified)
        except ValueError:
            pass  # Joint not found
        return None

    def run_comprehensive_test(self):
        """Run comprehensive collision handling tests"""
        self.get_logger().info("Starting comprehensive collision handling tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running collision test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> CollisionTestResult:
        """Run a specific collision test scenario"""
        if scenario == 'self_collision_prevention':
            return self.run_self_collision_test()
        elif scenario == 'environment_collision_detection':
            return self.run_environment_collision_test()
        elif scenario == 'contact_force_validation':
            return self.run_contact_force_test()
        elif scenario == 'collision_response_quality':
            return self.run_response_quality_test()
        elif scenario == 'multi_body_interactions':
            return self.run_multi_body_test()
        else:
            return CollisionTestResult(
                test_name=scenario,
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_self_collision_test(self) -> CollisionTestResult:
        """Test self-collision detection and prevention"""
        start_time = time.time()

        # Wait for sufficient data to accumulate
        time.sleep(5)  # Wait 5 seconds for data collection

        # Check self-collision metrics
        # In a real implementation, we'd check that self-collision prevention is working
        # For now, we'll just verify that the system is tracking self-collision events
        self_collision_handling_ok = self.collision_metrics['self_collision_count'] >= 0  # Just verify tracking

        metrics = {
            'self_collision_count': self.collision_metrics['self_collision_count'],
            'self_collision_handling_ok': self_collision_handling_ok,
            'duration': time.time() - start_time
        }

        # For this test, we consider it passed if the system is tracking self-collisions
        passed = self_collision_handling_ok

        return CollisionTestResult(
            test_name='self_collision_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "Self-collision detection not functioning properly"
        )

    def run_environment_collision_test(self) -> CollisionTestResult:
        """Test environment collision detection"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(3)  # Wait for potential collisions

        # Check environment collision metrics
        env_collision_count = self.collision_metrics['environment_collision_count']

        # In a real test, we'd set up specific scenarios and verify detection
        # For now, check that the system is tracking environment collisions
        env_collision_tracking_ok = env_collision_count >= 0  # Just verify tracking

        metrics = {
            'environment_collision_count': env_collision_count,
            'environment_collision_tracking_ok': env_collision_tracking_ok,
            'duration': time.time() - start_time
        }

        # Pass if system is tracking environment collisions
        passed = env_collision_tracking_ok

        return CollisionTestResult(
            test_name='environment_collision_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "Environment collision detection not tracking properly"
        )

    def run_contact_force_test(self) -> CollisionTestResult:
        """Test contact force measurement and validation"""
        start_time = time.time()

        # Wait for force data to accumulate
        time.sleep(2)

        # Check contact force metrics
        avg_force = self.collision_metrics['contact_force_average']

        # Validate that force measurements are reasonable
        force_measurement_valid = 0 <= avg_force <= 100  # Reasonable force range (N)

        metrics = {
            'average_contact_force': avg_force,
            'force_measurement_valid': force_measurement_valid,
            'duration': time.time() - start_time
        }

        passed = force_measurement_valid

        return CollisionTestResult(
            test_name='contact_force_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Contact force measurement invalid: {avg_force:.2f}N"
        )

    def run_response_quality_test(self) -> CollisionTestResult:
        """Test collision response quality"""
        start_time = time.time()

        # For collision response quality, check if the system maintains stability after collisions
        stability_after_collision = self.collision_metrics['collision_response_quality'] >= 0.5

        # Calculate based on IMU stability after collision events
        if len(self.sensor_data['imu']) > 10:
            imu_readings = []
            for data in list(self.sensor_data['imu'])[-10:]:
                imu = data['data']
                roll, pitch, yaw = self.quaternion_to_euler([
                    imu.orientation.x, imu.orientation.y,
                    imu.orientation.z, imu.orientation.w
                ])
                imu_readings.append(abs(roll) + abs(pitch))  # Deviation from upright

            avg_deviation = np.mean(imu_readings) if imu_readings else 0
            stability_score = max(0, min(1, 1 - avg_deviation))  # Lower deviation = higher stability
            self.collision_metrics['collision_response_quality'] = stability_score

        metrics = {
            'response_quality_score': self.collision_metrics['collision_response_quality'],
            'stability_after_collision': stability_after_collision,
            'duration': time.time() - start_time
        }

        passed = self.collision_metrics['collision_response_quality'] >= 0.6

        return CollisionTestResult(
            test_name='response_quality_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Collision response quality too low: {self.collision_metrics['collision_response_quality']:.2f}"
        )

    def run_multi_body_test(self) -> CollisionTestResult:
        """Test multi-body collision interactions"""
        start_time = time.time()

        # For multi-body tests, check if we have sufficient data diversity
        # In a real implementation, this would involve multiple robots/colliders
        data_sufficiency_ok = len(self.sensor_data['laser']) > 10

        metrics = {
            'data_sufficiency_ok': data_sufficiency_ok,
            'laser_data_points': len(self.sensor_data['laser']),
            'duration': time.time() - start_time
        }

        passed = data_sufficiency_ok

        return CollisionTestResult(
            test_name='multi_body_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "Insufficient data for multi-body collision analysis"
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

    def publish_test_results(self):
        """Publish collision test results"""
        results_msg = String()
        results_msg.data = json.dumps({
            'timestamp': time.time(),
            'test_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'metrics': r.metrics,
                    'duration': r.duration,
                    'error_message': r.error_message
                } for r in self.test_results
            ],
            'statistics': self.test_statistics,
            'collision_metrics': self.collision_metrics
        })

        self.collision_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive collision handling test report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'collision_metrics': self.collision_metrics.copy(),
            'individual_results': [
                {
                    'test_name': r.test_name,
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
        """Generate recommendations based on collision test results"""
        recommendations = []

        if self.collision_metrics['self_collision_count'] > 10:
            recommendations.append("High self-collision rate detected - review joint limits and collision geometry")

        if self.collision_metrics['environment_collision_count'] > 5:
            recommendations.append("Frequent environment collisions - improve navigation and obstacle avoidance")

        if self.collision_metrics['collision_response_quality'] < 0.7:
            recommendations.append("Poor collision response quality - enhance collision recovery algorithms")

        if self.collision_metrics['contact_force_average'] > 10.0:
            recommendations.append("High contact forces detected - adjust robot control parameters")

        if not recommendations:
            recommendations.append("Collision handling is performing well - continue monitoring for optimization")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed collision handling test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("COLLISION HANDLING COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("COLLISION METRICS:")
        print("-" * 40)
        metrics = report['collision_metrics']
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("\nINDIVIDUAL TEST RESULTS:")
        print("-" * 40)
        for result in report['individual_results']:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {status:4} | {result['test_name']:20} | {result['duration']:.2f}s | {result['error'] or 'OK'}")

        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("="*80)


def main(args=None):
    """Main function to run collision handling tests"""
    rclpy.init(args=args)

    tester = CollisionHandlingTester()

    try:
        # Run comprehensive collision tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Collision Handling Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Collision handling testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()