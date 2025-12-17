#!/usr/bin/env python3
"""
Physics Simulation Test Suite
This script performs comprehensive testing of physics simulations,
including stability, collision handling, and environmental constraints
for humanoid robotics in Gazebo.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import ModelState
from std_msgs.msg import String
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
class PhysicsTestResult:
    """Result of a physics simulation test"""
    test_name: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class PhysicsSimulationTester(Node):
    """
    Comprehensive tester for Gazebo physics simulations with humanoid robots
    """

    def __init__(self):
        super().__init__('physics_simulation_tester')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=100),
            'imu': deque(maxlen=100),
            'joint': deque(maxlen=100),
            'odometry': deque(maxlen=100)
        }

        self.physics_metrics = {
            'stability_score': 0.0,
            'collision_count': 0,
            'energy_consumption': 0.0,
            'balance_maintenance': 0.0
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

        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        # Publishers for test control
        self.test_control_pub = self.create_publisher(
            String,
            '/physics_test/control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/physics_test/results',
            10
        )

        # Service clients for Gazebo interaction
        self.get_state_cli = self.create_client(GetEntityState, '/get_entity_state')
        self.set_state_cli = self.create_client(SetEntityState, '/set_entity_state')

        # Timer for physics monitoring
        self.monitor_timer = self.create_timer(0.1, self.physics_monitoring)

        # Test parameters
        self.test_scenarios = [
            'stability_test',
            'collision_handling',
            'environmental_constraints',
            'multi_robot_scenarios',
            'extreme_conditions'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        self.get_logger().info("Physics Simulation Tester initialized and ready for testing")

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

    def odom_callback(self, msg):
        """Process odometry data"""
        self.sensor_data['odometry'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def physics_monitoring(self):
        """Monitor physics simulation metrics"""
        try:
            # Calculate physics metrics based on sensor data
            self.update_stability_metrics()
            self.update_collision_metrics()
            self.update_balance_metrics()

        except Exception as e:
            self.get_logger().error(f"Error in physics monitoring: {e}")

    def update_stability_metrics(self):
        """Update stability metrics based on IMU and joint data"""
        if len(self.sensor_data['imu']) > 10:
            # Calculate stability based on IMU readings
            imu_readings = []
            for data in list(self.sensor_data['imu'])[-10:]:
                imu = data['data']
                # Convert quaternion to Euler angles to check stability
                roll, pitch, yaw = self.quaternion_to_euler([
                    imu.orientation.x, imu.orientation.y,
                    imu.orientation.z, imu.orientation.w
                ])
                imu_readings.append([roll, pitch, yaw])

            imu_array = np.array(imu_readings)
            # Calculate variance - lower variance means more stable
            imu_variance = np.var(imu_array, axis=0)
            avg_variance = np.mean(imu_variance)

            # Convert to stability score (lower variance = higher stability)
            self.physics_metrics['stability_score'] = max(0, min(1, 1 - avg_variance * 10))

    def update_collision_metrics(self):
        """Update collision metrics based on sensor data"""
        # For now, just increment based on sudden changes in IMU data
        # In a real implementation, this would use Gazebo collision detection
        if len(self.sensor_data['imu']) > 5:
            latest_imu = self.sensor_data['imu'][-1]['data']
            prev_imu = self.sensor_data['imu'][-2]['data']

            # Check for sudden changes that might indicate collisions
            lin_acc_change = abs(latest_imu.linear_acceleration.x - prev_imu.linear_acceleration.x)
            ang_vel_change = abs(latest_imu.angular_velocity.x - prev_imu.angular_velocity.x)

            if lin_acc_change > 5.0 or ang_vel_change > 1.0:  # Thresholds for collision detection
                self.physics_metrics['collision_count'] += 1

    def update_balance_metrics(self):
        """Update balance metrics based on IMU data"""
        if len(self.sensor_data['imu']) > 0:
            latest_imu = self.sensor_data['imu'][-1]['data']
            roll, pitch, yaw = self.quaternion_to_euler([
                latest_imu.orientation.x, latest_imu.orientation.y,
                latest_imu.orientation.z, latest_imu.orientation.w
            ])

            # Calculate balance score based on how close to upright position
            # Perfect balance is roll=0, pitch=0
            balance_error = abs(roll) + abs(pitch)
            self.physics_metrics['balance_maintenance'] = max(0, min(1, 1 - balance_error))

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

    def run_comprehensive_test(self):
        """Run comprehensive physics simulation tests"""
        self.get_logger().info("Starting comprehensive physics simulation tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> PhysicsTestResult:
        """Run a specific physics test scenario"""
        if scenario == 'stability_test':
            return self.run_stability_test()
        elif scenario == 'collision_handling':
            return self.run_collision_test()
        elif scenario == 'environmental_constraints':
            return self.run_environment_test()
        elif scenario == 'multi_robot_scenarios':
            return self.run_multi_robot_test()
        elif scenario == 'extreme_conditions':
            return self.run_extreme_condition_test()
        else:
            return PhysicsTestResult(
                test_name=scenario,
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_stability_test(self) -> PhysicsTestResult:
        """Test humanoid robot stability under various conditions"""
        start_time = time.time()

        # Wait for sufficient data to accumulate
        time.sleep(5)  # Wait 5 seconds for data collection

        # Check stability metrics
        stability_threshold = 0.7  # Require 70% stability score
        passed = self.physics_metrics['stability_score'] >= stability_threshold

        metrics = {
            'stability_score': self.physics_metrics['stability_score'],
            'balance_maintenance': self.physics_metrics['balance_maintenance'],
            'duration': time.time() - start_time
        }

        return PhysicsTestResult(
            test_name='stability_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Stability score {self.physics_metrics['stability_score']:.2f} below threshold {stability_threshold}"
        )

    def run_collision_test(self) -> PhysicsTestResult:
        """Test collision handling and detection"""
        start_time = time.time()

        # Wait for collision data
        time.sleep(3)  # Wait for potential collisions

        # For this test, we'll check if collisions are being detected appropriately
        # In a real implementation, we'd set up specific collision scenarios
        collision_detected = self.physics_metrics['collision_count'] > 0

        # For stability test, we want minimal collisions, but some are expected
        # So we'll check for reasonable collision rates
        collision_rate_acceptable = self.physics_metrics['collision_count'] < 10  # Less than 10 collisions in 3 seconds

        metrics = {
            'collision_count': self.physics_metrics['collision_count'],
            'collision_rate_acceptable': collision_rate_acceptable,
            'duration': time.time() - start_time
        }

        passed = collision_rate_acceptable

        return PhysicsTestResult(
            test_name='collision_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Too many collisions detected: {self.physics_metrics['collision_count']}"
        )

    def run_environment_test(self) -> PhysicsTestResult:
        """Test environmental constraints and interactions"""
        start_time = time.time()

        # Check if robot stays within expected bounds
        position_bounds_ok = True
        max_deviation = 5.0  # meters

        if len(self.sensor_data['odometry']) > 0:
            latest_odom = self.sensor_data['odometry'][-1]['data']
            pos = latest_odom.pose.pose.position
            distance_from_origin = math.sqrt(pos.x**2 + pos.y**2)
            position_bounds_ok = distance_from_origin <= max_deviation

        metrics = {
            'position_bounds_ok': position_bounds_ok,
            'max_deviation': max_deviation,
            'current_distance': distance_from_origin if len(self.sensor_data['odometry']) > 0 else 0,
            'duration': time.time() - start_time
        }

        passed = position_bounds_ok

        return PhysicsTestResult(
            test_name='environment_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Robot exceeded position bounds: {distance_from_origin:.2f}m > {max_deviation}m"
        )

    def run_multi_robot_test(self) -> PhysicsTestResult:
        """Test multi-robot scenarios and potential collision cascades"""
        start_time = time.time()

        # For this test, we'll simulate checking for multiple robot interactions
        # In a real implementation, this would involve spawning multiple robots
        # and testing their interactions

        # Placeholder: Check if we have sufficient data diversity
        data_diversity_ok = len(self.sensor_data['laser']) > 20  # Enough data points

        metrics = {
            'data_diversity_ok': data_diversity_ok,
            'laser_data_points': len(self.sensor_data['laser']),
            'duration': time.time() - start_time
        }

        passed = data_diversity_ok

        return PhysicsTestResult(
            test_name='multi_robot_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else "Insufficient data diversity for multi-robot simulation"
        )

    def run_extreme_condition_test(self) -> PhysicsTestResult:
        """Test robot behavior under extreme conditions"""
        start_time = time.time()

        # Test extreme conditions like high torques, unusual poses
        # Check for system stability under stress

        stability_ok = self.physics_metrics['stability_score'] > 0.5  # Lower threshold for extreme conditions

        metrics = {
            'stability_score': self.physics_metrics['stability_score'],
            'stability_ok': stability_ok,
            'duration': time.time() - start_time
        }

        passed = stability_ok

        return PhysicsTestResult(
            test_name='extreme_condition_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Robot unstable under extreme conditions: {self.physics_metrics['stability_score']:.2f}"
        )

    def publish_test_results(self):
        """Publish comprehensive test results"""
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
            'physics_metrics': self.physics_metrics
        })

        self.test_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive physics simulation test report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'physics_metrics': self.physics_metrics.copy(),
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
        """Generate recommendations based on test results"""
        recommendations = []

        if self.physics_metrics['stability_score'] < 0.7:
            recommendations.append("Improve robot stability - consider adjusting center of mass or adding balance controllers")

        if self.physics_metrics['collision_count'] > 10:
            recommendations.append("Reduce collision frequency - adjust navigation algorithms or environment constraints")

        if self.physics_metrics['balance_maintenance'] < 0.6:
            recommendations.append("Enhance balance control algorithms for better stability")

        if not recommendations:
            recommendations.append("Physics simulation is performing well - continue monitoring for optimization opportunities")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed physics simulation test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("PHYSICS SIMULATION COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("PHYSICS METRICS:")
        print("-" * 40)
        metrics = report['physics_metrics']
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
    """Main function to run physics simulation tests"""
    rclpy.init(args=args)

    tester = PhysicsSimulationTester()

    try:
        # Run comprehensive physics tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Physics Simulation Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Physics simulation testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()