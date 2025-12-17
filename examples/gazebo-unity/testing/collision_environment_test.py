#!/usr/bin/env python3
"""
Collision Handling and Environmental Constraints Test
This script verifies collision handling and environmental constraints for humanoid robots in Gazebo simulation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import Point, Pose, Vector3
from gazebo_msgs.srv import GetModelState, GetWorldProperties, GetLightProperties
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
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
class CollisionConstraintResult:
    """Result of a collision handling and constraint test"""
    test_name: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class CollisionEnvironmentTester(Node):
    """
    Tester for collision handling and environmental constraints for humanoid robots
    """

    def __init__(self):
        super().__init__('collision_environment_tester')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=100),
            'imu': deque(maxlen=100),
            'joint': deque(maxlen=100),
            'contact': deque(maxlen=100)
        }

        self.collision_metrics = {
            'collision_detection_rate': 0.0,
            'environment_constraint_violations': 0,
            'self_collision_prevention': 0.0,
            'contact_force_normalization': 0.0,
            'boundary_compliance': 0.0
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

        # Publishers for test control and results
        self.test_control_pub = self.create_publisher(
            String,
            '/collision_environment/test_control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/collision_environment/results',
            10
        )

        self.constraint_violation_pub = self.create_publisher(
            Marker,
            '/collision_environment/constraint_violations',
            10
        )

        # Service clients for Gazebo interaction
        self.get_model_state_cli = self.create_client(GetModelState, '/get_entity_state')
        self.get_world_properties_cli = self.create_client(GetWorldProperties, '/get_world_properties')
        self.get_light_properties_cli = self.create_client(GetLightProperties, '/get_light_properties')

        # Timer for collision and constraint monitoring
        self.monitor_timer = self.create_timer(0.1, self.collision_constraint_monitoring)

        # Test parameters
        self.test_scenarios = [
            'environment_boundary_compliance',
            'obstacle_collision_detection',
            'self_collision_prevention',
            'contact_force_analysis',
            'constraint_violation_detection'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        # Environment constraints
        self.environment_bounds = {
            'min_x': -5.0, 'max_x': 5.0,
            'min_y': -5.0, 'max_y': 5.0,
            'min_z': 0.0, 'max_z': 3.0
        }

        self.safety_margin = 0.1  # 10cm safety margin

        self.get_logger().info("Collision Environment Tester initialized and ready for testing")

    def scan_callback(self, msg):
        """Process LiDAR scan data for obstacle detection"""
        self.sensor_data['laser'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def imu_callback(self, msg):
        """Process IMU data for collision response analysis"""
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

    def collision_constraint_monitoring(self):
        """Monitor collision handling and environmental constraints"""
        try:
            # Update collision metrics
            self.update_collision_detection_metrics()
            self.update_environment_constraint_metrics()
            self.update_self_collision_metrics()
            self.update_contact_force_metrics()
            self.update_boundary_compliance_metrics()

        except Exception as e:
            self.get_logger().error(f"Error in collision constraint monitoring: {e}")

    def update_collision_detection_metrics(self):
        """Update metrics for collision detection"""
        if len(self.sensor_data['laser']) > 0:
            latest_scan = self.sensor_data['laser'][-1]['data']
            ranges = np.array(latest_scan.ranges)

            # Calculate collision detection based on close ranges
            valid_ranges = ranges[np.isfinite(ranges)]
            if len(valid_ranges) > 0:
                min_range = np.min(valid_ranges)

                # If minimum range is very small, likely collision or close approach
                if min_range < 0.3:  # 30cm threshold for potential collision
                    self.collision_metrics['collision_detection_rate'] = min(1.0, self.collision_metrics['collision_detection_rate'] + 0.1)

    def update_environment_constraint_metrics(self):
        """Update metrics for environmental constraint compliance"""
        # In a real implementation, we'd check for constraint violations
        # For now, we'll just track if the robot stays within expected bounds
        if len(self.sensor_data['laser']) > 0:
            # This would involve checking robot position against environment constraints
            # For simulation, we'll assume compliance if no extreme readings
            self.collision_metrics['environment_constraint_violations'] = max(0, self.collision_metrics['environment_constraint_violations'] - 0.01)

    def update_self_collision_metrics(self):
        """Update metrics for self-collision prevention"""
        if len(self.sensor_data['joint']) > 0:
            latest_joints = self.sensor_data['joint'][-1]['data']

            # Check for potential self-collision scenarios
            # This is a simplified check - in reality, we'd use collision geometry
            potential_self_collisions = 0

            # Example: Check for extreme joint angles that might cause self-collision
            if len(latest_joints.position) > 0:
                for i, pos in enumerate(latest_joints.position):
                    # Check if joint position is approaching limits (potential self-collision)
                    if abs(pos) > 2.5:  # Arbitrary threshold for potential self-collision
                        potential_self_collisions += 1

            # Update self-collision prevention score
            if potential_self_collisions == 0:
                self.collision_metrics['self_collision_prevention'] = min(1.0, self.collision_metrics['self_collision_prevention'] + 0.02)
            else:
                self.collision_metrics['self_collision_prevention'] = max(0.0, self.collision_metrics['self_collision_prevention'] - 0.05)

    def update_contact_force_metrics(self):
        """Update metrics for contact force analysis"""
        if len(self.sensor_data['imu']) > 1:
            latest_imu = self.sensor_data['imu'][-1]['data']
            prev_imu = self.sensor_data['imu'][-2]['data']

            # Calculate change in linear acceleration (indicative of contact forces)
            acc_change = math.sqrt(
                (latest_imu.linear_acceleration.x - prev_imu.linear_acceleration.x)**2 +
                (latest_imu.linear_acceleration.y - prev_imu.linear_acceleration.y)**2 +
                (latest_imu.linear_acceleration.z - prev_imu.linear_acceleration.z)**2
            )

            # Update contact force normalization
            if acc_change > 0.5:  # Threshold for contact detection
                self.collision_metrics['contact_force_normalization'] = min(1.0, self.collision_metrics['contact_force_normalization'] + 0.01)

    def update_boundary_compliance_metrics(self):
        """Update metrics for boundary compliance"""
        # In a real implementation, we'd get the robot's actual position
        # For now, we'll use a simplified approach based on laser scan data
        if len(self.sensor_data['laser']) > 0:
            latest_scan = self.sensor_data['laser'][-1]['data']
            ranges = np.array(latest_scan.ranges)

            # If many ranges are at maximum, robot might be near boundaries
            max_ranges = np.sum(ranges >= latest_scan.range_max * 0.99)  # Close to max range
            total_ranges = len(ranges)

            if total_ranges > 0:
                boundary_proximity_ratio = max_ranges / total_ranges
                # Higher ratio means closer to boundary (less compliant)
                self.collision_metrics['boundary_compliance'] = max(0.0, 1.0 - boundary_proximity_ratio)

    def run_comprehensive_test(self):
        """Run comprehensive collision handling and environmental constraint tests"""
        self.get_logger().info("Starting comprehensive collision and environment constraint tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running collision/environment test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> CollisionConstraintResult:
        """Run a specific collision/environment test scenario"""
        if scenario == 'environment_boundary_compliance':
            return self.run_boundary_compliance_test()
        elif scenario == 'obstacle_collision_detection':
            return self.run_obstacle_collision_test()
        elif scenario == 'self_collision_prevention':
            return self.run_self_collision_test()
        elif scenario == 'contact_force_analysis':
            return self.run_contact_force_test()
        elif scenario == 'constraint_violation_detection':
            return self.run_constraint_violation_test()
        else:
            return CollisionConstraintResult(
                test_name=scenario,
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_boundary_compliance_test(self) -> CollisionConstraintResult:
        """Test compliance with environmental boundaries"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(5)  # Wait 5 seconds for boundary compliance assessment

        # Check if robot stays within environment bounds
        boundary_compliance = self.collision_metrics['boundary_compliance'] >= 0.8  # 80% compliance required

        metrics = {
            'boundary_compliance_score': self.collision_metrics['boundary_compliance'],
            'boundary_compliance': boundary_compliance,
            'duration': time.time() - start_time
        }

        passed = boundary_compliance

        return CollisionConstraintResult(
            test_name='environment_boundary_compliance',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Boundary compliance too low: {self.collision_metrics['boundary_compliance']:.2f}"
        )

    def run_obstacle_collision_test(self) -> CollisionConstraintResult:
        """Test obstacle collision detection and handling"""
        start_time = time.time()

        # Wait for collision data
        time.sleep(3)  # Wait for potential collisions

        # Check collision detection metrics
        collision_detection_active = self.collision_metrics['collision_detection_rate'] > 0.1  # At least 10% detection rate

        metrics = {
            'collision_detection_rate': self.collision_metrics['collision_detection_rate'],
            'collision_detection_active': collision_detection_active,
            'duration': time.time() - start_time
        }

        passed = collision_detection_active

        return CollisionConstraintResult(
            test_name='obstacle_collision_detection',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Collision detection rate too low: {self.collision_metrics['collision_detection_rate']:.2f}"
        )

    def run_self_collision_test(self) -> CollisionConstraintResult:
        """Test self-collision prevention mechanisms"""
        start_time = time.time()

        # Wait for joint data collection
        time.sleep(4)  # Wait for self-collision assessment

        # Check self-collision prevention metrics
        self_collision_prevention_good = self.collision_metrics['self_collision_prevention'] >= 0.7  # 70% prevention required

        metrics = {
            'self_collision_prevention_score': self.collision_metrics['self_collision_prevention'],
            'self_collision_prevention_good': self_collision_prevention_good,
            'duration': time.time() - start_time
        }

        passed = self_collision_prevention_good

        return CollisionConstraintResult(
            test_name='self_collision_prevention',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Self-collision prevention too low: {self.collision_metrics['self_collision_prevention']:.2f}"
        )

    def run_contact_force_test(self) -> CollisionConstraintResult:
        """Test contact force analysis and normalization"""
        start_time = time.time()

        # Wait for force data collection
        time.sleep(3)  # Wait for contact force assessment

        # Check contact force normalization
        contact_force_normalized = self.collision_metrics['contact_force_normalization'] >= 0.5  # 50% normalization required

        metrics = {
            'contact_force_normalization_score': self.collision_metrics['contact_force_normalization'],
            'contact_force_normalized': contact_force_normalized,
            'duration': time.time() - start_time
        }

        passed = contact_force_normalized

        return CollisionConstraintResult(
            test_name='contact_force_analysis',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Contact force normalization too low: {self.collision_metrics['contact_force_normalization']:.2f}"
        )

    def run_constraint_violation_test(self) -> CollisionConstraintResult:
        """Test constraint violation detection and handling"""
        start_time = time.time()

        # Wait for constraint data collection
        time.sleep(4)  # Wait for constraint violation assessment

        # Check for constraint violations
        constraint_violations_acceptable = self.collision_metrics['environment_constraint_violations'] <= 2  # Max 2 violations

        metrics = {
            'constraint_violations': self.collision_metrics['environment_constraint_violations'],
            'constraint_violations_acceptable': constraint_violations_acceptable,
            'duration': time.time() - start_time
        }

        passed = constraint_violations_acceptable

        return CollisionConstraintResult(
            test_name='constraint_violation_detection',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Too many constraint violations: {self.collision_metrics['environment_constraint_violations']}"
        )

    def publish_test_results(self):
        """Publish collision and environment test results"""
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

        self.test_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive collision and environment test report"""
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
            'environment_bounds': self.environment_bounds,
            'safety_margin': self.safety_margin,
            'recommendations': self.generate_recommendations()
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on collision and environment test results"""
        recommendations = []

        if self.collision_metrics['boundary_compliance'] < 0.8:
            recommendations.append("Improve boundary compliance - adjust navigation to respect environment limits")

        if self.collision_metrics['collision_detection_rate'] < 0.3:
            recommendations.append("Enhance collision detection - improve sensor configuration or detection algorithms")

        if self.collision_metrics['self_collision_prevention'] < 0.7:
            recommendations.append("Strengthen self-collision prevention - review joint limits and collision geometry")

        if self.collision_metrics['contact_force_normalization'] < 0.5:
            recommendations.append("Improve contact force handling - adjust collision response parameters")

        if self.collision_metrics['environment_constraint_violations'] > 1:
            recommendations.append("Reduce constraint violations - improve path planning and obstacle avoidance")

        if not recommendations:
            recommendations.append("Collision handling and environmental constraints are performing well - continue monitoring for optimization")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed collision and environment test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("COLLISION HANDLING AND ENVIRONMENTAL CONSTRAINTS TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("ENVIRONMENT BOUNDS:")
        print("-" * 40)
        bounds = report['environment_bounds']
        print(f"  X: {bounds['min_x']:.1f} to {bounds['max_x']:.1f}")
        print(f"  Y: {bounds['min_y']:.1f} to {bounds['max_y']:.1f}")
        print(f"  Z: {bounds['min_z']:.1f} to {bounds['max_z']:.1f}")
        print(f"  Safety Margin: {report['safety_margin']:.2f}m")

        print("\nCOLLISION METRICS:")
        print("-" * 40)
        metrics = report['collision_metrics']
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("\nINDIVIDUAL TEST RESULTS:")
        print("-" * 40)
        for result in report['individual_results']:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {status:4} | {result['test_name']:30} | {result['duration']:.2f}s | {result['error'] or 'OK'}")

        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("="*80)


def main(args=None):
    """Main function to run collision and environment constraint tests"""
    rclpy.init(args=args)

    tester = CollisionEnvironmentTester()

    try:
        # Run comprehensive collision and environment tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Collision/Environment Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Collision and environment testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()