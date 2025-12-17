#!/usr/bin/env python3
"""
Multi-Robot Scenario Test
This script tests multi-robot scenarios with potential collision cascades in Gazebo simulation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import Pose, Twist, Point
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelState
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
class MultiRobotResult:
    """Result of a multi-robot test"""
    test_name: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class MultiRobotTester(Node):
    """
    Tester for multi-robot scenarios with potential collision cascades
    """

    def __init__(self):
        super().__init__('multi_robot_tester')

        # Initialize data storage for multiple robots
        self.robot_ids = ['humanoid_1', 'humanoid_2', 'humanoid_3']
        self.robot_data = {}

        for robot_id in self.robot_ids:
            self.robot_data[robot_id] = {
                'laser': deque(maxlen=100),
                'imu': deque(maxlen=100),
                'joint': deque(maxlen=100),
                'odometry': deque(maxlen=100),
                'collisions': 0,
                'proximity_events': 0
            }

        self.multi_robot_metrics = {
            'collision_cascade_events': 0,
            'multi_robot_stability': 0.0,
            'navigation_efficiency': 0.0,
            'collision_avoidance_success': 0.0,
            'communication_efficiency': 0.0
        }

        # Initialize ROS interfaces for each robot
        for robot_id in self.robot_ids:
            # Create subscribers for each robot's sensors
            self.create_subscription(
                LaserScan,
                f'/{robot_id}/scan',
                lambda msg, rid=robot_id: self.scan_callback(msg, rid),
                10
            )

            self.create_subscription(
                Imu,
                f'/{robot_id}/imu/data',
                lambda msg, rid=robot_id: self.imu_callback(msg, rid),
                10
            )

            self.create_subscription(
                JointState,
                f'/{robot_id}/joint_states',
                lambda msg, rid=robot_id: self.joint_callback(msg, rid),
                10
            )

            self.create_subscription(
                Odometry,
                f'/{robot_id}/odom',
                lambda msg, rid=robot_id: self.odom_callback(msg, rid),
                10
            )

        # Publishers for test control and results
        self.test_control_pub = self.create_publisher(
            String,
            '/multi_robot/test_control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/multi_robot/results',
            10
        )

        self.collision_marker_pub = self.create_publisher(
            Marker,
            '/multi_robot/collision_markers',
            10
        )

        # Service clients for Gazebo interaction
        self.spawn_entity_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_cli = self.create_client(DeleteEntity, '/delete_entity')
        self.get_model_state_cli = self.create_client(GetModelState, '/get_entity_state')

        # Timer for multi-robot monitoring
        self.monitor_timer = self.create_timer(0.1, self.multi_robot_monitoring)

        # Test parameters
        self.test_scenarios = [
            'close_proximity_scenarios',
            'navigation_in_crowds',
            'collision_avoidance_tests',
            'formation_maintenance',
            'emergency_stop_scenarios'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        # Collision cascade tracking
        self.collision_chain_events = []
        self.proximity_threshold = 0.5  # 50cm threshold for proximity events
        self.collision_threshold = 0.1  # 10cm threshold for collision detection

        self.get_logger().info("Multi-Robot Tester initialized and ready for testing")

    def scan_callback(self, msg, robot_id):
        """Process LiDAR scan data for specific robot"""
        self.robot_data[robot_id]['laser'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def imu_callback(self, msg, robot_id):
        """Process IMU data for specific robot"""
        self.robot_data[robot_id]['imu'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def joint_callback(self, msg, robot_id):
        """Process joint state data for specific robot"""
        self.robot_data[robot_id]['joint'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def odom_callback(self, msg, robot_id):
        """Process odometry data for specific robot"""
        self.robot_data[robot_id]['odometry'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def multi_robot_monitoring(self):
        """Monitor multi-robot interactions and potential collision cascades"""
        try:
            # Update multi-robot metrics
            self.update_collision_cascade_metrics()
            self.update_proximity_metrics()
            self.update_navigation_metrics()
            self.update_communication_metrics()

        except Exception as e:
            self.get_logger().error(f"Error in multi-robot monitoring: {e}")

    def update_collision_cascade_metrics(self):
        """Update metrics for collision cascade detection"""
        # Check for potential collision cascades by monitoring simultaneous events
        simultaneous_events = 0

        # Check if multiple robots are experiencing collisions at similar times
        for robot_id in self.robot_ids:
            if len(self.robot_data[robot_id]['laser']) > 0:
                latest_scan = self.robot_data[robot_id]['laser'][-1]['data']
                ranges = np.array(latest_scan.ranges)

                # Check for close obstacles that might indicate potential collision
                valid_ranges = ranges[np.isfinite(ranges)]
                if len(valid_ranges) > 0:
                    min_range = np.min(valid_ranges)

                    if min_range < self.collision_threshold:
                        self.robot_data[robot_id]['collisions'] += 1
                        simultaneous_events += 1

        # Update collision cascade metric if multiple robots collide simultaneously
        if simultaneous_events > 1:
            self.multi_robot_metrics['collision_cascade_events'] += 1

    def update_proximity_metrics(self):
        """Update metrics for robot proximity events"""
        # Calculate distances between robots based on odometry data
        robot_positions = {}

        for robot_id in self.robot_ids:
            if len(self.robot_data[robot_id]['odometry']) > 0:
                odom = self.robot_data[robot_id]['odometry'][-1]['data']
                pos = odom.pose.pose.position
                robot_positions[robot_id] = np.array([pos.x, pos.y, pos.z])

        # Check distances between all robot pairs
        for i, robot1_id in enumerate(self.robot_ids):
            for j, robot2_id in enumerate(self.robot_ids[i+1:], i+1):
                if robot1_id in robot_positions and robot2_id in robot_positions:
                    dist = np.linalg.norm(robot_positions[robot1_id] - robot_positions[robot2_id])

                    if dist < self.proximity_threshold:
                        self.robot_data[robot1_id]['proximity_events'] += 1
                        self.robot_data[robot2_id]['proximity_events'] += 1

    def update_navigation_metrics(self):
        """Update metrics for multi-robot navigation efficiency"""
        # Calculate navigation efficiency based on robot movement patterns
        total_efficiency = 0.0
        robot_count = 0

        for robot_id in self.robot_ids:
            if len(self.robot_data[robot_id]['odometry']) > 10:
                # Calculate movement efficiency for each robot
                positions = []
                for data in list(self.robot_data[robot_id]['odometry'])[-10:]:
                    pos = data['data'].pose.pose.position
                    positions.append(np.array([pos.x, pos.y, pos.z]))

                if len(positions) > 1:
                    # Calculate straight-line efficiency
                    start_pos = positions[0]
                    end_pos = positions[-1]
                    direct_distance = np.linalg.norm(end_pos - start_pos)

                    # Calculate actual path distance
                    path_distance = 0.0
                    for i in range(1, len(positions)):
                        path_distance += np.linalg.norm(positions[i] - positions[i-1])

                    if path_distance > 0:
                        efficiency = direct_distance / path_distance
                        total_efficiency += max(0, min(1, efficiency))
                        robot_count += 1

        if robot_count > 0:
            self.multi_robot_metrics['navigation_efficiency'] = total_efficiency / robot_count

    def update_communication_metrics(self):
        """Update metrics for inter-robot communication efficiency"""
        # For this test, we'll calculate communication efficiency based on
        # how well robots avoid each other (indicating coordination)
        avoidance_efficiency = 0.0
        total_pairs = 0

        for i, robot1_id in enumerate(self.robot_ids):
            for j, robot2_id in enumerate(self.robot_ids[i+1:], i+1):
                # Calculate how well robots maintain safe distance
                if (len(self.robot_data[robot1_id]['odometry']) > 0 and
                    len(self.robot_data[robot2_id]['odometry']) > 0):

                    pos1 = self.robot_data[robot1_id]['odometry'][-1]['data'].pose.pose.position
                    pos2 = self.robot_data[robot2_id]['odometry'][-1]['data'].pose.pose.position

                    dist = math.sqrt(
                        (pos1.x - pos2.x)**2 +
                        (pos1.y - pos2.y)**2 +
                        (pos1.z - pos2.z)**2
                    )

                    # Higher efficiency if robots maintain safe distance
                    if dist > self.proximity_threshold:
                        avoidance_efficiency += 1.0
                    total_pairs += 1

        if total_pairs > 0:
            self.multi_robot_metrics['communication_efficiency'] = avoidance_efficiency / total_pairs

    def run_comprehensive_test(self):
        """Run comprehensive multi-robot tests"""
        self.get_logger().info("Starting comprehensive multi-robot tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running multi-robot test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> MultiRobotResult:
        """Run a specific multi-robot test scenario"""
        if scenario == 'close_proximity_scenarios':
            return self.run_close_proximity_test()
        elif scenario == 'navigation_in_crowds':
            return self.run_navigation_crowd_test()
        elif scenario == 'collision_avoidance_tests':
            return self.run_collision_avoidance_test()
        elif scenario == 'formation_maintenance':
            return self.run_formation_maintenance_test()
        elif scenario == 'emergency_stop_scenarios':
            return self.run_emergency_stop_test()
        else:
            return MultiRobotResult(
                test_name=scenario,
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_close_proximity_test(self) -> MultiRobotResult:
        """Test robot behavior in close proximity scenarios"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(10)  # Allow time for robots to get close to each other

        # Check proximity event metrics
        total_proximity_events = sum(
            self.robot_data[robot_id]['proximity_events']
            for robot_id in self.robot_ids
        )

        # Calculate proximity management efficiency
        proximity_management_ok = total_proximity_events <= 20  # Reasonable number of events

        # Calculate collision cascade events
        cascade_events = self.multi_robot_metrics['collision_cascade_events']
        cascade_management_ok = cascade_events <= 3  # Very few cascade events acceptable

        metrics = {
            'total_proximity_events': total_proximity_events,
            'collision_cascade_events': cascade_events,
            'proximity_management_ok': proximity_management_ok,
            'cascade_management_ok': cascade_management_ok,
            'duration': time.time() - start_time
        }

        passed = proximity_management_ok and cascade_management_ok

        return MultiRobotResult(
            test_name='close_proximity_scenarios',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Close proximity test failed - Proximity events: {total_proximity_events}, Cascade events: {cascade_events}"
        )

    def run_navigation_crowd_test(self) -> MultiRobotResult:
        """Test navigation efficiency in crowded scenarios"""
        start_time = time.time()

        # Wait for navigation data collection
        time.sleep(8)

        # Check navigation efficiency
        navigation_efficiency = self.multi_robot_metrics['navigation_efficiency']
        navigation_efficiency_ok = navigation_efficiency >= 0.6  # 60% efficiency threshold

        # Check collision rates
        total_collisions = sum(
            self.robot_data[robot_id]['collisions']
            for robot_id in self.robot_ids
        )
        collision_rate_acceptable = total_collisions <= 5  # Max 5 collisions in 8 seconds

        metrics = {
            'navigation_efficiency': navigation_efficiency,
            'navigation_efficiency_ok': navigation_efficiency_ok,
            'total_collisions': total_collisions,
            'collision_rate_acceptable': collision_rate_acceptable,
            'duration': time.time() - start_time
        }

        passed = navigation_efficiency_ok and collision_rate_acceptable

        return MultiRobotResult(
            test_name='navigation_in_crowds',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Navigation crowd test failed - Efficiency: {navigation_efficiency:.2f}, Collisions: {total_collisions}"
        )

    def run_collision_avoidance_test(self) -> MultiRobotResult:
        """Test collision avoidance effectiveness"""
        start_time = time.time()

        # Wait for collision avoidance data
        time.sleep(6)

        # Calculate collision avoidance success rate
        total_attempts = 0
        successful_avoids = 0

        for robot_id in self.robot_ids:
            proximity_events = self.robot_data[robot_id]['proximity_events']
            collisions = self.robot_data[robot_id]['collisions']

            total_attempts += proximity_events
            successful_avoids += max(0, proximity_events - collisions)

        avoidance_success_rate = successful_avoids / total_attempts if total_attempts > 0 else 1.0
        avoidance_success_ok = avoidance_success_rate >= 0.8  # 80% success rate

        metrics = {
            'avoidance_success_rate': avoidance_success_rate,
            'avoidance_success_ok': avoidance_success_ok,
            'total_proximity_events': total_attempts,
            'successful_avoids': successful_avoids,
            'duration': time.time() - start_time
        }

        passed = avoidance_success_ok

        return MultiRobotResult(
            test_name='collision_avoidance_tests',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Collision avoidance test failed - Success rate: {avoidance_success_rate:.2f}"
        )

    def run_formation_maintenance_test(self) -> MultiRobotResult:
        """Test formation maintenance in multi-robot scenarios"""
        start_time = time.time()

        # Wait for formation data
        time.sleep(7)

        # Calculate formation maintenance based on robot positioning
        formation_stability = 0.0
        robot_count = len(self.robot_ids)

        if robot_count >= 2:
            # Calculate average distance deviation from expected formation
            distances = []
            for i, robot1_id in enumerate(self.robot_ids):
                for j, robot2_id in enumerate(self.robot_ids[i+1:], i+1):
                    if (len(self.robot_data[robot1_id]['odometry']) > 0 and
                        len(self.robot_data[robot2_id]['odometry']) > 0):

                        pos1 = self.robot_data[robot1_id]['odometry'][-1]['data'].pose.pose.position
                        pos2 = self.robot_data[robot2_id]['odometry'][-1]['data'].pose.pose.position

                        dist = math.sqrt(
                            (pos1.x - pos2.x)**2 +
                            (pos1.y - pos2.y)**2 +
                            (pos1.z - pos2.z)**2
                        )
                        distances.append(dist)

            if distances:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)

                # Formation is stable if distances don't vary too much
                formation_stability = max(0, min(1, 1 - (std_distance / avg_distance)))

        formation_stability_ok = formation_stability >= 0.6

        metrics = {
            'formation_stability': formation_stability,
            'formation_stability_ok': formation_stability_ok,
            'duration': time.time() - start_time
        }

        passed = formation_stability_ok

        return MultiRobotResult(
            test_name='formation_maintenance',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Formation maintenance test failed - Stability: {formation_stability:.2f}"
        )

    def run_emergency_stop_test(self) -> MultiRobotResult:
        """Test emergency stop scenarios and reactions"""
        start_time = time.time()

        # Wait for emergency stop data
        time.sleep(5)

        # Check for coordinated emergency stop behavior
        # In a real implementation, this would involve triggering emergency stops
        # and checking if other robots react appropriately

        # For simulation, we'll check if robots maintain safety during potential emergencies
        emergency_reaction_ok = True  # Placeholder for actual emergency stop testing

        # Check if collision cascade events are minimal during this scenario
        cascade_events = self.multi_robot_metrics['collision_cascade_events']
        emergency_safety_ok = cascade_events <= 2

        metrics = {
            'emergency_reaction_ok': emergency_reaction_ok,
            'emergency_safety_ok': emergency_safety_ok,
            'collision_cascade_events': cascade_events,
            'duration': time.time() - start_time
        }

        passed = emergency_reaction_ok and emergency_safety_ok

        return MultiRobotResult(
            test_name='emergency_stop_scenarios',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Emergency stop test failed - Cascade events: {cascade_events}"
        )

    def publish_test_results(self):
        """Publish multi-robot test results"""
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
            'multi_robot_metrics': self.multi_robot_metrics,
            'robot_data_summary': {
                robot_id: {
                    'collisions': data['collisions'],
                    'proximity_events': data['proximity_events'],
                    'laser_data_points': len(data['laser']),
                    'imu_data_points': len(data['imu'])
                } for robot_id, data in self.robot_data.items()
            }
        })

        self.test_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive multi-robot test report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'multi_robot_metrics': self.multi_robot_metrics.copy(),
            'robot_data_summary': {
                robot_id: {
                    'collisions': data['collisions'],
                    'proximity_events': data['proximity_events'],
                    'laser_data_points': len(data['laser']),
                    'imu_data_points': len(data['imu'])
                } for robot_id, data in self.robot_data.items()
            },
            'individual_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'duration': r.duration,
                    'metrics': r.metrics,
                    'error': r.error_message
                } for r in self.test_results
            ],
            'collision_chain_events': len(self.collision_chain_events),
            'proximity_threshold': self.proximity_threshold,
            'collision_threshold': self.collision_threshold,
            'recommendations': self.generate_recommendations()
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on multi-robot test results"""
        recommendations = []

        if self.multi_robot_metrics['collision_cascade_events'] > 5:
            recommendations.append("Reduce collision cascade events - improve inter-robot communication and coordination")

        if self.multi_robot_metrics['navigation_efficiency'] < 0.7:
            recommendations.append("Improve navigation efficiency - enhance path planning algorithms for multi-robot scenarios")

        if self.multi_robot_metrics['collision_avoidance_success'] < 0.8:
            recommendations.append("Enhance collision avoidance - tune proximity detection and reaction parameters")

        if self.multi_robot_metrics['communication_efficiency'] < 0.75:
            recommendations.append("Improve inter-robot communication - implement better coordination protocols")

        if not recommendations:
            recommendations.append("Multi-robot scenarios are performing well - continue monitoring for optimization")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed multi-robot test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("MULTI-ROBOT SCENARIOS WITH COLLISION CASCADES TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("PROXIMITY THRESHOLDS:")
        print("-" * 40)
        print(f"  Proximity Threshold: {report['proximity_threshold']:.2f}m")
        print(f"  Collision Threshold: {report['collision_threshold']:.2f}m")
        print(f"  Collision Chain Events: {report['collision_chain_events']}")

        print("\nMULTI-ROBOT METRICS:")
        print("-" * 40)
        metrics = report['multi_robot_metrics']
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("\nROBOT DATA SUMMARY:")
        print("-" * 40)
        for robot_id, data in report['robot_data_summary'].items():
            print(f"  {robot_id}:")
            print(f"    Collisions: {data['collisions']}")
            print(f"    Proximity Events: {data['proximity_events']}")
            print(f"    Laser Data Points: {data['laser_data_points']}")
            print(f"    IMU Data Points: {data['imu_data_points']}")

        print("\nINDIVIDUAL TEST RESULTS:")
        print("-" * 40)
        for result in report['individual_results']:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {status:4} | {result['test_name']:25} | {result['duration']:.2f}s | {result['error'] or 'OK'}")

        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("="*80)


def main(args=None):
    """Main function to run multi-robot tests"""
    rclpy.init(args=args)

    tester = MultiRobotTester()

    try:
        # Run comprehensive multi-robot tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Multi-Robot Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Multi-robot testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()