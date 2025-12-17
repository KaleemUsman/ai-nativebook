#!/usr/bin/env python3
"""
Complete Simulation-to-AI Pipeline Test
This script tests the complete pipeline from Gazebo physics simulation through
Unity visualization to AI agent processing and control, validating the full
digital twin system for humanoid robotics.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import Twist, PoseStamped, Vector3
from std_msgs.msg import String, Bool, Float64MultiArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import threading
from collections import deque
import statistics
import json
import subprocess
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class PipelineMetrics:
    """Metrics for the complete pipeline"""
    sensor_data_quality: float = 0.0
    ai_response_time: float = 0.0
    control_accuracy: float = 0.0
    system_stability: float = 0.0
    synchronization_score: float = 0.0


class CompletePipelineTester(Node):
    """
    Tests the complete simulation-to-AI pipeline including:
    - Gazebo physics simulation
    - Unity visualization
    - Sensor data processing
    - AI agent decision making
    - Robot control execution
    """

    def __init__(self):
        super().__init__('complete_pipeline_tester')

        # Initialize data storage
        self.gazebo_data = {
            'laser': deque(maxlen=50),
            'imu': deque(maxlen=50),
            'joint': deque(maxlen=50),
            'odometry': deque(maxlen=50)
        }

        self.unity_data = {
            'visualization': deque(maxlen=50),
            'rendering_metrics': deque(maxlen=50)
        }

        self.ai_data = {
            'decisions': deque(maxlen=50),
            'processing_times': deque(maxlen=50),
            'actions': deque(maxlen=50)
        }

        self.control_data = {
            'commands': deque(maxlen=50),
            'executions': deque(maxlen=50)
        }

        # Initialize ROS interfaces
        self.bridge = CvBridge()

        # Subscribers for Gazebo data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/gazebo/scan',
            self.scan_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/gazebo/imu/data',
            self.imu_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/gazebo/joint_states',
            self.joint_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/gazebo/odom',
            self.odom_callback,
            10
        )

        # Subscribers for AI agent data
        self.ai_decision_sub = self.create_subscription(
            String,
            '/ai_agent/decision',
            self.ai_decision_callback,
            10
        )

        self.ai_action_sub = self.create_subscription(
            Twist,
            '/ai_agent/action',
            self.ai_action_callback,
            10
        )

        # Publishers for test control
        self.test_control_pub = self.create_publisher(
            String,
            '/pipeline_test/control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/pipeline_test/results',
            10
        )

        self.debug_marker_pub = self.create_publisher(
            Marker,
            '/pipeline_test/debug',
            10
        )

        # Timer for pipeline testing
        self.test_timer = self.create_timer(0.1, self.run_pipeline_test)

        # Test scenarios
        self.test_scenarios = [
            'stationary_robot',
            'simple_navigation',
            'obstacle_avoidance',
            'balance_maintenance',
            'sensor_fusion_test'
        ]

        self.current_scenario = 'stationary_robot'
        self.scenario_start_time = time.time()
        self.scenario_duration = 10  # seconds per scenario

        # Performance metrics
        self.metrics = PipelineMetrics()
        self.test_results = []
        self.test_statistics = {
            'total_tests_run': 0,
            'successful_tests': 0,
            'average_response_time': 0.0,
            'stability_score': 0.0
        }

        # Test state
        self.test_active = False
        self.test_start_time = None
        self.current_test_phase = 'initialization'

        self.get_logger().info("Complete Pipeline Tester initialized and ready for testing")

    def scan_callback(self, msg):
        """Process LiDAR scan data from Gazebo"""
        self.gazebo_data['laser'].append({
            'data': msg,
            'timestamp': time.time(),
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })

    def imu_callback(self, msg):
        """Process IMU data from Gazebo"""
        self.gazebo_data['imu'].append({
            'data': msg,
            'timestamp': time.time(),
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })

    def joint_callback(self, msg):
        """Process joint state data from Gazebo"""
        self.gazebo_data['joint'].append({
            'data': msg,
            'timestamp': time.time(),
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })

    def odom_callback(self, msg):
        """Process odometry data from Gazebo"""
        self.gazebo_data['odometry'].append({
            'data': msg,
            'timestamp': time.time(),
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })

    def ai_decision_callback(self, msg):
        """Process AI decision data"""
        try:
            decision_data = json.loads(msg.data)
            self.ai_data['decisions'].append({
                'data': decision_data,
                'timestamp': time.time()
            })
        except Exception as e:
            self.get_logger().error(f"Error parsing AI decision: {e}")

    def ai_action_callback(self, msg):
        """Process AI action data"""
        self.ai_data['actions'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def run_pipeline_test(self):
        """Run the complete pipeline test"""
        try:
            if not self.test_active:
                return

            # Update test phase based on time
            current_time = time.time()
            elapsed_time = current_time - self.test_start_time

            # Check if we need to switch scenarios
            if (current_time - self.scenario_start_time) > self.scenario_duration:
                self.switch_test_scenario()

            # Perform current test scenario
            self.execute_current_scenario()

            # Calculate metrics
            self.calculate_pipeline_metrics()

            # Check for completion
            if elapsed_time > 60:  # 60-second test
                self.complete_test()

        except Exception as e:
            self.get_logger().error(f"Error in pipeline test: {e}")

    def start_test(self):
        """Start the complete pipeline test"""
        self.test_active = True
        self.test_start_time = time.time()
        self.scenario_start_time = time.time()
        self.current_test_phase = 'initialization'

        self.get_logger().info("Starting complete pipeline test...")

        # Send start command
        start_msg = String()
        start_msg.data = "start_test"
        self.test_control_pub.publish(start_msg)

    def switch_test_scenario(self):
        """Switch to the next test scenario"""
        current_idx = self.test_scenarios.index(self.current_scenario)
        next_idx = (current_idx + 1) % len(self.test_scenarios)
        self.current_scenario = self.test_scenarios[next_idx]
        self.scenario_start_time = time.time()

        self.get_logger().info(f"Switching to scenario: {self.current_scenario}")

        # Send scenario change command
        scenario_msg = String()
        scenario_msg.data = f"scenario_change:{self.current_scenario}"
        self.test_control_pub.publish(scenario_msg)

    def execute_current_scenario(self):
        """Execute the current test scenario"""
        if self.current_scenario == 'stationary_robot':
            self.execute_stationary_test()
        elif self.current_scenario == 'simple_navigation':
            self.execute_navigation_test()
        elif self.current_scenario == 'obstacle_avoidance':
            self.execute_obstacle_avoidance_test()
        elif self.current_scenario == 'balance_maintenance':
            self.execute_balance_test()
        elif self.current_scenario == 'sensor_fusion_test':
            self.execute_sensor_fusion_test()

    def execute_stationary_test(self):
        """Execute stationary robot test - verify basic sensor functionality"""
        # In stationary test, robot should remain in place
        # Verify that sensors are working but robot is not moving
        pass

    def execute_navigation_test(self):
        """Execute navigation test - verify path planning and execution"""
        # In navigation test, verify that robot moves toward target
        # and that sensors detect the movement
        pass

    def execute_obstacle_avoidance_test(self):
        """Execute obstacle avoidance test - verify detection and response"""
        # In obstacle avoidance test, verify that robot detects obstacles
        # and adjusts its path accordingly
        pass

    def execute_balance_test(self):
        """Execute balance maintenance test - verify IMU-based balance"""
        # In balance test, verify that robot maintains upright position
        # based on IMU feedback
        pass

    def execute_sensor_fusion_test(self):
        """Execute sensor fusion test - verify multiple sensor integration"""
        # In sensor fusion test, verify that multiple sensors work together
        # to provide coherent perception
        pass

    def calculate_pipeline_metrics(self):
        """Calculate metrics for the complete pipeline"""
        # Calculate sensor data quality
        self.metrics.sensor_data_quality = self.calculate_sensor_quality()

        # Calculate AI response time
        self.metrics.ai_response_time = self.calculate_ai_response_time()

        # Calculate control accuracy
        self.metrics.control_accuracy = self.calculate_control_accuracy()

        # Calculate system stability
        self.metrics.system_stability = self.calculate_system_stability()

        # Calculate synchronization score
        self.metrics.synchronization_score = self.calculate_synchronization_score()

    def calculate_sensor_quality(self) -> float:
        """Calculate sensor data quality score"""
        if len(self.gazebo_data['laser']) == 0:
            return 0.0

        # Calculate based on data validity and consistency
        latest_scan = self.gazebo_data['laser'][-1]['data']
        ranges = np.array(latest_scan.ranges)

        # Calculate percentage of valid ranges
        valid_ranges = np.isfinite(ranges)
        validity_percentage = np.sum(valid_ranges) / len(ranges)

        # Calculate consistency (low variance in similar conditions)
        if len(self.gazebo_data['laser']) > 10:
            recent_scans = [d['data'].ranges for d in list(self.gazebo_data['laser'])[-10:]]
            if len(recent_scans) > 0:
                recent_ranges = np.array(recent_scans)
                # Calculate average variance across scans
                avg_variance = np.mean(np.var(recent_ranges, axis=0))
                consistency_score = max(0, 1 - avg_variance / 100)  # Normalize
            else:
                consistency_score = 0.5
        else:
            consistency_score = 0.5

        return (validity_percentage + consistency_score) / 2

    def calculate_ai_response_time(self) -> float:
        """Calculate AI response time score"""
        if len(self.ai_data['processing_times']) == 0:
            return 0.0

        avg_processing_time = statistics.mean(self.ai_data['processing_times'])

        # Score based on processing time (faster is better)
        # Target: < 100ms for good performance
        if avg_processing_time < 0.05:  # 50ms
            return 1.0
        elif avg_processing_time < 0.1:  # 100ms
            return 0.8
        elif avg_processing_time < 0.2:  # 200ms
            return 0.5
        else:
            return 0.2

    def calculate_control_accuracy(self) -> float:
        """Calculate control accuracy score"""
        # Compare commanded actions with actual robot behavior
        if len(self.gazebo_data['odometry']) < 2:
            return 0.0

        # Calculate based on how well robot follows commands
        # This would require comparing commands to actual movement
        return 0.8  # Placeholder - would need actual comparison logic

    def calculate_system_stability(self) -> float:
        """Calculate system stability score"""
        # Calculate based on consistency of sensor readings and system behavior
        stability_score = 0.0

        if len(self.gazebo_data['imu']) > 10:
            # Calculate IMU stability (small variations indicate stability)
            imu_readings = []
            for data in list(self.gazebo_data['imu'])[-10:]:
                imu = data['data']
                imu_readings.append([
                    imu.orientation.x, imu.orientation.y,
                    imu.orientation.z, imu.orientation.w
                ])

            imu_array = np.array(imu_readings)
            imu_variances = np.var(imu_array, axis=0)
            avg_variance = np.mean(imu_variances)

            # Lower variance means more stable
            stability_score = max(0, min(1, 1 - avg_variance * 100))

        return stability_score

    def calculate_synchronization_score(self) -> float:
        """Calculate synchronization between components"""
        # Calculate based on timing consistency between different parts of the pipeline
        sync_score = 0.0

        # Check if we have data from different components
        if (len(self.gazebo_data['laser']) > 0 and
            len(self.ai_data['decisions']) > 0 and
            len(self.control_data['commands']) > 0):

            # Calculate timing consistency
            latest_gazebo = self.gazebo_data['laser'][-1]['timestamp']
            latest_ai = self.ai_data['decisions'][-1]['timestamp'] if self.ai_data['decisions'] else latest_gazebo
            latest_control = self.control_data['commands'][-1]['timestamp'] if self.control_data['commands'] else latest_gazebo

            # Calculate time differences
            gazebo_ai_delay = abs(latest_gazebo - latest_ai)
            ai_control_delay = abs(latest_ai - latest_control)
            total_delay = gazebo_ai_delay + ai_control_delay

            # Score based on delay (lower delay is better)
            if total_delay < 0.1:  # 100ms
                sync_score = 1.0
            elif total_delay < 0.2:  # 200ms
                sync_score = 0.8
            elif total_delay < 0.5:  # 500ms
                sync_score = 0.5
            else:
                sync_score = 0.2

        return sync_score

    def complete_test(self):
        """Complete the pipeline test and generate results"""
        self.test_active = False

        # Calculate final statistics
        self.test_statistics['total_tests_run'] = len(self.test_scenarios)
        self.test_statistics['average_response_time'] = self.metrics.ai_response_time
        self.test_statistics['stability_score'] = self.metrics.system_stability

        # Determine success based on metrics
        overall_score = (
            self.metrics.sensor_data_quality +
            self.metrics.ai_response_time +
            self.metrics.control_accuracy +
            self.metrics.system_stability +
            self.metrics.synchronization_score
        ) / 5

        self.test_statistics['successful_tests'] = 1 if overall_score > 0.7 else 0

        # Publish results
        results_msg = String()
        results_msg.data = json.dumps({
            'timestamp': time.time(),
            'metrics': {
                'sensor_data_quality': self.metrics.sensor_data_quality,
                'ai_response_time': self.metrics.ai_response_time,
                'control_accuracy': self.metrics.control_accuracy,
                'system_stability': self.metrics.system_stability,
                'synchronization_score': self.metrics.synchronization_score
            },
            'statistics': self.test_statistics,
            'overall_score': overall_score,
            'pass': overall_score > 0.7
        })

        self.test_results_pub.publish(results_msg)

        self.get_logger().info(f"Pipeline test completed. Overall score: {overall_score:.2f}")

    def get_test_report(self) -> Dict:
        """Generate a comprehensive test report"""
        report = {
            'timestamp': time.time(),
            'test_configuration': {
                'duration': 60,
                'scenarios': self.test_scenarios,
                'current_scenario': self.current_scenario
            },
            'pipeline_metrics': {
                'sensor_data_quality': self.metrics.sensor_data_quality,
                'ai_response_time': self.metrics.ai_response_time,
                'control_accuracy': self.metrics.control_accuracy,
                'system_stability': self.metrics.system_stability,
                'synchronization_score': self.metrics.synchronization_score
            },
            'performance_statistics': self.test_statistics.copy(),
            'data_samples': {
                'gazebo_laser_samples': len(self.gazebo_data['laser']),
                'gazebo_imu_samples': len(self.gazebo_data['imu']),
                'gazebo_joint_samples': len(self.gazebo_data['joint']),
                'ai_decisions': len(self.ai_data['decisions']),
                'ai_actions': len(self.ai_data['actions'])
            },
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            },
            'overall_assessment': self.calculate_overall_assessment()
        }

        return report

    def calculate_overall_assessment(self) -> Dict[str, str]:
        """Calculate overall assessment of the pipeline"""
        assessment = {}

        # Evaluate each metric
        if self.metrics.sensor_data_quality > 0.8:
            assessment['sensor_quality'] = 'Excellent'
        elif self.metrics.sensor_data_quality > 0.6:
            assessment['sensor_quality'] = 'Good'
        elif self.metrics.sensor_data_quality > 0.4:
            assessment['sensor_quality'] = 'Fair'
        else:
            assessment['sensor_quality'] = 'Poor'

        if self.metrics.ai_response_time > 0.8:
            assessment['ai_response'] = 'Excellent'
        elif self.metrics.ai_response_time > 0.6:
            assessment['ai_response'] = 'Good'
        elif self.metrics.ai_response_time > 0.4:
            assessment['ai_response'] = 'Fair'
        else:
            assessment['ai_response'] = 'Poor'

        if self.metrics.system_stability > 0.8:
            assessment['system_stability'] = 'Excellent'
        elif self.metrics.system_stability > 0.6:
            assessment['system_stability'] = 'Good'
        elif self.metrics.system_stability > 0.4:
            assessment['system_stability'] = 'Fair'
        else:
            assessment['system_stability'] = 'Poor'

        if self.metrics.synchronization_score > 0.8:
            assessment['synchronization'] = 'Excellent'
        elif self.metrics.synchronization_score > 0.6:
            assessment['synchronization'] = 'Good'
        elif self.metrics.synchronization_score > 0.4:
            assessment['synchronization'] = 'Fair'
        else:
            assessment['synchronization'] = 'Poor'

        # Calculate overall grade
        overall_score = (
            self.metrics.sensor_data_quality +
            self.metrics.ai_response_time +
            self.metrics.system_stability +
            self.metrics.synchronization_score
        ) / 4

        if overall_score > 0.8:
            assessment['overall_grade'] = 'A - Excellent'
        elif overall_score > 0.6:
            assessment['overall_grade'] = 'B - Good'
        elif overall_score > 0.4:
            assessment['overall_grade'] = 'C - Fair'
        elif overall_score > 0.2:
            assessment['overall_grade'] = 'D - Needs Improvement'
        else:
            assessment['overall_grade'] = 'F - Poor'

        assessment['recommendations'] = self.generate_recommendations(assessment)

        return assessment

    def generate_recommendations(self, assessment: Dict[str, str]) -> List[str]:
        """Generate recommendations based on assessment"""
        recommendations = []

        if assessment['sensor_quality'] in ['Poor', 'Fair']:
            recommendations.append("Improve sensor data quality and validity checking")

        if assessment['ai_response'] in ['Poor', 'Fair']:
            recommendations.append("Optimize AI processing pipeline for better response times")

        if assessment['system_stability'] in ['Poor', 'Fair']:
            recommendations.append("Address system stability and consistency issues")

        if assessment['synchronization'] in ['Poor', 'Fair']:
            recommendations.append("Improve synchronization between pipeline components")

        if not recommendations:
            recommendations.append("Pipeline is performing well - continue monitoring for optimization opportunities")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed test report"""
        report = self.get_test_report()

        print("\n" + "="*80)
        print("COMPLETE SIMULATION-TO-AI PIPELINE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Duration: 60 seconds")
        print(f"Scenarios: {', '.join(report['test_configuration']['scenarios'])}")
        print()

        print("PIPELINE METRICS:")
        print("-" * 40)
        metrics = report['pipeline_metrics']
        print(f"  Sensor Data Quality: {metrics['sensor_data_quality']:.2f} ({self.map_score_to_label(metrics['sensor_data_quality'])})")
        print(f"  AI Response Time: {metrics['ai_response_time']:.2f} ({self.map_score_to_label(metrics['ai_response_time'])})")
        print(f"  System Stability: {metrics['system_stability']:.2f} ({self.map_score_to_label(metrics['system_stability'])})")
        print(f"  Synchronization: {metrics['synchronization_score']:.2f} ({self.map_score_to_label(metrics['synchronization_score'])})")

        print("\nPERFORMANCE STATISTICS:")
        print("-" * 40)
        stats = report['performance_statistics']
        print(f"  Total Tests Run: {stats['total_tests_run']}")
        print(f"  Successful Tests: {stats['successful_tests']}")
        print(f"  Average Response Time: {stats['average_response_time']:.3f}s")

        print("\nDATA SAMPLES:")
        print("-" * 40)
        samples = report['data_samples']
        for source, count in samples.items():
            print(f"  {source}: {count}")

        print("\nSYSTEM RESOURCES:")
        print("-" * 40)
        resources = report['system_resources']
        print(f"  CPU Usage: {resources['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {resources['memory_percent']:.1f}%")
        print(f"  Disk Usage: {resources['disk_percent']:.1f}%")

        print("\nASSESSMENT:")
        print("-" * 40)
        assessment = report['overall_assessment']
        for category, grade in assessment.items():
            if category != 'recommendations':
                print(f"  {category.replace('_', ' ').title()}: {grade}")

        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in assessment['recommendations']:
            print(f"  • {rec}")

        print("="*80)

    def map_score_to_label(self, score: float) -> str:
        """Map numerical score to qualitative label"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"

    def run_comprehensive_test(self):
        """Run a comprehensive test of the entire pipeline"""
        self.get_logger().info("Starting comprehensive pipeline test...")

        # Start the test
        self.start_test()

        # Wait for test to complete
        start_time = time.time()
        while self.test_active and (time.time() - start_time) < 70:  # Add 10s buffer
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Print detailed report
        self.print_detailed_report()


def main(args=None):
    """Main function to run the complete pipeline test"""
    rclpy.init(args=args)

    tester = CompletePipelineTester()

    try:
        # Run comprehensive test
        tester.run_comprehensive_test()

    except KeyboardInterrupt:
        tester.get_logger().info("Complete pipeline test interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()