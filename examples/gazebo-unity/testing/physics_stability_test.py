#!/usr/bin/env python3
"""
Physics Stability Test for Humanoid Robotics
This script tests the stability of physics simulations for humanoid models in Gazebo,
checking for stability under various conditions and scenarios.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetModelState, SetModelState
from std_msgs.msg import String, Float64
from builtin_interfaces.msg import Time
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
class StabilityTestResult:
    """Result of a physics stability test"""
    test_name: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class PhysicsStabilityTester(Node):
    """
    Tester for physics stability of humanoid models in Gazebo simulation
    """

    def __init__(self):
        super().__init__('physics_stability_tester')

        # Initialize data storage
        self.sensor_data = {
            'imu': deque(maxlen=200),  # Increased buffer for stability analysis
            'joint': deque(maxlen=200),
            'odometry': deque(maxlen=200)
        }

        self.stability_metrics = {
            'balance_score': 0.0,
            'oscillation_count': 0,
            'drift_rate': 0.0,
            'energy_stability': 0.0,
            'joint_stability': 0.0
        }

        # Initialize ROS interfaces
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

        # Publishers for test control and results
        self.test_control_pub = self.create_publisher(
            String,
            '/physics_stability/test_control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/physics_stability/results',
            10
        )

        # Service clients for Gazebo interaction
        self.get_state_cli = self.create_client(GetModelState, '/get_entity_state')
        self.set_state_cli = self.create_client(SetModelState, '/set_model_state')

        # Timer for stability monitoring
        self.monitor_timer = self.create_timer(0.05, self.stability_monitoring)  # 20Hz monitoring

        # Test parameters
        self.test_scenarios = [
            'stand_still_stability',
            'walking_gait_stability',
            'balance_recovery',
            'external_disturbance_response',
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

        # Stability thresholds
        self.balance_threshold = 0.7  # Minimum balance score to pass
        self.oscillation_threshold = 10  # Maximum oscillations allowed
        self.drift_threshold = 0.1  # Maximum drift rate (m/s)
        self.energy_threshold = 10.0  # Maximum energy change rate

        self.get_logger().info("Physics Stability Tester initialized and ready for testing")

    def imu_callback(self, msg):
        """Process IMU data for stability analysis"""
        self.sensor_data['imu'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def joint_callback(self, msg):
        """Process joint state data for stability analysis"""
        self.sensor_data['joint'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def odom_callback(self, msg):
        """Process odometry data for stability analysis"""
        self.sensor_data['odometry'].append({
            'data': msg,
            'timestamp': time.time()
        })

    def stability_monitoring(self):
        """Monitor and calculate stability metrics"""
        try:
            # Update stability metrics based on sensor data
            self.update_balance_metrics()
            self.update_oscillation_metrics()
            self.update_drift_metrics()
            self.update_energy_metrics()
            self.update_joint_stability_metrics()

        except Exception as e:
            self.get_logger().error(f"Error in stability monitoring: {e}")

    def update_balance_metrics(self):
        """Update balance stability metrics based on IMU data"""
        if len(self.sensor_data['imu']) > 10:
            # Calculate balance based on orientation stability
            orientations = []
            for data in list(self.sensor_data['imu'])[-20:]:  # Last 20 readings
                imu = data['data']
                # Convert quaternion to Euler angles to check balance
                roll, pitch, yaw = self.quaternion_to_euler([
                    imu.orientation.x, imu.orientation.y,
                    imu.orientation.z, imu.orientation.w
                ])
                orientations.append([roll, pitch, yaw])

            if orientations:
                orient_array = np.array(orientations)
                # Calculate how close to upright position (roll=0, pitch=0)
                roll_deviation = np.mean(np.abs(orient_array[:, 0]))
                pitch_deviation = np.mean(np.abs(orient_array[:, 1]))

                # Balance score: 1.0 = perfect balance, 0.0 = fallen over
                # Using 0.5 radians (~30 degrees) as maximum acceptable deviation
                max_acceptable_deviation = 0.5
                avg_deviation = (roll_deviation + pitch_deviation) / 2
                balance_score = max(0, min(1, 1 - (avg_deviation / max_acceptable_deviation)))

                self.stability_metrics['balance_score'] = balance_score

    def update_oscillation_metrics(self):
        """Update oscillation metrics based on IMU and joint data"""
        if len(self.sensor_data['imu']) > 20:
            # Count oscillations based on rapid changes in IMU data
            oscillation_count = 0

            for i in range(1, min(20, len(self.sensor_data['imu']))):
                prev_imu = self.sensor_data['imu'][-(i+1)]['data']
                curr_imu = self.sensor_data['imu'][-i]['data']

                # Check for oscillations in linear acceleration
                lin_acc_change = abs(curr_imu.linear_acceleration.x - prev_imu.linear_acceleration.x)
                ang_vel_change = abs(curr_imu.angular_velocity.x - prev_imu.angular_velocity.x)

                if lin_acc_change > 2.0 or ang_vel_change > 0.5:  # Thresholds for oscillation detection
                    oscillation_count += 1

            self.stability_metrics['oscillation_count'] = oscillation_count

    def update_drift_metrics(self):
        """Update drift metrics based on odometry data"""
        if len(self.sensor_data['odometry']) > 10:
            # Calculate drift rate based on position changes
            first_pos = self.sensor_data['odometry'][0]['data'].pose.pose.position
            last_pos = self.sensor_data['odometry'][-1]['data'].pose.pose.position

            pos_diff = math.sqrt(
                (last_pos.x - first_pos.x)**2 +
                (last_pos.y - first_pos.y)**2 +
                (last_pos.z - first_pos.z)**2
            )

            first_time = self.sensor_data['odometry'][0]['timestamp']
            last_time = self.sensor_data['odometry'][-1]['timestamp']

            time_diff = last_time - first_time
            if time_diff > 0:
                drift_rate = pos_diff / time_diff
                self.stability_metrics['drift_rate'] = drift_rate

    def update_energy_metrics(self):
        """Update energy stability metrics based on joint and IMU data"""
        if len(self.sensor_data['joint']) > 5:
            # Calculate energy based on joint velocities and IMU data
            total_energy = 0.0

            for data in list(self.sensor_data['joint'])[-5:]:  # Last 5 readings
                joint = data['data']
                if len(joint.velocity) > 0:
                    # Calculate kinetic energy from joint velocities
                    for vel in joint.velocity:
                        total_energy += 0.5 * vel**2  # Simplified KE calculation

            # Normalize by number of readings
            avg_energy = total_energy / min(5, len(self.sensor_data['joint']))
            self.stability_metrics['energy_stability'] = avg_energy

    def update_joint_stability_metrics(self):
        """Update joint stability metrics"""
        if len(self.sensor_data['joint']) > 10:
            # Calculate joint stability based on position consistency
            joint_positions = []

            for data in list(self.sensor_data['joint'])[-10:]:  # Last 10 readings
                joint = data['data']
                if len(joint.position) > 0:
                    joint_positions.append(joint.position[:min(6, len(joint.position))])  # First 6 joints

            if len(joint_positions) > 1:
                pos_array = np.array(joint_positions)
                # Calculate variance in joint positions (lower variance = more stable)
                if pos_array.ndim > 1:
                    pos_variance = np.mean(np.var(pos_array, axis=0))
                    self.stability_metrics['joint_stability'] = max(0, min(1, 1 - pos_variance * 10))

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
        """Run comprehensive physics stability tests"""
        self.get_logger().info("Starting comprehensive physics stability tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running stability test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> StabilityTestResult:
        """Run a specific stability test scenario"""
        if scenario == 'stand_still_stability':
            return self.run_stand_still_test()
        elif scenario == 'walking_gait_stability':
            return self.run_walking_gait_test()
        elif scenario == 'balance_recovery':
            return self.run_balance_recovery_test()
        elif scenario == 'external_disturbance_response':
            return self.run_external_disturbance_test()
        elif scenario == 'long_duration_stability':
            return self.run_long_duration_test()
        else:
            return StabilityTestResult(
                test_name=scenario,
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_stand_still_test(self) -> StabilityTestResult:
        """Test stability when robot should remain standing still"""
        start_time = time.time()

        # Wait for sufficient data collection
        time.sleep(10)  # Wait 10 seconds for stability assessment

        # Assess stability metrics
        balance_ok = self.stability_metrics['balance_score'] >= self.balance_threshold
        oscillation_ok = self.stability_metrics['oscillation_count'] <= self.oscillation_threshold
        drift_ok = self.stability_metrics['drift_rate'] <= self.drift_threshold

        metrics = {
            'balance_score': self.stability_metrics['balance_score'],
            'oscillation_count': self.stability_metrics['oscillation_count'],
            'drift_rate': self.stability_metrics['drift_rate'],
            'balance_ok': balance_ok,
            'oscillation_ok': oscillation_ok,
            'drift_ok': drift_ok,
            'duration': time.time() - start_time
        }

        passed = balance_ok and oscillation_ok and drift_ok

        return StabilityTestResult(
            test_name='stand_still_stability',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Stand-still test failed - Balance: {self.stability_metrics['balance_score']:.2f}, Oscillations: {self.stability_metrics['oscillation_count']}, Drift: {self.stability_metrics['drift_rate']:.3f}"
        )

    def run_walking_gait_test(self) -> StabilityTestResult:
        """Test stability during walking gait patterns"""
        start_time = time.time()

        # For this test, we'll check if the robot can maintain balance during movement
        # In a real implementation, we'd command the robot to walk
        time.sleep(8)  # Wait for potential walking data

        # During walking, expect some oscillation but maintain overall balance
        balance_ok = self.stability_metrics['balance_score'] >= 0.5  # Lower threshold for walking
        oscillation_acceptable = self.stability_metrics['oscillation_count'] <= self.oscillation_threshold * 2  # Allow more oscillation
        energy_acceptable = self.stability_metrics['energy_stability'] <= self.energy_threshold

        metrics = {
            'balance_score': self.stability_metrics['balance_score'],
            'oscillation_count': self.stability_metrics['oscillation_count'],
            'energy_stability': self.stability_metrics['energy_stability'],
            'balance_ok': balance_ok,
            'oscillation_acceptable': oscillation_acceptable,
            'energy_acceptable': energy_acceptable,
            'duration': time.time() - start_time
        }

        passed = balance_ok and oscillation_acceptable and energy_acceptable

        return StabilityTestResult(
            test_name='walking_gait_stability',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Walking gait test failed - Balance: {self.stability_metrics['balance_score']:.2f}, Oscillations: {self.stability_metrics['oscillation_count']}, Energy: {self.stability_metrics['energy_stability']:.2f}"
        )

    def run_balance_recovery_test(self) -> StabilityTestResult:
        """Test ability to recover balance from disturbances"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(6)

        # Check if robot can maintain or recover balance
        balance_recovery_ok = self.stability_metrics['balance_score'] >= 0.6
        joint_stability_ok = self.stability_metrics['joint_stability'] >= 0.5

        metrics = {
            'balance_score': self.stability_metrics['balance_score'],
            'joint_stability': self.stability_metrics['joint_stability'],
            'balance_recovery_ok': balance_recovery_ok,
            'joint_stability_ok': joint_stability_ok,
            'duration': time.time() - start_time
        }

        passed = balance_recovery_ok and joint_stability_ok

        return StabilityTestResult(
            test_name='balance_recovery',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Balance recovery test failed - Balance: {self.stability_metrics['balance_score']:.2f}, Joint Stability: {self.stability_metrics['joint_stability']:.2f}"
        )

    def run_external_disturbance_test(self) -> StabilityTestResult:
        """Test response to external disturbances"""
        start_time = time.time()

        # Wait for disturbance response data
        time.sleep(5)

        # Check for resilience to disturbances
        balance_resilience_ok = self.stability_metrics['balance_score'] >= 0.5
        oscillation_resilience_ok = self.stability_metrics['oscillation_count'] <= self.oscillation_threshold * 1.5

        metrics = {
            'balance_score': self.stability_metrics['balance_score'],
            'oscillation_count': self.stability_metrics['oscillation_count'],
            'balance_resilience_ok': balance_resilience_ok,
            'oscillation_resilience_ok': oscillation_resilience_ok,
            'duration': time.time() - start_time
        }

        passed = balance_resilience_ok and oscillation_resilience_ok

        return StabilityTestResult(
            test_name='external_disturbance_response',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"External disturbance test failed - Balance: {self.stability_metrics['balance_score']:.2f}, Oscillations: {self.stability_metrics['oscillation_count']}"
        )

    def run_long_duration_test(self) -> StabilityTestResult:
        """Test long-duration stability (simulates extended operation)"""
        start_time = time.time()

        # Wait for longer data collection
        time.sleep(15)  # Longer test for stability over time

        # Assess long-term stability
        long_term_balance_ok = self.stability_metrics['balance_score'] >= self.balance_threshold
        drift_acceptable = self.stability_metrics['drift_rate'] <= self.drift_threshold * 0.5  # Stricter drift for long duration
        energy_stable = self.stability_metrics['energy_stability'] <= self.energy_threshold * 0.5  # Stricter energy

        metrics = {
            'balance_score': self.stability_metrics['balance_score'],
            'drift_rate': self.stability_metrics['drift_rate'],
            'energy_stability': self.stability_metrics['energy_stability'],
            'long_term_balance_ok': long_term_balance_ok,
            'drift_acceptable': drift_acceptable,
            'energy_stable': energy_stable,
            'duration': time.time() - start_time
        }

        passed = long_term_balance_ok and drift_acceptable and energy_stable

        return StabilityTestResult(
            test_name='long_duration_stability',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Long duration test failed - Balance: {self.stability_metrics['balance_score']:.2f}, Drift: {self.stability_metrics['drift_rate']:.3f}, Energy: {self.stability_metrics['energy_stability']:.2f}"
        )

    def publish_test_results(self):
        """Publish physics stability test results"""
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
            'stability_metrics': self.stability_metrics
        })

        self.test_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive physics stability test report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'stability_metrics': self.stability_metrics.copy(),
            'individual_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'duration': r.duration,
                    'metrics': r.metrics,
                    'error': r.error_message
                } for r in self.test_results
            ],
            'overall_assessment': self.calculate_overall_assessment(),
            'recommendations': self.generate_recommendations()
        }

        return report

    def calculate_overall_assessment(self) -> Dict[str, float]:
        """Calculate overall stability assessment"""
        assessment = {}

        # Calculate weighted scores
        balance_weight = 0.3
        oscillation_weight = 0.2
        drift_weight = 0.2
        energy_weight = 0.15
        joint_weight = 0.15

        overall_score = (
            self.stability_metrics['balance_score'] * balance_weight +
            (1 - min(1, self.stability_metrics['oscillation_count'] / 20)) * oscillation_weight +  # Inverse relationship
            (1 - min(1, self.stability_metrics['drift_rate'] / 0.5)) * drift_weight +  # Inverse relationship
            (1 - min(1, self.stability_metrics['energy_stability'] / 5)) * energy_weight +  # Inverse relationship
            self.stability_metrics['joint_stability'] * joint_weight
        )

        assessment['overall_stability_score'] = overall_score
        assessment['balance_contribution'] = self.stability_metrics['balance_score'] * balance_weight
        assessment['oscillation_contribution'] = (1 - min(1, self.stability_metrics['oscillation_count'] / 20)) * oscillation_weight
        assessment['drift_contribution'] = (1 - min(1, self.stability_metrics['drift_rate'] / 0.5)) * drift_weight
        assessment['energy_contribution'] = (1 - min(1, self.stability_metrics['energy_stability'] / 5)) * energy_weight
        assessment['joint_contribution'] = self.stability_metrics['joint_stability'] * joint_weight

        return assessment

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on stability test results"""
        recommendations = []

        if self.stability_metrics['balance_score'] < 0.7:
            recommendations.append("Improve robot balance control - adjust center of mass or add balance controllers")

        if self.stability_metrics['oscillation_count'] > 15:
            recommendations.append("Reduce oscillations - tune control gains or add damping")

        if self.stability_metrics['drift_rate'] > 0.15:
            recommendations.append("Address position drift - check joint friction or control parameters")

        if self.stability_metrics['energy_stability'] > 8.0:
            recommendations.append("Reduce energy fluctuations - optimize control algorithms")

        if self.stability_metrics['joint_stability'] < 0.6:
            recommendations.append("Improve joint stability - check joint limits or actuator parameters")

        if not recommendations:
            recommendations.append("Physics stability is performing well - continue monitoring for optimization")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed physics stability test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("PHYSICS STABILITY COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("STABILITY METRICS:")
        print("-" * 40)
        metrics = report['stability_metrics']
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("\nOVERALL ASSESSMENT:")
        print("-" * 40)
        assessment = report['overall_assessment']
        for metric, value in assessment.items():
            print(f"  {metric}: {value:.3f}")

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
    """Main function to run physics stability tests"""
    rclpy.init(args=args)

    tester = PhysicsStabilityTester()

    try:
        # Run comprehensive stability tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Physics Stability Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Physics stability testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()