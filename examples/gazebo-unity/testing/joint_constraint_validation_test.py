#!/usr/bin/env python3
"""
Joint Constraint Validation Test
This script validates complex humanoid joint constraints in Unity physics simulation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String, Float64
from gazebo_msgs.srv import GetJointProperties, SetJointProperties
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
class JointConstraintResult:
    """Result of a joint constraint validation test"""
    test_name: str
    joint_name: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class JointConstraintValidator(Node):
    """
    Validator for complex humanoid joint constraints in Unity physics
    """

    def __init__(self):
        super().__init__('joint_constraint_validator')

        # Initialize data storage
        self.joint_data = deque(maxlen=200)
        self.constraint_metrics = {
            'position_accuracy': 0.0,
            'velocity_constraints': 0.0,
            'effort_limits': 0.0,
            'joint_coupling': 0.0,
            'kinematic_chain_integrity': 0.0
        }

        # Initialize ROS interfaces
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.constraint_violation_pub = self.create_publisher(
            Marker,
            '/joint_constraints/violation_markers',
            10
        )

        self.test_control_pub = self.create_publisher(
            String,
            '/joint_constraint/test_control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/joint_constraint/results',
            10
        )

        # Service clients for joint property management
        self.get_joint_props_cli = self.create_client(GetJointProperties, '/get_joint_properties')
        self.set_joint_props_cli = self.create_client(SetJointProperties, '/set_joint_properties')

        # Timer for constraint monitoring
        self.monitor_timer = self.create_timer(0.05, self.constraint_monitoring)  # 20Hz monitoring

        # Humanoid joint definitions
        self.humanoid_joints = {
            'head': {
                'joints': ['head_pan', 'head_tilt'],
                'type': 'revolute',
                'limits': {'position': [-1.57, 1.57], 'velocity': [-2.0, 2.0], 'effort': [0, 100]}
            },
            'left_arm': {
                'joints': ['left_shoulder_pan', 'left_shoulder_lift', 'left_elbow_flex',
                          'left_wrist_flex', 'left_wrist_roll'],
                'type': 'revolute',
                'limits': {'position': [-2.0, 2.0], 'velocity': [-3.0, 3.0], 'effort': [0, 200]}
            },
            'right_arm': {
                'joints': ['right_shoulder_pan', 'right_shoulder_lift', 'right_elbow_flex',
                          'right_wrist_flex', 'right_wrist_roll'],
                'type': 'revolute',
                'limits': {'position': [-2.0, 2.0], 'velocity': [-3.0, 3.0], 'effort': [0, 200]}
            },
            'left_leg': {
                'joints': ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
                          'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll'],
                'type': 'revolute',
                'limits': {'position': [-1.57, 0.785], 'velocity': [-2.5, 2.5], 'effort': [0, 500]}
            },
            'right_leg': {
                'joints': ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
                      'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll'],
                'type': 'revolute',
                'limits': {'position': [-1.57, 0.785], 'velocity': [-2.5, 2.5], 'effort': [0, 500]}
            }
        }

        # Test parameters
        self.test_scenarios = [
            'joint_limit_enforcement',
            'velocity_constraint_validation',
            'effort_limit_verification',
            'kinematic_chain_integrity',
            'coupled_joint_coordination',
            'extreme_pose_stability'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        # Joint constraint validation thresholds
        self.position_tolerance = 0.05  # 5cm tolerance
        self.velocity_tolerance = 0.1   # 0.1 rad/s tolerance
        self.effort_tolerance = 5.0     # 5.0 Nm tolerance
        self.coupling_tolerance = 0.02  # 2% coupling error tolerance

        self.get_logger().info("Joint Constraint Validator initialized and ready for testing")

    def joint_callback(self, msg):
        """Process joint state data for constraint validation"""
        self.joint_data.append({
            'data': msg,
            'timestamp': time.time()
        })

    def constraint_monitoring(self):
        """Monitor joint constraint compliance"""
        try:
            # Update constraint metrics based on joint data
            self.update_position_constraints()
            self.update_velocity_constraints()
            self.update_effort_constraints()
            self.update_coupling_constraints()
            self.update_kinematic_chain_metrics()

        except Exception as e:
            self.get_logger().error(f"Error in constraint monitoring: {e}")

    def update_position_constraints(self):
        """Update metrics for position constraint compliance"""
        if len(self.joint_data) > 0:
            latest_joints = self.joint_data[-1]['data']

            violations = 0
            total_joints = 0

            for i, joint_name in enumerate(latest_joints.name):
                if i < len(latest_joints.position):
                    position = latest_joints.position[i]

                    # Find joint definition
                    joint_def = self.find_joint_definition(joint_name)
                    if joint_def and 'position' in joint_def['limits']:
                        min_pos, max_pos = joint_def['limits']['position']

                        if position < min_pos - self.position_tolerance or position > max_pos + self.position_tolerance:
                            violations += 1

                            # Publish violation marker
                            self.publish_violation_marker(joint_name, position, min_pos, max_pos)

                    total_joints += 1

            if total_joints > 0:
                compliance_rate = 1.0 - (violations / total_joints)
                self.constraint_metrics['position_accuracy'] = compliance_rate

    def update_velocity_constraints(self):
        """Update metrics for velocity constraint compliance"""
        if len(self.joint_data) > 1:
            current_joints = self.joint_data[-1]['data']
            previous_joints = self.joint_data[-2]['data']

            violations = 0
            total_joints = 0

            for i, joint_name in enumerate(current_joints.name):
                if (i < len(current_joints.velocity) and
                    i < len(previous_joints.position) and
                    i < len(current_joints.position)):

                    # Calculate velocity from position change
                    dt = self.joint_data[-1]['timestamp'] - self.joint_data[-2]['timestamp']
                    if dt > 0:
                        calculated_velocity = (current_joints.position[i] - previous_joints.position[i]) / dt
                        reported_velocity = current_joints.velocity[i]

                        # Use the reported velocity for constraint checking
                        velocity = reported_velocity

                        # Find joint definition
                        joint_def = self.find_joint_definition(joint_name)
                        if joint_def and 'velocity' in joint_def['limits']:
                            min_vel, max_vel = joint_def['limits']['velocity']

                            if velocity < min_vel - self.velocity_tolerance or velocity > max_vel + self.velocity_tolerance:
                                violations += 1

                    total_joints += 1

            if total_joints > 0:
                compliance_rate = 1.0 - (violations / total_joints)
                self.constraint_metrics['velocity_constraints'] = compliance_rate

    def update_effort_constraints(self):
        """Update metrics for effort constraint compliance"""
        if len(self.joint_data) > 0:
            latest_joints = self.joint_data[-1]['data']

            violations = 0
            total_joints = 0

            for i, joint_name in enumerate(latest_joints.name):
                if i < len(latest_joints.effort):
                    effort = abs(latest_joints.effort[i])

                    # Find joint definition
                    joint_def = self.find_joint_definition(joint_name)
                    if joint_def and 'effort' in joint_def['limits']:
                        min_effort, max_effort = joint_def['limits']['effort']

                        if effort > max_effort + self.effort_tolerance:
                            violations += 1

                    total_joints += 1

            if total_joints > 0:
                compliance_rate = 1.0 - (violations / total_joints)
                self.constraint_metrics['effort_limits'] = compliance_rate

    def update_coupling_constraints(self):
        """Update metrics for coupled joint constraints"""
        # Check for coupled joints (e.g., left/right symmetry)
        if len(self.joint_data) > 0:
            latest_joints = self.joint_data[-1]['data']

            # Create a dictionary for easy lookup
            joint_positions = dict(zip(latest_joints.name, latest_joints.position))

            # Check symmetric joints for coordination
            symmetric_pairs = [
                ('left_shoulder_pan', 'right_shoulder_pan'),
                ('left_elbow_flex', 'right_elbow_flex'),
                ('left_hip_yaw', 'right_hip_yaw'),
                ('left_knee_pitch', 'right_knee_pitch')
            ]

            coordination_errors = 0
            total_pairs = 0

            for left_joint, right_joint in symmetric_pairs:
                if left_joint in joint_positions and right_joint in joint_positions:
                    left_pos = joint_positions[left_joint]
                    right_pos = joint_positions[right_joint]

                    # Check if symmetric joints are moving in coordinated manner
                    # (this depends on the intended behavior - for some movements they should be opposite)
                    # For this test, we'll check for reasonable coordination
                    diff = abs(left_pos - right_pos)
                    if diff > 2.0:  # Threshold for coordination violation
                        coordination_errors += 1

                    total_pairs += 1

            if total_pairs > 0:
                coordination_rate = 1.0 - (coordination_errors / total_pairs)
                self.constraint_metrics['joint_coupling'] = coordination_rate

    def find_joint_definition(self, joint_name: str) -> Optional[Dict]:
        """Find the definition for a specific joint"""
        for group_name, group_def in self.humanoid_joints.items():
            if joint_name in group_def['joints']:
                return group_def
        return None

    def publish_violation_marker(self, joint_name: str, actual_value: float, min_limit: float, max_limit: float):
        """Publish visualization marker for constraint violation"""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "joint_violations"
        marker.id = hash(joint_name) % 10000  # Simple hash to get unique ID
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0  # Above the robot
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.1  # Text size
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.text = f"{joint_name}: {actual_value:.3f}\nLimits: [{min_limit:.3f}, {max_limit:.3f}]"

        self.constraint_violation_pub.publish(marker)

    def run_comprehensive_test(self):
        """Run comprehensive joint constraint validation tests"""
        self.get_logger().info("Starting comprehensive joint constraint validation tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running joint constraint test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> JointConstraintResult:
        """Run a specific joint constraint validation scenario"""
        if scenario == 'joint_limit_enforcement':
            return self.run_limit_enforcement_test()
        elif scenario == 'velocity_constraint_validation':
            return self.run_velocity_constraint_test()
        elif scenario == 'effort_limit_verification':
            return self.run_effort_limit_test()
        elif scenario == 'kinematic_chain_integrity':
            return self.run_kinematic_chain_test()
        elif scenario == 'coupled_joint_coordination':
            return self.run_coupled_joint_test()
        elif scenario == 'extreme_pose_stability':
            return self.run_extreme_pose_test()
        else:
            return JointConstraintResult(
                test_name=scenario,
                joint_name='multi',
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_limit_enforcement_test(self) -> JointConstraintResult:
        """Test enforcement of joint position limits"""
        start_time = time.time()

        # Wait for data collection
        time.sleep(6)  # Allow time for position limit testing

        # Assess position limit compliance
        position_compliance = self.constraint_metrics['position_accuracy']
        limit_enforcement_ok = position_compliance >= 0.95  # 95% compliance required

        metrics = {
            'position_compliance_rate': position_compliance,
            'limit_enforcement_ok': limit_enforcement_ok,
            'duration': time.time() - start_time
        }

        passed = limit_enforcement_ok

        return JointConstraintResult(
            test_name='joint_limit_enforcement',
            joint_name='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Joint limit enforcement failed - Compliance: {position_compliance:.3f} < 0.95"
        )

    def run_velocity_constraint_test(self) -> JointConstraintResult:
        """Test velocity constraint compliance"""
        start_time = time.time()

        # Wait for velocity data collection
        time.sleep(5)

        # Assess velocity constraint compliance
        velocity_compliance = self.constraint_metrics['velocity_constraints']
        velocity_constraints_ok = velocity_compliance >= 0.90  # 90% compliance required

        metrics = {
            'velocity_compliance_rate': velocity_compliance,
            'velocity_constraints_ok': velocity_constraints_ok,
            'duration': time.time() - start_time
        }

        passed = velocity_constraints_ok

        return JointConstraintResult(
            test_name='velocity_constraint_validation',
            joint_name='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Velocity constraint validation failed - Compliance: {velocity_compliance:.3f} < 0.90"
        )

    def run_effort_limit_test(self) -> JointConstraintResult:
        """Test effort limit compliance"""
        start_time = time.time()

        # Wait for effort data collection
        time.sleep(4)

        # Assess effort limit compliance
        effort_compliance = self.constraint_metrics['effort_limits']
        effort_limits_ok = effort_compliance >= 0.85  # 85% compliance required

        metrics = {
            'effort_compliance_rate': effort_compliance,
            'effort_limits_ok': effort_limits_ok,
            'duration': time.time() - start_time
        }

        passed = effort_limits_ok

        return JointConstraintResult(
            test_name='effort_limit_verification',
            joint_name='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Effort limit verification failed - Compliance: {effort_compliance:.3f} < 0.85"
        )

    def run_kinematic_chain_test(self) -> JointConstraintResult:
        """Test kinematic chain integrity"""
        start_time = time.time()

        # For kinematic chain integrity, we'll check if joint positions are consistent
        # with the expected kinematic model
        time.sleep(5)

        # This would typically involve forward/inverse kinematics validation
        # For simulation, we'll use a placeholder metric
        kinematic_integrity_score = self.constraint_metrics['kinematic_chain_integrity']
        kinematic_integrity_ok = kinematic_integrity_score >= 0.8

        metrics = {
            'kinematic_integrity_score': kinematic_integrity_score,
            'kinematic_integrity_ok': kinematic_integrity_ok,
            'duration': time.time() - start_time
        }

        passed = kinematic_integrity_ok

        return JointConstraintResult(
            test_name='kinematic_chain_integrity',
            joint_name='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Kinematic chain integrity test failed - Score: {kinematic_integrity_score:.3f} < 0.80"
        )

    def run_coupled_joint_test(self) -> JointConstraintResult:
        """Test coupled joint coordination"""
        start_time = time.time()

        # Wait for coupled joint data
        time.sleep(4)

        # Assess coupled joint coordination
        coupling_score = self.constraint_metrics['joint_coupling']
        coupling_ok = coupling_score >= 0.75  # 75% coordination required

        metrics = {
            'coupling_score': coupling_score,
            'coupling_ok': coupling_ok,
            'duration': time.time() - start_time
        }

        passed = coupling_ok

        return JointConstraintResult(
            test_name='coupled_joint_coordination',
            joint_name='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Coupled joint coordination failed - Score: {coupling_score:.3f} < 0.75"
        )

    def run_extreme_pose_test(self) -> JointConstraintResult:
        """Test joint constraints under extreme poses"""
        start_time = time.time()

        # Wait for extreme pose data collection
        time.sleep(6)

        # For extreme poses, check that constraints still hold
        extreme_pose_stability = (
            self.constraint_metrics['position_accuracy'] >= 0.90 and
            self.constraint_metrics['velocity_constraints'] >= 0.85 and
            self.constraint_metrics['effort_limits'] >= 0.80
        )

        metrics = {
            'position_accuracy': self.constraint_metrics['position_accuracy'],
            'velocity_constraints': self.constraint_metrics['velocity_constraints'],
            'effort_limits': self.constraint_metrics['effort_limits'],
            'extreme_pose_stability': extreme_pose_stability,
            'duration': time.time() - start_time
        }

        passed = extreme_pose_stability

        return JointConstraintResult(
            test_name='extreme_pose_stability',
            joint_name='multi',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Extreme pose stability failed - Pos: {self.constraint_metrics['position_accuracy']:.3f}, Vel: {self.constraint_metrics['velocity_constraints']:.3f}, Eff: {self.constraint_metrics['effort_limits']:.3f}"
        )

    def get_joint_properties(self, joint_name: str) -> Optional[dict]:
        """Get current joint properties from Gazebo"""
        if self.get_joint_props_cli.service_is_ready():
            request = GetJointProperties.Request()
            request.joint_name = joint_name

            future = self.get_joint_props_cli.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            response = future.result()
            if response is not None:
                return {
                    'type': response.type,
                    'position': list(response.position),
                    'rate': list(response.rate),
                    'success': response.success,
                    'status_message': response.status_message
                }
        return None

    def set_joint_properties(self, joint_name: str, properties: dict) -> bool:
        """Set joint properties in Gazebo"""
        if self.set_joint_props_cli.service_is_ready():
            request = SetJointProperties.Request()
            request.joint_name = joint_name

            # Set properties based on the provided dict
            if 'ode_joint_config' in properties:
                request.ode_joint_config = properties['ode_joint_config']

            future = self.set_joint_props_cli.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            response = future.result()
            return response.success if response is not None else False
        return False

    def publish_test_results(self):
        """Publish joint constraint validation results"""
        results_msg = String()
        results_msg.data = json.dumps({
            'timestamp': time.time(),
            'test_results': [
                {
                    'test_name': r.test_name,
                    'joint_name': r.joint_name,
                    'passed': r.passed,
                    'metrics': r.metrics,
                    'duration': r.duration,
                    'error_message': r.error_message
                } for r in self.test_results
            ],
            'statistics': self.test_statistics,
            'constraint_metrics': self.constraint_metrics,
            'humanoid_joints': self.humanoid_joints
        })

        self.test_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive joint constraint validation report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'constraint_metrics': self.constraint_metrics.copy(),
            'humanoid_joint_definitions': self.humanoid_joints.copy(),
            'individual_results': [
                {
                    'test_name': r.test_name,
                    'joint_name': r.joint_name,
                    'passed': r.passed,
                    'duration': r.duration,
                    'metrics': r.metrics,
                    'error': r.error_message
                } for r in self.test_results
            ],
            'constraint_summary': self.generate_constraint_summary(),
            'violation_analysis': self.analyze_constraint_violations(),
            'recommendations': self.generate_recommendations()
        }

        return report

    def generate_constraint_summary(self) -> Dict:
        """Generate summary of constraint compliance by joint type"""
        summary = {}

        for group_name, group_def in self.humanoid_joints.items():
            # Calculate compliance for each joint group
            group_results = [r for r in self.test_results if r.joint_name in group_def['joints'] or r.joint_name == 'multi']

            if group_results:
                passed_count = sum(1 for r in group_results if r.passed)
                total_count = len(group_results)
                success_rate = passed_count / total_count if total_count > 0 else 0

                summary[group_name] = {
                    'joints': group_def['joints'],
                    'total_tests': total_count,
                    'passed_tests': passed_count,
                    'success_rate': success_rate
                }

        return summary

    def analyze_constraint_violations(self) -> Dict:
        """Analyze constraint violations by type and frequency"""
        violation_analysis = {
            'position_violations': 0,
            'velocity_violations': 0,
            'effort_violations': 0,
            'most_problematic_joints': [],
            'violation_frequency': {}
        }

        # In a real implementation, this would analyze actual violation data
        # For this simulation, we'll provide a template
        return violation_analysis

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on joint constraint validation results"""
        recommendations = []

        if self.constraint_metrics['position_accuracy'] < 0.95:
            recommendations.append("Improve position constraint enforcement - tighten position limits or adjust control parameters")

        if self.constraint_metrics['velocity_constraints'] < 0.90:
            recommendations.append("Improve velocity constraint enforcement - adjust velocity limits or control gains")

        if self.constraint_metrics['effort_limits'] < 0.85:
            recommendations.append("Improve effort limit compliance - adjust actuator models or control strategies")

        if self.constraint_metrics['joint_coupling'] < 0.75:
            recommendations.append("Improve coupled joint coordination - implement better synchronization algorithms")

        if self.constraint_metrics['kinematic_chain_integrity'] < 0.8:
            recommendations.append("Verify kinematic chain integrity - check joint limit configurations and link connections")

        if not recommendations:
            recommendations.append("Joint constraints are performing well - continue monitoring for optimization")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed joint constraint validation report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("JOINT CONSTRAINT VALIDATION COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("CONSTRAINT METRICS:")
        print("-" * 40)
        for metric, value in report['constraint_metrics'].items():
            print(f"  {metric}: {value:.3f}")

        print("\nJOINT GROUP SUMMARY:")
        print("-" * 40)
        for group, summary in report['constraint_summary'].items():
            print(f"  {group.upper()}: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['success_rate']*100:.1f}%)")

        print("\nINDIVIDUAL TEST RESULTS:")
        print("-" * 40)
        for result in report['individual_results']:
            status = "PASS" if result['passed'] else "FAIL"
            joint = result['joint_name']
            test = result['test_name']
            print(f"  {status:4} | {joint:15} | {test:25} | {result['duration']:.2f}s | {result['error'] or 'OK'}")

        print("\nVIOLATION ANALYSIS:")
        print("-" * 40)
        violation_analysis = report['violation_analysis']
        for key, value in violation_analysis.items():
            print(f"  {key}: {value}")

        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("="*80)


def main(args=None):
    """Main function to run joint constraint validation tests"""
    rclpy.init(args=args)

    validator = JointConstraintValidator()

    try:
        # Run comprehensive joint constraint validation tests
        results = validator.run_comprehensive_test()

        # Print detailed report
        validator.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Joint Constraint Validation Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        validator.get_logger().info("Joint constraint validation testing interrupted by user")
    finally:
        # Print final report
        validator.print_detailed_report()
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()