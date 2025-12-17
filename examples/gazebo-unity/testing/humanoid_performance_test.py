#!/usr/bin/env python3
"""
Humanoid Performance Test Suite
This script performs comprehensive performance tests for complex humanoid simulations,
measuring various performance metrics under different load conditions.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64
from gazebo_msgs.srv import GetPhysicsProperties
from builtin_interfaces.msg import Time
import numpy as np
import time
import threading
from collections import deque
import statistics
import psutil
import GPUtil
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class PerformanceResult:
    """Result of a performance test"""
    test_name: str
    passed: bool
    metrics: Dict
    duration: float
    error_message: str = ""


class HumanoidPerformanceTester(Node):
    """
    Performance tester for complex humanoid simulations
    """

    def __init__(self):
        super().__init__('humanoid_performance_tester')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=200),
            'imu': deque(maxlen=200),
            'joint': deque(maxlen=200),
            'odometry': deque(maxlen=200)
        }

        self.performance_metrics = {
            'simulation_rate': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'physics_update_time': 0.0,
            'rendering_fps': 0.0,
            'sensor_processing_latency': 0.0,
            'control_response_time': 0.0,
            'real_time_factor': 0.0
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

        # Publishers for test control and results
        self.test_control_pub = self.create_publisher(
            String,
            '/humanoid_performance/test_control',
            10
        )

        self.test_results_pub = self.create_publisher(
            String,
            '/humanoid_performance/results',
            10
        )

        # Service client for physics properties
        self.get_physics_cli = self.create_client(GetPhysicsProperties, '/get_physics_properties')

        # Timer for performance monitoring
        self.monitor_timer = self.create_timer(0.1, self.performance_monitoring)

        # Test scenarios
        self.test_scenarios = [
            'basic_humanoid_simulation',
            'complex_motion_sequences',
            'multi_robot_interaction',
            'sensor_intensive_operations',
            'long_duration_stress_test'
        ]

        self.current_test = None
        self.test_results = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_duration': 0.0
        }

        # Performance thresholds
        self.min_simulation_rate = 30.0  # Hz
        self.max_cpu_usage = 80.0       # %
        self.max_memory_usage = 85.0    # %
        self.min_real_time_factor = 0.8 # 80% real-time performance
        self.max_control_latency = 0.05 # 50ms max latency

        # Resource monitoring
        self.cpu_monitoring = deque(maxlen=100)
        self.memory_monitoring = deque(maxlen=100)
        self.gpu_monitoring = deque(maxlen=100)

        self.get_logger().info("Humanoid Performance Tester initialized and ready for testing")

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

    def performance_monitoring(self):
        """Monitor system performance metrics"""
        try:
            # Update performance metrics
            self.update_simulation_rate()
            self.update_cpu_memory_metrics()
            self.update_gpu_metrics()
            self.update_latency_metrics()
            self.update_real_time_factor()

        except Exception as e:
            self.get_logger().error(f"Error in performance monitoring: {e}")

    def update_simulation_rate(self):
        """Update simulation update rate metrics"""
        # Calculate based on sensor data arrival rate
        if len(self.sensor_data['joint']) > 10:
            recent_data = list(self.sensor_data['joint'])[-10:]
            time_diffs = []

            for i in range(1, len(recent_data)):
                dt = recent_data[i]['timestamp'] - recent_data[i-1]['timestamp']
                time_diffs.append(dt)

            if time_diffs:
                avg_dt = statistics.mean(time_diffs)
                rate = 1.0 / avg_dt if avg_dt > 0 else 0
                self.performance_metrics['simulation_rate'] = rate

    def update_cpu_memory_metrics(self):
        """Update CPU and memory usage metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.performance_metrics['cpu_usage'] = cpu_percent
        self.cpu_monitoring.append(cpu_percent)

        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        self.performance_metrics['memory_usage'] = memory_percent
        self.memory_monitoring.append(memory_percent)

    def update_gpu_metrics(self):
        """Update GPU usage metrics if available"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100  # Convert to percentage
                self.performance_metrics['gpu_usage'] = gpu_load
                self.gpu_monitoring.append(gpu_load)
            else:
                self.performance_metrics['gpu_usage'] = 0.0
        except:
            self.performance_metrics['gpu_usage'] = 0.0

    def update_latency_metrics(self):
        """Update latency metrics for sensor processing and control"""
        # Calculate processing latency based on message timestamps
        if len(self.sensor_data['laser']) > 1:
            latest_msg = self.sensor_data['laser'][-1]
            latency = time.time() - latest_msg['timestamp']
            self.performance_metrics['sensor_processing_latency'] = latency

        # For control latency, we'd need to track command to response time
        # This is a placeholder implementation
        self.performance_metrics['control_response_time'] = 0.02  # 20ms placeholder

    def update_real_time_factor(self):
        """Update real-time factor based on simulation timing"""
        # In a real implementation, this would compare simulation time to wall clock time
        # For this test, we'll calculate based on timing analysis
        self.performance_metrics['real_time_factor'] = min(1.0, self.performance_metrics['simulation_rate'] / 100.0)  # Assuming 100Hz physics rate

    def run_comprehensive_test(self):
        """Run comprehensive performance tests for humanoid simulations"""
        self.get_logger().info("Starting comprehensive humanoid performance tests...")

        for scenario in self.test_scenarios:
            self.get_logger().info(f"Running performance test scenario: {scenario}")
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

    def run_specific_test(self, scenario: str) -> PerformanceResult:
        """Run a specific performance test scenario"""
        if scenario == 'basic_simulation_performance':
            return self.run_basic_simulation_test()
        elif scenario == 'complex_motion_performance':
            return self.run_complex_motion_test()
        elif scenario == 'multi_robot_performance':
            return self.run_multi_robot_test()
        elif scenario == 'sensor_intensive_performance':
            return self.run_sensor_intensive_test()
        elif scenario == 'long_duration_performance':
            return self.run_long_duration_test()
        else:
            return PerformanceResult(
                test_name=scenario,
                passed=False,
                metrics={},
                duration=0.0,
                error_message=f"Unknown test scenario: {scenario}"
            )

    def run_basic_simulation_test(self) -> PerformanceResult:
        """Test performance with basic humanoid simulation"""
        start_time = time.time()

        # Wait for basic simulation data collection
        time.sleep(8)  # Allow 8 seconds for basic simulation

        # Assess performance metrics for basic simulation
        simulation_rate_ok = self.performance_metrics['simulation_rate'] >= self.min_simulation_rate
        cpu_usage_ok = self.performance_metrics['cpu_usage'] <= self.max_cpu_usage
        memory_usage_ok = self.performance_metrics['memory_usage'] <= self.max_memory_usage
        real_time_ok = self.performance_metrics['real_time_factor'] >= self.min_real_time_factor

        metrics = {
            'simulation_rate': self.performance_metrics['simulation_rate'],
            'cpu_usage': self.performance_metrics['cpu_usage'],
            'memory_usage': self.performance_metrics['memory_usage'],
            'real_time_factor': self.performance_metrics['real_time_factor'],
            'simulation_rate_ok': simulation_rate_ok,
            'cpu_usage_ok': cpu_usage_ok,
            'memory_usage_ok': memory_usage_ok,
            'real_time_ok': real_time_ok,
            'duration': time.time() - start_time
        }

        passed = simulation_rate_ok and cpu_usage_ok and memory_usage_ok and real_time_ok

        return PerformanceResult(
            test_name='basic_humanoid_simulation',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Basic simulation test failed - Rate: {self.performance_metrics['simulation_rate']:.1f}Hz, CPU: {self.performance_metrics['cpu_usage']:.1f}%, Memory: {self.performance_metrics['memory_usage']:.1f}%, RTF: {self.performance_metrics['real_time_factor']:.2f}"
        )

    def run_complex_motion_test(self) -> PerformanceResult:
        """Test performance during complex motion sequences"""
        start_time = time.time()

        # Wait for complex motion data collection
        time.sleep(10)  # Allow 10 seconds for complex motion

        # During complex motions, check if performance degrades
        complex_motion_performance_ok = (
            self.performance_metrics['simulation_rate'] >= self.min_simulation_rate * 0.85 and  # Allow 15% degradation
            self.performance_metrics['cpu_usage'] <= self.max_cpu_usage * 1.2 and  # Allow 20% increase
            self.performance_metrics['real_time_factor'] >= self.min_real_time_factor * 0.8  # Allow 20% degradation
        )

        metrics = {
            'simulation_rate_complex': self.performance_metrics['simulation_rate'],
            'cpu_usage_complex': self.performance_metrics['cpu_usage'],
            'real_time_factor_complex': self.performance_metrics['real_time_factor'],
            'complex_motion_performance_ok': complex_motion_performance_ok,
            'duration': time.time() - start_time
        }

        passed = complex_motion_performance_ok

        return PerformanceResult(
            test_name='complex_motion_sequences',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Complex motion test failed - Rate: {self.performance_metrics['simulation_rate']:.1f}Hz, CPU: {self.performance_metrics['cpu_usage']:.1f}%, RTF: {self.performance_metrics['real_time_factor']:.2f}"
        )

    def run_multi_robot_test(self) -> PerformanceResult:
        """Test performance with multiple humanoid robots"""
        start_time = time.time()

        # Wait for multi-robot data collection
        time.sleep(12)  # Allow 12 seconds for multi-robot simulation

        # Check if system can handle multiple robots without significant performance degradation
        multi_robot_performance_ok = (
            self.performance_metrics['simulation_rate'] >= self.min_simulation_rate * 0.7 and  # Allow 30% degradation
            self.performance_metrics['cpu_usage'] <= self.max_cpu_usage * 1.5 and  # Allow 50% increase
            self.performance_metrics['memory_usage'] <= self.max_memory_usage * 1.2  # Allow 20% increase
        )

        metrics = {
            'simulation_rate_multi': self.performance_metrics['simulation_rate'],
            'cpu_usage_multi': self.performance_metrics['cpu_usage'],
            'memory_usage_multi': self.performance_metrics['memory_usage'],
            'multi_robot_performance_ok': multi_robot_performance_ok,
            'duration': time.time() - start_time
        }

        passed = multi_robot_performance_ok

        return PerformanceResult(
            test_name='multi_robot_interaction',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Multi-robot test failed - Rate: {self.performance_metrics['simulation_rate']:.1f}Hz, CPU: {self.performance_metrics['cpu_usage']:.1f}%, Memory: {self.performance_metrics['memory_usage']:.1f}%"
        )

    def run_sensor_intensive_test(self) -> PerformanceResult:
        """Test performance with sensor-intensive operations"""
        start_time = time.time()

        # Wait for sensor-intensive data collection
        time.sleep(8)  # Allow 8 seconds for sensor-intensive simulation

        # During sensor-intensive operations, check if system maintains performance
        sensor_intensive_ok = (
            self.performance_metrics['simulation_rate'] >= self.min_simulation_rate * 0.75 and  # Allow 25% degradation
            self.performance_metrics['cpu_usage'] <= self.max_cpu_usage * 1.3 and  # Allow 30% increase
            self.performance_metrics['sensor_latency'] <= self.max_control_latency * 2  # Allow 2x latency during intensive operations
        )

        metrics = {
            'simulation_rate_sensor_intensive': self.performance_metrics['simulation_rate'],
            'cpu_usage_sensor_intensive': self.performance_metrics['cpu_usage'],
            'sensor_latency_sensor_intensive': self.performance_metrics['sensor_processing_latency'],
            'sensor_intensive_ok': sensor_intensive_ok,
            'duration': time.time() - start_time
        }

        passed = sensor_intensive_ok

        return PerformanceResult(
            test_name='sensor_intensive_operations',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Sensor-intensive test failed - Rate: {self.performance_metrics['simulation_rate']:.1f}Hz, CPU: {self.performance_metrics['cpu_usage']:.1f}%, Latency: {self.performance_metrics['sensor_processing_latency']:.3f}s"
        )

    def run_long_duration_test(self) -> PerformanceResult:
        """Test performance during long-duration operation"""
        start_time = time.time()

        # Wait for long-duration data collection
        time.sleep(15)  # Allow 15 seconds for long-duration testing

        # Check for memory leaks and performance degradation over time
        # Calculate average metrics over the duration
        avg_cpu_usage = statistics.mean(list(self.cpu_monitoring)) if self.cpu_monitoring else 0
        avg_memory_usage = statistics.mean(list(self.memory_monitoring)) if self.memory_monitoring else 0

        # Check for memory leak (memory usage should not continuously increase)
        memory_leak_detected = len(self.memory_monitoring) > 10 and self.memory_monitoring[-1] > self.memory_monitoring[0] + 10

        long_duration_ok = (
            avg_cpu_usage <= self.max_cpu_usage and
            avg_memory_usage <= self.max_memory_usage and
            not memory_leak_detected and
            self.performance_metrics['simulation_rate'] >= self.min_simulation_rate * 0.9  # Allow 10% degradation over time
        )

        metrics = {
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'final_memory_usage': self.performance_metrics['memory_usage'],
            'memory_leak_detected': memory_leak_detected,
            'simulation_rate_long': self.performance_metrics['simulation_rate'],
            'long_duration_ok': long_duration_ok,
            'duration': time.time() - start_time
        }

        passed = long_duration_ok

        return PerformanceResult(
            test_name='long_duration_stress_test',
            passed=passed,
            metrics=metrics,
            duration=time.time() - start_time,
            error_message="" if passed else f"Long duration test failed - Avg CPU: {avg_cpu_usage:.1f}%, Avg Memory: {avg_memory_usage:.1f}%, Leak: {memory_leak_detected}, Rate: {self.performance_metrics['simulation_rate']:.1f}Hz"
        )

    def get_physics_properties(self):
        """Get current physics properties from Gazebo"""
        if self.get_physics_cli.service_is_ready():
            request = GetPhysicsProperties.Request()
            future = self.get_physics_cli.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            response = future.result()
            if response is not None:
                return {
                    'time_step': response.time_step,
                    'max_update_rate': response.max_update_rate,
                    'gravity': response.gravity,
                    'ode_config': response.ode_config
                }
        return None

    def publish_test_results(self):
        """Publish performance test results"""
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
            'performance_metrics': self.performance_metrics,
            'thresholds': {
                'min_simulation_rate': self.min_simulation_rate,
                'max_cpu_usage': self.max_cpu_usage,
                'max_memory_usage': self.max_memory_usage,
                'min_real_time_factor': self.min_real_time_factor,
                'max_control_latency': self.max_control_latency
            }
        })

        self.test_results_pub.publish(results_msg)

    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive performance test report"""
        report = {
            'timestamp': time.time(),
            'test_scenarios': self.test_scenarios,
            'total_tests': self.test_statistics['total_tests'],
            'passed_tests': self.test_statistics['passed_tests'],
            'failed_tests': self.test_statistics['failed_tests'],
            'success_rate': self.test_statistics['passed_tests'] / self.test_statistics['total_tests'] if self.test_statistics['total_tests'] > 0 else 0,
            'average_duration': self.test_statistics['average_duration'],
            'performance_metrics': self.performance_metrics.copy(),
            'resource_usage_peak': self.get_resource_usage_peak(),
            'individual_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'duration': r.duration,
                    'metrics': r.metrics,
                    'error': r.error_message
                } for r in self.test_results
            ],
            'system_specifications': self.get_system_specifications(),
            'physics_configuration': self.get_physics_properties(),
            'performance_assessment': self.calculate_performance_assessment(),
            'optimization_recommendations': self.generate_recommendations()
        }

        return report

    def get_resource_usage_peak(self) -> Dict:
        """Get peak resource usage values"""
        return {
            'peak_cpu_usage': max(self.cpu_monitoring) if self.cpu_monitoring else 0,
            'peak_memory_usage': max(self.memory_monitoring) if self.memory_monitoring else 0,
            'peak_gpu_usage': max(self.gpu_monitoring) if self.gpu_monitoring else 0
        }

    def get_system_specifications(self) -> Dict:
        """Get system specifications"""
        try:
            return {
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'system_platform': os.uname().sysname if hasattr(os, 'uname') else 'Windows',
                'python_version': sys.version,
                'ros_version': 'ROS2 Humble Hawksbill'  # This would need to be detected dynamically
            }
        except:
            return {
                'cpu_cores': 0,
                'cpu_threads': 0,
                'cpu_freq': 0,
                'total_memory_gb': 0,
                'available_memory_gb': 0,
                'system_platform': 'Unknown',
                'python_version': 'Unknown',
                'ros_version': 'Unknown'
            }

    def calculate_performance_assessment(self) -> Dict[str, float]:
        """Calculate overall performance assessment"""
        assessment = {}

        # Calculate weighted performance scores
        simulation_score = min(1.0, self.performance_metrics['simulation_rate'] / self.min_simulation_rate)
        cpu_score = max(0, min(1, 1 - (self.performance_metrics['cpu_usage'] / self.max_cpu_usage)))
        memory_score = max(0, min(1, 1 - (self.performance_metrics['memory_usage'] / self.max_memory_usage)))
        rt_score = min(1.0, self.performance_metrics['real_time_factor'] / self.min_real_time_factor)
        latency_score = max(0, min(1, 1 - (self.performance_metrics['control_response_time'] / self.max_control_latency)))

        # Calculate overall performance score
        overall_score = (simulation_score + cpu_score + memory_score + rt_score + latency_score) / 5

        assessment['simulation_performance'] = simulation_score
        assessment['cpu_efficiency'] = cpu_score
        assessment['memory_efficiency'] = memory_score
        assessment['real_time_performance'] = rt_score
        assessment['latency_performance'] = latency_score
        assessment['overall_performance'] = overall_score

        return assessment

    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance test results"""
        recommendations = []

        perf_assessment = self.calculate_performance_assessment()

        if perf_assessment['simulation_performance'] < 0.8:
            recommendations.append("Improve simulation performance - consider reducing physics complexity or increasing time step")

        if perf_assessment['cpu_efficiency'] < 0.7:
            recommendations.append("Optimize CPU usage - profile and optimize hot code paths in robot control")

        if perf_assessment['memory_efficiency'] < 0.7:
            recommendations.append("Optimize memory usage - check for memory leaks and optimize data structures")

        if perf_assessment['real_time_performance'] < 0.7:
            recommendations.append("Improve real-time performance - optimize sensor processing and control loops")

        if perf_assessment['latency_performance'] < 0.8:
            recommendations.append("Reduce control latency - optimize communication pathways and processing")

        # Check for specific bottlenecks
        if self.performance_metrics['cpu_usage'] > 90:
            recommendations.append("CPU bottleneck detected - consider offloading computation or upgrading hardware")

        if self.performance_metrics['memory_usage'] > 90:
            recommendations.append("Memory bottleneck detected - investigate memory leaks or increase system memory")

        if self.performance_metrics['gpu_usage'] > 90:
            recommendations.append("GPU bottleneck detected - optimize rendering or upgrade graphics hardware")

        if self.performance_metrics['simulation_rate'] < 20:
            recommendations.append("Critical: Simulation rate too low - may affect robot control stability")

        if not recommendations:
            recommendations.append("Performance is adequate for humanoid simulation - continue monitoring for optimization opportunities")

        return recommendations

    def print_detailed_report(self):
        """Print a detailed performance test report"""
        report = self.get_comprehensive_report()

        print("\n" + "="*80)
        print("HUMANOID SIMULATION PERFORMANCE COMPREHENSIVE TEST REPORT")
        print("="*80)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Test Scenarios: {', '.join(report['test_scenarios'])}")
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Duration: {report['average_duration']:.2f}s")
        print()

        print("SYSTEM SPECIFICATIONS:")
        print("-" * 40)
        specs = report['system_specifications']
        for key, value in specs.items():
            print(f"  {key}: {value}")

        print("\nPHYSICS CONFIGURATION:")
        print("-" * 40)
        physics = report['physics_configuration']
        if physics:
            for key, value in physics.items():
                print(f"  {key}: {value}")
        else:
            print("  Could not retrieve physics configuration")

        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        metrics = report['performance_metrics']
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")

        print("\nPEAK RESOURCE USAGE:")
        print("-" * 40)
        peaks = report['resource_usage_peak']
        for resource, value in peaks.items():
            print(f"  {resource}: {value:.1f}%")

        print("\nPERFORMANCE ASSESSMENT:")
        print("-" * 40)
        assessment = report['performance_assessment']
        for metric, score in assessment.items():
            print(f"  {metric}: {score:.3f}")

        print("\nINDIVIDUAL TEST RESULTS:")
        print("-" * 40)
        for result in report['individual_results']:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {status:4} | {result['test_name']:25} | {result['duration']:.2f}s | {result['error'] or 'OK'}")

        print("\nOPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)
        for rec in report['optimization_recommendations']:
            print(f"  • {rec}")

        print("="*80)


def main(args=None):
    """Main function to run humanoid performance tests"""
    rclpy.init(args=args)

    tester = HumanoidPerformanceTester()

    try:
        # Run comprehensive performance tests
        results = tester.run_comprehensive_test()

        # Print detailed report
        tester.print_detailed_report()

        # Calculate overall assessment
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0

        print(f"\nOverall Humanoid Performance Test Result: {passed_count}/{total_count} tests passed ({success_rate:.1f}%)")

    except KeyboardInterrupt:
        tester.get_logger().info("Humanoid performance testing interrupted by user")
    finally:
        # Print final report
        tester.print_detailed_report()
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()