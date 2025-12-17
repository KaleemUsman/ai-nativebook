#!/usr/bin/env python3
"""
AI Bridge Verification Script
This script verifies that AI agents can properly access sensor data
via the ROS 2 bridge system, including connection stability, data quality,
and access patterns.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState, Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import threading
from collections import deque
import statistics
import json
from typing import Dict, List, Tuple, Optional
import socket
import psutil


class AIBridgeVerifier(Node):
    """
    Verifies that AI agents can properly access sensor data via ROS 2 bridge
    """

    def __init__(self):
        super().__init__('ai_bridge_verifier')

        # Initialize data storage
        self.sensor_data = {
            'laser': deque(maxlen=50),
            'imu': deque(maxlen=50),
            'joint': deque(maxlen=50),
            'camera': deque(maxlen=10),
            'odometry': deque(maxlen=50)
        }

        self.connection_status = {
            'laser_connected': False,
            'imu_connected': False,
            'joint_connected': False,
            'camera_connected': False,
            'odometry_connected': False
        }

        self.data_quality_metrics = {
            'laser': {'latency': [], 'frequency': [], 'data_validity': []},
            'imu': {'latency': [], 'frequency': [], 'data_validity': []},
            'joint': {'latency': [], 'frequency': [], 'data_validity': []},
            'camera': {'latency': [], 'frequency': [], 'data_validity': []},
            'odometry': {'latency': [], 'frequency': [], 'data_validity': []}
        }

        self.ai_access_patterns = {
            'subscribed_topics': [],
            'access_frequency': {},
            'data_consumption_rate': {}
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

        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid/odom',
            self.odom_callback,
            10
        )

        # Publishers for verification results
        self.verification_status_pub = self.create_publisher(
            String,
            '/ai_bridge_verification/status',
            10
        )

        self.connection_status_pub = self.create_publisher(
            String,
            '/ai_bridge_verification/connection_status',
            10
        )

        # Timer for verification checks
        self.verification_timer = self.create_timer(1.0, self.run_verification_cycle)

        # Statistics
        self.stats = {
            'total_messages_received': 0,
            'verification_cycles': 0,
            'connection_stability_score': 0.0,
            'data_quality_score': 0.0,
            'accessibility_score': 0.0
        }

        self.get_logger().info("AI Bridge Verifier initialized and monitoring sensor access")

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        timestamp = time.time()
        self.sensor_data['laser'].append({
            'data': msg,
            'timestamp': timestamp,
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })
        self.connection_status['laser_connected'] = True
        self.stats['total_messages_received'] += 1

        # Calculate latency
        latency = timestamp - msg.header.stamp.sec - msg.header.stamp.nanosec / 1e9
        self.data_quality_metrics['laser']['latency'].append(latency)

    def imu_callback(self, msg):
        """Process IMU data"""
        timestamp = time.time()
        self.sensor_data['imu'].append({
            'data': msg,
            'timestamp': timestamp,
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })
        self.connection_status['imu_connected'] = True
        self.stats['total_messages_received'] += 1

        # Calculate latency
        latency = timestamp - msg.header.stamp.sec - msg.header.stamp.nanosec / 1e9
        self.data_quality_metrics['imu']['latency'].append(latency)

    def joint_callback(self, msg):
        """Process joint state data"""
        timestamp = time.time()
        self.sensor_data['joint'].append({
            'data': msg,
            'timestamp': timestamp,
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })
        self.connection_status['joint_connected'] = True
        self.stats['total_messages_received'] += 1

        # Calculate latency
        latency = timestamp - msg.header.stamp.sec - msg.header.stamp.nanosec / 1e9
        self.data_quality_metrics['joint']['latency'].append(latency)

    def camera_callback(self, msg):
        """Process camera image data"""
        timestamp = time.time()
        self.sensor_data['camera'].append({
            'data': msg,
            'timestamp': timestamp,
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })
        self.connection_status['camera_connected'] = True
        self.stats['total_messages_received'] += 1

        # Calculate latency
        latency = timestamp - msg.header.stamp.sec - msg.header.stamp.nanosec / 1e9
        self.data_quality_metrics['camera']['latency'].append(latency)

    def odom_callback(self, msg):
        """Process odometry data"""
        timestamp = time.time()
        self.sensor_data['odometry'].append({
            'data': msg,
            'timestamp': timestamp,
            'receive_time': self.get_clock().now().nanoseconds / 1e9
        })
        self.connection_status['odometry_connected'] = True
        self.stats['total_messages_received'] += 1

        # Calculate latency
        latency = timestamp - msg.header.stamp.sec - msg.header.stamp.nanosec / 1e9
        self.data_quality_metrics['odometry']['latency'].append(latency)

    def run_verification_cycle(self):
        """Run a cycle of verification checks"""
        try:
            # Update statistics
            self.stats['verification_cycles'] += 1

            # Check connection status
            self.check_connections()

            # Calculate data quality metrics
            self.calculate_data_quality_metrics()

            # Evaluate AI accessibility
            self.evaluate_ai_accessibility()

            # Calculate overall scores
            self.calculate_scores()

            # Publish status
            self.publish_verification_status()

            # Log periodic summary
            if self.stats['verification_cycles'] % 10 == 0:
                self.log_verification_summary()

        except Exception as e:
            self.get_logger().error(f"Error in verification cycle: {e}")

    def check_connections(self):
        """Check if all sensor connections are active"""
        # Check if we've received data in the last 5 seconds
        current_time = time.time()
        connection_timeout = 5.0

        for sensor_type in self.sensor_data.keys():
            if len(self.sensor_data[sensor_type]) > 0:
                latest_time = self.sensor_data[sensor_type][-1]['timestamp']
                if current_time - latest_time < connection_timeout:
                    self.connection_status[f'{sensor_type}_connected'] = True
                else:
                    self.connection_status[f'{sensor_type}_connected'] = False
            else:
                self.connection_status[f'{sensor_type}_connected'] = False

    def calculate_data_quality_metrics(self):
        """Calculate data quality metrics for each sensor type"""
        for sensor_type in self.sensor_data.keys():
            if len(self.sensor_data[sensor_type]) > 1:
                # Calculate frequency based on timestamps
                timestamps = [d['timestamp'] for d in self.sensor_data[sensor_type]]
                time_diffs = np.diff(timestamps)
                if len(time_diffs) > 0:
                    avg_frequency = 1.0 / np.mean(time_diffs)
                    self.data_quality_metrics[sensor_type]['frequency'].append(avg_frequency)

                # Calculate latency statistics
                latencies = self.data_quality_metrics[sensor_type]['latency']
                if len(latencies) > 0:
                    avg_latency = np.mean(latencies[-10:])  # Last 10 measurements
                    self.data_quality_metrics[sensor_type]['latency'] = latencies[-10:]  # Keep last 10

                # Calculate data validity (check for NaN, inf, etc.)
                if sensor_type == 'laser':
                    latest_scan = self.sensor_data[sensor_type][-1]['data']
                    ranges = np.array(latest_scan.ranges)
                    valid_count = np.sum(np.isfinite(ranges))
                    total_count = len(ranges)
                    validity = valid_count / total_count if total_count > 0 else 0
                    self.data_quality_metrics[sensor_type]['data_validity'].append(validity)

    def evaluate_ai_accessibility(self):
        """Evaluate how accessible the sensor data is for AI agents"""
        # Check if data is in expected format for AI processing
        accessibility_score = 0
        total_checks = 0

        # Check laser data accessibility
        if len(self.sensor_data['laser']) > 0:
            latest_scan = self.sensor_data['laser'][-1]['data']
            # Check if laser data is in expected format for AI
            if len(latest_scan.ranges) > 0 and np.isfinite(latest_scan.ranges[0]):
                accessibility_score += 1
            total_checks += 1

        # Check IMU data accessibility
        if len(self.sensor_data['imu']) > 0:
            latest_imu = self.sensor_data['imu'][-1]['data']
            # Check if IMU data is in expected format for AI
            if (latest_imu.orientation.w != 0 or
                latest_imu.angular_velocity.x != 0 or
                latest_imu.linear_acceleration.x != 0):
                accessibility_score += 1
            total_checks += 1

        # Check joint data accessibility
        if len(self.sensor_data['joint']) > 0:
            latest_joint = self.sensor_data['joint'][-1]['data']
            # Check if joint data is in expected format for AI
            if len(latest_joint.position) > 0:
                accessibility_score += 1
            total_checks += 1

        # Calculate accessibility score
        self.stats['accessibility_score'] = accessibility_score / total_checks if total_checks > 0 else 0

    def calculate_scores(self):
        """Calculate overall verification scores"""
        # Connection stability score (0-1)
        connected_count = sum(1 for status in self.connection_status.values() if status)
        total_connections = len(self.connection_status)
        self.stats['connection_stability_score'] = connected_count / total_connections if total_connections > 0 else 0

        # Data quality score based on latency and frequency
        avg_latency = 0
        avg_frequency = 0
        latency_count = 0
        frequency_count = 0

        for sensor_type in self.data_quality_metrics.keys():
            latencies = self.data_quality_metrics[sensor_type]['latency']
            frequencies = self.data_quality_metrics[sensor_type]['frequency']

            if len(latencies) > 0:
                avg_latency += np.mean(latencies)
                latency_count += 1

            if len(frequencies) > 0:
                avg_frequency += np.mean(frequencies)
                frequency_count += 1

        avg_latency = avg_latency / latency_count if latency_count > 0 else float('inf')
        avg_frequency = avg_frequency / frequency_count if frequency_count > 0 else 0

        # Normalize scores (latency should be low, frequency should be appropriate)
        latency_score = max(0, min(1, 0.1 / avg_latency)) if avg_latency != float('inf') else 0
        frequency_score = min(1, avg_frequency / 30.0) if avg_frequency != 0 else 0  # Assume 30Hz is good

        self.stats['data_quality_score'] = (latency_score + frequency_score) / 2

    def publish_verification_status(self):
        """Publish verification status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'timestamp': time.time(),
            'connection_status': self.connection_status,
            'data_quality': {
                'laser_latency_avg': np.mean(self.data_quality_metrics['laser']['latency']) if self.data_quality_metrics['laser']['latency'] else 0,
                'imu_latency_avg': np.mean(self.data_quality_metrics['imu']['latency']) if self.data_quality_metrics['imu']['latency'] else 0,
                'joint_latency_avg': np.mean(self.data_quality_metrics['joint']['latency']) if self.data_quality_metrics['joint']['latency'] else 0,
                'camera_latency_avg': np.mean(self.data_quality_metrics['camera']['latency']) if self.data_quality_metrics['camera']['latency'] else 0,
                'odometry_latency_avg': np.mean(self.data_quality_metrics['odometry']['latency']) if self.data_quality_metrics['odometry']['latency'] else 0,
            },
            'scores': self.stats
        })
        self.verification_status_pub.publish(status_msg)

    def log_verification_summary(self):
        """Log a summary of verification results"""
        connected_count = sum(1 for status in self.connection_status.values() if status)
        total_connections = len(self.connection_status)

        self.get_logger().info(
            f"Verification Summary - Cycles: {self.stats['verification_cycles']}, "
            f"Messages: {self.stats['total_messages_received']}, "
            f"Connections: {connected_count}/{total_connections}, "
            f"Stability: {self.stats['connection_stability_score']:.2f}, "
            f"Quality: {self.stats['data_quality_score']:.2f}, "
            f"Accessibility: {self.stats['accessibility_score']:.2f}"
        )

    def get_detailed_report(self) -> Dict:
        """Generate a detailed verification report"""
        report = {
            'timestamp': time.time(),
            'connection_status': self.connection_status.copy(),
            'data_quality_metrics': {},
            'ai_accessibility': {
                'accessibility_score': self.stats['accessibility_score'],
                'format_compliance': self.check_data_format_compliance()
            },
            'network_performance': self.get_network_performance(),
            'resource_utilization': self.get_resource_utilization(),
            'overall_scores': {
                'connection_stability': self.stats['connection_stability_score'],
                'data_quality': self.stats['data_quality_score'],
                'ai_accessibility': self.stats['accessibility_score'],
                'overall': (self.stats['connection_stability_score'] +
                           self.stats['data_quality_score'] +
                           self.stats['accessibility_score']) / 3
            },
            'statistics': self.stats.copy()
        }

        # Add detailed metrics for each sensor type
        for sensor_type in self.data_quality_metrics.keys():
            metrics = self.data_quality_metrics[sensor_type]
            report['data_quality_metrics'][sensor_type] = {
                'latency_avg': np.mean(metrics['latency']) if metrics['latency'] else 0,
                'latency_std': np.std(metrics['latency']) if metrics['latency'] else 0,
                'frequency_avg': np.mean(metrics['frequency']) if metrics['frequency'] else 0,
                'validity_avg': np.mean(metrics['data_validity']) if metrics['data_validity'] else 0,
                'sample_count': len(metrics['latency'])
            }

        return report

    def check_data_format_compliance(self) -> Dict[str, bool]:
        """Check if sensor data formats comply with AI agent expectations"""
        compliance = {}

        # Check laser data format
        if len(self.sensor_data['laser']) > 0:
            latest_scan = self.sensor_data['laser'][-1]['data']
            compliance['laser'] = {
                'has_ranges': len(latest_scan.ranges) > 0,
                'ranges_valid': all(np.isfinite(r) for r in latest_scan.ranges),
                'has_expected_fields': all(hasattr(latest_scan, attr)
                                        for attr in ['angle_min', 'angle_max', 'range_min', 'range_max'])
            }
        else:
            compliance['laser'] = {'compliant': False}

        # Check IMU data format
        if len(self.sensor_data['imu']) > 0:
            latest_imu = self.sensor_data['imu'][-1]['data']
            compliance['imu'] = {
                'has_orientation': latest_imu.orientation.w != 0,
                'has_angular_velocity': latest_imu.angular_velocity.x != 0,
                'has_linear_acceleration': latest_imu.linear_acceleration.x != 0
            }
        else:
            compliance['imu'] = {'compliant': False}

        return compliance

    def get_network_performance(self) -> Dict[str, float]:
        """Get network performance metrics"""
        # Get network I/O statistics
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }

    def get_resource_utilization(self) -> Dict[str, float]:
        """Get system resource utilization"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }

    def print_detailed_report(self):
        """Print a detailed verification report"""
        report = self.get_detailed_report()

        print("\n" + "="*70)
        print("AI BRIDGE VERIFICATION DETAILED REPORT")
        print("="*70)
        print(f"Timestamp: {time.ctime(report['timestamp'])}")
        print(f"Verification Cycles: {report['statistics']['verification_cycles']}")
        print(f"Total Messages: {report['statistics']['total_messages_received']}")
        print()

        print("CONNECTION STATUS:")
        print("-" * 30)
        for sensor, status in report['connection_status'].items():
            status_str = "CONNECTED" if status else "DISCONNECTED"
            print(f"  {sensor.upper()}: {status_str}")

        print("\nDATA QUALITY METRICS:")
        print("-" * 30)
        for sensor_type, metrics in report['data_quality_metrics'].items():
            print(f"  {sensor_type.upper()}:")
            print(f"    Latency: {metrics['latency_avg']:.3f}s avg ±{metrics['latency_std']:.3f}s")
            print(f"    Frequency: {metrics['frequency_avg']:.2f} Hz avg")
            print(f"    Validity: {metrics['validity_avg']:.2f} avg")
            print(f"    Samples: {metrics['sample_count']}")

        print("\nAI ACCESSIBILITY:")
        print("-" * 30)
        print(f"  Accessibility Score: {report['ai_accessibility']['accessibility_score']:.2f}")
        print("  Format Compliance:")
        for sensor, compliance in report['ai_accessibility']['format_compliance'].items():
            print(f"    {sensor.upper()}: {compliance}")

        print("\nOVERALL SCORES:")
        print("-" * 30)
        scores = report['overall_scores']
        for name, score in scores.items():
            print(f"  {name.replace('_', ' ').title()}: {score:.2f}")

        print("="*70)

    def verify_ai_agent_access(self, expected_topics: List[str]) -> bool:
        """
        Verify that AI agents can access expected sensor topics
        """
        accessible_topics = []

        # Check if expected topics have active publishers
        for topic in expected_topics:
            try:
                # Check topic info to see if there are publishers
                topic_info = self.get_publishers_info_by_topic(topic)
                if len(topic_info) > 0:
                    accessible_topics.append(topic)
            except:
                continue

        # Calculate accessibility percentage
        if len(expected_topics) > 0:
            accessibility_rate = len(accessible_topics) / len(expected_topics)
            return accessibility_rate >= 0.8  # Require 80% accessibility
        else:
            return True


def main(args=None):
    """Main function to run AI bridge verification"""
    rclpy.init(args=args)

    verifier = AIBridgeVerifier()

    try:
        # Run for a period of time to collect verification data
        start_time = time.time()
        run_duration = 15  # Run for 15 seconds

        while time.time() - start_time < run_duration:
            rclpy.spin_once(verifier, timeout_sec=0.1)

        # Print detailed report
        verifier.print_detailed_report()

        # Verify AI agent access to expected topics
        expected_topics = [
            '/humanoid/scan',
            '/humanoid/imu/data',
            '/joint_states',
            '/humanoid/camera/color/image_raw',
            '/humanoid/odom'
        ]

        ai_access_ok = verifier.verify_ai_agent_access(expected_topics)
        print(f"\nAI Agent Access Verification: {'PASSED' if ai_access_ok else 'FAILED'}")
        print(f"Expected topics: {expected_topics}")

    except KeyboardInterrupt:
        verifier.get_logger().info("AI Bridge verification interrupted by user")
    finally:
        # Print final report
        verifier.print_detailed_report()
        verifier.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()