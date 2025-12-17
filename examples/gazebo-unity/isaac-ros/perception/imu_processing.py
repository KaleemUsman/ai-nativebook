#!/usr/bin/env python3
"""
IMU Processing Node for Isaac ROS Perception Pipeline

This node processes IMU data for the humanoid robot's perception system,
performing orientation estimation, bias correction, and motion state tracking.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped, QuaternionStamped
from std_msgs.msg import Bool
import numpy as np
import threading
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class ImuState:
    """State container for IMU processing."""
    orientation: np.ndarray  # Quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # [x, y, z] rad/s
    linear_acceleration: np.ndarray  # [x, y, z] m/s^2
    gyro_bias: np.ndarray  # Estimated gyroscope bias
    accel_bias: np.ndarray  # Estimated accelerometer bias
    is_stationary: bool
    timestamp: float


class ComplementaryFilter:
    """
    Complementary filter for orientation estimation.
    
    Fuses gyroscope and accelerometer data for robust orientation estimation.
    """
    
    def __init__(self, alpha: float = 0.98):
        """
        Initialize complementary filter.
        
        Args:
            alpha: Weight for gyroscope integration (0-1).
                   Higher values trust gyroscope more.
        """
        self.alpha = alpha
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.last_timestamp = None
    
    def update(self, gyro: np.ndarray, accel: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Update orientation estimate with new sensor data.
        
        Args:
            gyro: Angular velocity [x, y, z] in rad/s
            accel: Linear acceleration [x, y, z] in m/s^2
            timestamp: Current timestamp in seconds
            
        Returns:
            Updated orientation as quaternion [w, x, y, z]
        """
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            # Initialize orientation from accelerometer
            self.orientation = self._accel_to_orientation(accel)
            return self.orientation
        
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        if dt <= 0 or dt > 1.0:
            return self.orientation
        
        # Integrate gyroscope
        gyro_orientation = self._integrate_gyro(self.orientation, gyro, dt)
        
        # Get orientation from accelerometer
        accel_orientation = self._accel_to_orientation(accel)
        
        # Complementary filter: blend gyro and accel orientations
        self.orientation = self._slerp(accel_orientation, gyro_orientation, self.alpha)
        self.orientation = self._normalize_quaternion(self.orientation)
        
        return self.orientation
    
    def _integrate_gyro(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Integrate angular velocity to update quaternion."""
        # Create quaternion from angular velocity
        omega_mag = np.linalg.norm(gyro)
        if omega_mag < 1e-10:
            return q
        
        axis = gyro / omega_mag
        angle = omega_mag * dt
        
        # Quaternion representing rotation
        dq = np.array([
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2)
        ])
        
        # Apply rotation
        return self._quaternion_multiply(q, dq)
    
    def _accel_to_orientation(self, accel: np.ndarray) -> np.ndarray:
        """Estimate orientation from accelerometer (roll and pitch only)."""
        ax, ay, az = accel
        
        # Normalize acceleration
        norm = np.linalg.norm(accel)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        ax, ay, az = accel / norm
        
        # Calculate roll and pitch
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        
        # Convert to quaternion (yaw = 0)
        return self._euler_to_quaternion(roll, pitch, 0.0)
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion."""
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)
        
        # Ensure shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            # Linear interpolation for nearly parallel quaternions
            result = q1 + t * (q2 - q1)
            return self._normalize_quaternion(result)
        
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        q_perp = q2 - q1 * dot
        q_perp = self._normalize_quaternion(q_perp)
        
        return q1 * np.cos(theta) + q_perp * np.sin(theta)
    
    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm


class ImuProcessingNode(Node):
    """
    IMU processing node for the Isaac ROS perception pipeline.
    
    Provides:
    - Orientation estimation using complementary filter
    - Gyroscope bias estimation
    - Motion state detection (stationary vs moving)
    - Filtered IMU data output
    """
    
    def __init__(self):
        super().__init__('imu_processing_node')
        
        # Declare parameters
        self.declare_parameter('filter_alpha', 0.98)
        self.declare_parameter('gyro_bias_alpha', 0.001)
        self.declare_parameter('stationary_threshold_gyro', 0.01)
        self.declare_parameter('stationary_threshold_accel', 0.1)
        self.declare_parameter('gravity_magnitude', 9.81)
        self.declare_parameter('publishing_rate', 100.0)
        
        # Get parameters
        self.filter_alpha = self.get_parameter('filter_alpha').value
        self.gyro_bias_alpha = self.get_parameter('gyro_bias_alpha').value
        self.stationary_threshold_gyro = self.get_parameter('stationary_threshold_gyro').value
        self.stationary_threshold_accel = self.get_parameter('stationary_threshold_accel').value
        self.gravity_magnitude = self.get_parameter('gravity_magnitude').value
        self.publishing_rate = self.get_parameter('publishing_rate').value
        
        # Initialize filter
        self.filter = ComplementaryFilter(alpha=self.filter_alpha)
        
        # Initialize state
        self.state = ImuState(
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3),
            is_stationary=True,
            timestamp=0.0
        )
        self.state_lock = threading.Lock()
        
        # Calibration state
        self.calibration_samples = []
        self.calibration_complete = False
        self.calibration_sample_count = 100
        
        # Create QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscriber
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.imu_callback,
            qos_profile
        )
        
        # Create publishers
        self.imu_filtered_pub = self.create_publisher(
            Imu,
            '/imu/data',
            10
        )
        
        self.orientation_pub = self.create_publisher(
            QuaternionStamped,
            '/imu/orientation',
            10
        )
        
        self.angular_velocity_pub = self.create_publisher(
            Vector3Stamped,
            '/imu/angular_velocity',
            10
        )
        
        self.linear_acceleration_pub = self.create_publisher(
            Vector3Stamped,
            '/imu/linear_acceleration',
            10
        )
        
        self.stationary_pub = self.create_publisher(
            Bool,
            '/imu/is_stationary',
            10
        )
        
        self.get_logger().info('IMU Processing Node initialized')
        self.get_logger().info('Waiting for calibration samples...')
    
    def imu_callback(self, msg: Imu):
        """Process incoming IMU data."""
        # Extract raw data
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Perform calibration if not complete
        if not self.calibration_complete:
            self._collect_calibration_sample(gyro, accel)
            return
        
        # Apply bias correction
        gyro_corrected = gyro - self.state.gyro_bias
        accel_corrected = accel - self.state.accel_bias
        
        # Detect stationary state
        is_stationary = self._detect_stationary(gyro_corrected, accel_corrected)
        
        # Update gyro bias during stationary periods
        if is_stationary:
            self.state.gyro_bias = (
                (1 - self.gyro_bias_alpha) * self.state.gyro_bias +
                self.gyro_bias_alpha * gyro
            )
        
        # Update orientation estimate
        orientation = self.filter.update(gyro_corrected, accel_corrected, timestamp)
        
        # Remove gravity from acceleration
        linear_accel = self._remove_gravity(accel_corrected, orientation)
        
        # Update state
        with self.state_lock:
            self.state.orientation = orientation
            self.state.angular_velocity = gyro_corrected
            self.state.linear_acceleration = linear_accel
            self.state.is_stationary = is_stationary
            self.state.timestamp = timestamp
        
        # Publish processed data
        self._publish_data(msg.header)
    
    def _collect_calibration_sample(self, gyro: np.ndarray, accel: np.ndarray):
        """Collect samples for initial bias calibration."""
        self.calibration_samples.append((gyro.copy(), accel.copy()))
        
        if len(self.calibration_samples) >= self.calibration_sample_count:
            # Compute initial biases
            gyro_samples = np.array([s[0] for s in self.calibration_samples])
            accel_samples = np.array([s[1] for s in self.calibration_samples])
            
            # Gyro bias is mean of stationary readings
            self.state.gyro_bias = np.mean(gyro_samples, axis=0)
            
            # Accel bias: subtract expected gravity vector (assuming sensor is level)
            mean_accel = np.mean(accel_samples, axis=0)
            expected_gravity = np.array([0.0, 0.0, self.gravity_magnitude])
            self.state.accel_bias = mean_accel - expected_gravity
            
            self.calibration_complete = True
            self.calibration_samples = []
            
            self.get_logger().info(
                f'Calibration complete. Gyro bias: {self.state.gyro_bias}, '
                f'Accel bias: {self.state.accel_bias}'
            )
    
    def _detect_stationary(self, gyro: np.ndarray, accel: np.ndarray) -> bool:
        """Detect if the IMU is stationary."""
        gyro_magnitude = np.linalg.norm(gyro)
        accel_deviation = abs(np.linalg.norm(accel) - self.gravity_magnitude)
        
        return (
            gyro_magnitude < self.stationary_threshold_gyro and
            accel_deviation < self.stationary_threshold_accel
        )
    
    def _remove_gravity(self, accel: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Remove gravity component from acceleration using orientation."""
        # Gravity in world frame
        gravity_world = np.array([0.0, 0.0, self.gravity_magnitude])
        
        # Rotate gravity to body frame
        gravity_body = self._rotate_vector_by_quaternion(
            gravity_world, self._quaternion_inverse(orientation)
        )
        
        # Subtract gravity
        return accel - gravity_body
    
    def _rotate_vector_by_quaternion(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        # Convert vector to quaternion form [0, x, y, z]
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        
        # Rotate: q * v * q^(-1)
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        
        # Quaternion multiplication
        temp = self._quaternion_multiply(q, v_quat)
        result = self._quaternion_multiply(temp, q_conj)
        
        return result[1:4]
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse (conjugate for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _publish_data(self, header):
        """Publish all processed IMU data."""
        with self.state_lock:
            state = ImuState(
                orientation=self.state.orientation.copy(),
                angular_velocity=self.state.angular_velocity.copy(),
                linear_acceleration=self.state.linear_acceleration.copy(),
                gyro_bias=self.state.gyro_bias.copy(),
                accel_bias=self.state.accel_bias.copy(),
                is_stationary=self.state.is_stationary,
                timestamp=self.state.timestamp
            )
        
        # Publish filtered IMU message
        imu_msg = Imu()
        imu_msg.header = header
        imu_msg.orientation.w = state.orientation[0]
        imu_msg.orientation.x = state.orientation[1]
        imu_msg.orientation.y = state.orientation[2]
        imu_msg.orientation.z = state.orientation[3]
        imu_msg.angular_velocity.x = state.angular_velocity[0]
        imu_msg.angular_velocity.y = state.angular_velocity[1]
        imu_msg.angular_velocity.z = state.angular_velocity[2]
        imu_msg.linear_acceleration.x = state.linear_acceleration[0]
        imu_msg.linear_acceleration.y = state.linear_acceleration[1]
        imu_msg.linear_acceleration.z = state.linear_acceleration[2]
        self.imu_filtered_pub.publish(imu_msg)
        
        # Publish orientation
        orientation_msg = QuaternionStamped()
        orientation_msg.header = header
        orientation_msg.quaternion.w = state.orientation[0]
        orientation_msg.quaternion.x = state.orientation[1]
        orientation_msg.quaternion.y = state.orientation[2]
        orientation_msg.quaternion.z = state.orientation[3]
        self.orientation_pub.publish(orientation_msg)
        
        # Publish angular velocity
        angular_vel_msg = Vector3Stamped()
        angular_vel_msg.header = header
        angular_vel_msg.vector.x = state.angular_velocity[0]
        angular_vel_msg.vector.y = state.angular_velocity[1]
        angular_vel_msg.vector.z = state.angular_velocity[2]
        self.angular_velocity_pub.publish(angular_vel_msg)
        
        # Publish linear acceleration
        linear_accel_msg = Vector3Stamped()
        linear_accel_msg.header = header
        linear_accel_msg.vector.x = state.linear_acceleration[0]
        linear_accel_msg.vector.y = state.linear_acceleration[1]
        linear_accel_msg.vector.z = state.linear_acceleration[2]
        self.linear_acceleration_pub.publish(linear_accel_msg)
        
        # Publish stationary state
        stationary_msg = Bool()
        stationary_msg.data = state.is_stationary
        self.stationary_pub.publish(stationary_msg)


def main(args=None):
    rclpy.init(args=args)
    
    imu_processing_node = ImuProcessingNode()
    
    try:
        rclpy.spin(imu_processing_node)
    except KeyboardInterrupt:
        pass
    finally:
        imu_processing_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
