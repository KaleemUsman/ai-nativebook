#!/usr/bin/env python3
"""
Camera Processing Node for Isaac ROS Perception Pipeline

This node processes camera images for the humanoid robot's perception system,
performing rectification, feature extraction, and initial processing.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
import threading


class CameraProcessingNode(Node):
    """
    Camera processing node for the Isaac ROS perception pipeline
    """
    def __init__(self):
        super().__init__('camera_processing_node')

        # Declare parameters
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('processing_rate', 30.0)
        self.declare_parameter('enable_rectification', True)
        self.declare_parameter('enable_feature_extraction', True)
        self.declare_parameter('enable_distortion_correction', True)

        # Get parameters
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.processing_rate = self.get_parameter('processing_rate').value
        self.enable_rectification = self.get_parameter('enable_rectification').value
        self.enable_feature_extraction = self.get_parameter('enable_feature_extraction').value
        self.enable_distortion_correction = self.get_parameter('enable_distortion_correction').value

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create QoS profile for image topics
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            qos_profile
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            qos_profile
        )

        # Create publishers
        self.rectified_pub = self.create_publisher(
            Image,
            '/camera/rgb/image_rect_color',
            1
        )

        if self.enable_feature_extraction:
            self.feature_pub = self.create_publisher(
                # In a real implementation, this would be a custom message type
                Image,  # Placeholder - would be custom feature message in practice
                '/camera/features',
                1
            )

        # Initialize variables
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.latest_image = None
        self.latest_camera_info = None
        self.image_lock = threading.Lock()

        # Create processing timer
        self.process_timer = self.create_timer(
            1.0 / self.processing_rate,
            self.process_image
        )

        self.get_logger().info('Camera Processing Node initialized')

    def camera_info_callback(self, msg):
        """
        Callback for camera info messages
        """
        with self.image_lock:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """
        Callback for image messages
        """
        with self.image_lock:
            self.latest_image = msg
            if self.latest_camera_info is not None:
                # Update camera parameters if new info is available
                self.camera_matrix = np.array(self.latest_camera_info.k).reshape(3, 3)
                self.distortion_coeffs = np.array(self.latest_camera_info.d)

    def process_image(self):
        """
        Process the latest image with rectification and feature extraction
        """
        with self.image_lock:
            if self.latest_image is None or self.camera_matrix is None:
                return

            try:
                # Convert ROS image to OpenCV
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')

                # Apply rectification if enabled
                if self.enable_rectification and self.enable_distortion_correction:
                    # Apply undistortion
                    h, w = cv_image.shape[:2]
                    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                        self.camera_matrix, self.distortion_coeffs, (w, h), 1, (w, h)
                    )
                    rectified_image = cv2.undistort(
                        cv_image, self.camera_matrix, self.distortion_coeffs, None, new_camera_matrix
                    )

                    # Crop image based on ROI
                    x, y, w, h = roi
                    rectified_image = rectified_image[y:y+h, x:x+w]
                else:
                    rectified_image = cv_image

                # Resize image if needed
                if rectified_image.shape[1] != self.image_width or rectified_image.shape[0] != self.image_height:
                    rectified_image = cv2.resize(rectified_image, (self.image_width, self.image_height))

                # Publish rectified image
                rectified_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding='bgr8')
                rectified_msg.header = self.latest_image.header
                self.rectified_pub.publish(rectified_msg)

                # Extract features if enabled
                if self.enable_feature_extraction:
                    features = self.extract_features(rectified_image)
                    # In a real implementation, publish feature message
                    # feature_msg = self.create_feature_message(features, self.latest_image.header)
                    # self.feature_pub.publish(feature_msg)

                self.get_logger().debug(f'Processed image: {rectified_image.shape}')

            except Exception as e:
                self.get_logger().error(f'Error processing image: {e}')

    def extract_features(self, image):
        """
        Extract features from the image using various algorithms
        """
        features = {}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Feature detection using ORB (for efficiency)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if keypoints is not None:
            features['keypoints'] = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
            features['descriptors'] = descriptors
            features['count'] = len(keypoints)

        # Additional feature extraction algorithms could be added here
        # For example: SIFT, SURF, FAST, etc.

        return features

    def create_feature_message(self, features, header):
        """
        Create a feature message from extracted features
        Note: This is a simplified representation
        """
        # In a real implementation, this would create a custom message
        # with the appropriate feature data structure
        pass


def main(args=None):
    rclpy.init(args=args)

    camera_processing_node = CameraProcessingNode()

    try:
        rclpy.spin(camera_processing_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_processing_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()