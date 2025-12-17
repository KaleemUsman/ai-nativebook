/**
 * @file bipedal_constraint_plugin.cpp
 * @brief Nav2 plugin implementing bipedal locomotion constraints for humanoid robots
 * 
 * This plugin provides path validation and trajectory constraints specific to
 * bipedal humanoid robot locomotion, including step length limits, stability
 * margins, and terrain compatibility checks.
 */

#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

namespace humanoid_nav2_plugins
{

/**
 * @brief Parameters for bipedal locomotion constraints
 */
struct BipedalConstraints
{
  double max_step_length = 0.30;      // Maximum step length (m)
  double max_step_width = 0.20;       // Maximum lateral step (m)
  double max_step_height = 0.15;      // Maximum step-up height (m)
  double min_step_time = 0.4;         // Minimum time per step (s)
  double max_lean_angle = 0.15;       // Maximum body lean angle (rad)
  double min_stability_margin = 0.05; // Minimum stability margin (m)
  double max_slope_angle = 0.26;      // Maximum traversable slope (~15 deg)
  double max_angular_velocity = 0.5;  // Maximum rotation speed (rad/s)
  double max_linear_velocity = 0.4;   // Maximum forward speed (m/s)
};

/**
 * @brief Bipedal constraint validation plugin for Nav2
 * 
 * This plugin validates paths and trajectories against bipedal locomotion
 * constraints to ensure safe humanoid robot navigation.
 */
class BipedalConstraintPlugin
{
public:
  BipedalConstraintPlugin() = default;
  ~BipedalConstraintPlugin() = default;

  /**
   * @brief Initialize the plugin with ROS parameters
   * @param parent Parent node for parameter access
   * @param name Plugin name
   */
  void initialize(
    const rclcpp::Node::SharedPtr & parent,
    const std::string & name)
  {
    node_ = parent;
    plugin_name_ = name;
    logger_ = node_->get_logger();

    // Declare and get parameters
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".max_step_length",
      rclcpp::ParameterValue(0.30));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".max_step_width",
      rclcpp::ParameterValue(0.20));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".max_step_height",
      rclcpp::ParameterValue(0.15));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".min_step_time",
      rclcpp::ParameterValue(0.4));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".max_lean_angle",
      rclcpp::ParameterValue(0.15));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".min_stability_margin",
      rclcpp::ParameterValue(0.05));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".max_slope_angle",
      rclcpp::ParameterValue(0.26));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".max_angular_velocity",
      rclcpp::ParameterValue(0.5));
    nav2_util::declare_parameter_if_not_declared(
      node_, plugin_name_ + ".max_linear_velocity",
      rclcpp::ParameterValue(0.4));

    node_->get_parameter(plugin_name_ + ".max_step_length", constraints_.max_step_length);
    node_->get_parameter(plugin_name_ + ".max_step_width", constraints_.max_step_width);
    node_->get_parameter(plugin_name_ + ".max_step_height", constraints_.max_step_height);
    node_->get_parameter(plugin_name_ + ".min_step_time", constraints_.min_step_time);
    node_->get_parameter(plugin_name_ + ".max_lean_angle", constraints_.max_lean_angle);
    node_->get_parameter(plugin_name_ + ".min_stability_margin", constraints_.min_stability_margin);
    node_->get_parameter(plugin_name_ + ".max_slope_angle", constraints_.max_slope_angle);
    node_->get_parameter(plugin_name_ + ".max_angular_velocity", constraints_.max_angular_velocity);
    node_->get_parameter(plugin_name_ + ".max_linear_velocity", constraints_.max_linear_velocity);

    RCLCPP_INFO(logger_, "BipedalConstraintPlugin initialized with max_step_length: %.2f m",
      constraints_.max_step_length);
  }

  /**
   * @brief Validate a path against bipedal constraints
   * @param path The path to validate
   * @return true if the path satisfies all bipedal constraints
   */
  bool validatePath(const nav_msgs::msg::Path & path)
  {
    if (path.poses.size() < 2) {
      return true;
    }

    for (size_t i = 1; i < path.poses.size(); ++i) {
      const auto & prev_pose = path.poses[i - 1].pose;
      const auto & curr_pose = path.poses[i].pose;

      // Check step length constraint
      double dx = curr_pose.position.x - prev_pose.position.x;
      double dy = curr_pose.position.y - prev_pose.position.y;
      double step_length = std::sqrt(dx * dx + dy * dy);

      if (step_length > constraints_.max_step_length) {
        RCLCPP_WARN(logger_, "Path segment %zu exceeds max step length: %.2f > %.2f",
          i, step_length, constraints_.max_step_length);
        return false;
      }

      // Check lateral step constraint
      double heading = std::atan2(dy, dx);
      double prev_yaw = getYawFromQuaternion(prev_pose.orientation);
      double lateral_component = std::abs(std::sin(heading - prev_yaw) * step_length);

      if (lateral_component > constraints_.max_step_width) {
        RCLCPP_WARN(logger_, "Path segment %zu exceeds max lateral step: %.2f > %.2f",
          i, lateral_component, constraints_.max_step_width);
        return false;
      }

      // Check height change (requires height map - simplified here)
      double dz = curr_pose.position.z - prev_pose.position.z;
      if (std::abs(dz) > constraints_.max_step_height) {
        RCLCPP_WARN(logger_, "Path segment %zu exceeds max step height: %.2f > %.2f",
          i, std::abs(dz), constraints_.max_step_height);
        return false;
      }

      // Check angular change
      double curr_yaw = getYawFromQuaternion(curr_pose.orientation);
      double angular_change = normalizeAngle(curr_yaw - prev_yaw);
      
      // Estimate time for this segment based on step length
      double segment_time = std::max(constraints_.min_step_time,
        step_length / constraints_.max_linear_velocity);
      double angular_velocity = std::abs(angular_change) / segment_time;

      if (angular_velocity > constraints_.max_angular_velocity) {
        RCLCPP_WARN(logger_, "Path segment %zu exceeds max angular velocity",
          i);
        return false;
      }
    }

    return true;
  }

  /**
   * @brief Apply bipedal constraints to a velocity command
   * @param cmd_vel Input velocity command
   * @return Constrained velocity command
   */
  geometry_msgs::msg::TwistStamped constrainVelocity(
    const geometry_msgs::msg::TwistStamped & cmd_vel)
  {
    auto constrained = cmd_vel;

    // Limit linear velocity
    double linear_vel = std::sqrt(
      cmd_vel.twist.linear.x * cmd_vel.twist.linear.x +
      cmd_vel.twist.linear.y * cmd_vel.twist.linear.y);

    if (linear_vel > constraints_.max_linear_velocity) {
      double scale = constraints_.max_linear_velocity / linear_vel;
      constrained.twist.linear.x *= scale;
      constrained.twist.linear.y *= scale;
    }

    // Limit angular velocity
    if (std::abs(cmd_vel.twist.angular.z) > constraints_.max_angular_velocity) {
      constrained.twist.angular.z = std::copysign(
        constraints_.max_angular_velocity, cmd_vel.twist.angular.z);
    }

    // Apply coupled constraints: reduce linear velocity during high rotation
    double angular_ratio = std::abs(constrained.twist.angular.z) / 
      constraints_.max_angular_velocity;
    if (angular_ratio > 0.5) {
      double linear_reduction = 1.0 - (angular_ratio - 0.5);
      constrained.twist.linear.x *= linear_reduction;
      constrained.twist.linear.y *= linear_reduction;
    }

    return constrained;
  }

  /**
   * @brief Check if a pose is stable for a humanoid robot
   * @param pose The pose to check
   * @param pitch Estimated terrain pitch at the pose
   * @param roll Estimated terrain roll at the pose
   * @return true if the pose is stable
   */
  bool isStablePose(
    const geometry_msgs::msg::Pose & pose,
    double pitch,
    double roll)
  {
    // Check terrain slope
    double slope = std::sqrt(pitch * pitch + roll * roll);
    if (slope > constraints_.max_slope_angle) {
      return false;
    }

    // Check body lean angle from pose orientation
    double body_pitch, body_roll, body_yaw;
    getEulerFromQuaternion(pose.orientation, body_roll, body_pitch, body_yaw);

    if (std::abs(body_pitch) > constraints_.max_lean_angle ||
        std::abs(body_roll) > constraints_.max_lean_angle)
    {
      return false;
    }

    return true;
  }

  /**
   * @brief Get the bipedal constraints
   */
  const BipedalConstraints & getConstraints() const
  {
    return constraints_;
  }

private:
  /**
   * @brief Extract yaw from quaternion
   */
  double getYawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
  {
    // Simplified: assumes mostly upright orientation
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  /**
   * @brief Extract Euler angles from quaternion
   */
  void getEulerFromQuaternion(
    const geometry_msgs::msg::Quaternion & q,
    double & roll, double & pitch, double & yaw)
  {
    // Roll (x-axis rotation)
    double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1.0) {
      pitch = std::copysign(M_PI / 2, sinp);
    } else {
      pitch = std::asin(sinp);
    }

    // Yaw (z-axis rotation)
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
  }

  /**
   * @brief Normalize angle to [-pi, pi]
   */
  double normalizeAngle(double angle)
  {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }

  rclcpp::Node::SharedPtr node_;
  std::string plugin_name_;
  rclcpp::Logger logger_{rclcpp::get_logger("bipedal_constraint_plugin")};
  BipedalConstraints constraints_;
};

}  // namespace humanoid_nav2_plugins

// Note: In a real ROS 2 package, you would register this as a plugin:
// PLUGINLIB_EXPORT_CLASS(humanoid_nav2_plugins::BipedalConstraintPlugin, nav2_core::Controller)
