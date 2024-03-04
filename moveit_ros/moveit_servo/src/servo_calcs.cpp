/*******************************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2019, Los Alamos National Security, LLC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

/*      Title     : servo_calcs.cpp
 *      Project   : moveit_servo
 *      Created   : 1/11/2019
 *      Author    : Brian O'Neil, Andy Zelenak, Blake Anderson
 */

#include <cassert>
#include <thread>
#include <chrono>
#include <mutex>

#include <realtime_tools/thread_priority.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <moveit_servo/servo_calcs.h>
#include <moveit_servo/utilities.h>
#include "ik_common/dynamically_adjustable_ik.hpp"
#include <yaml-cpp/yaml.h>

// Disable -Wold-style-cast because all _THROTTLE macros trigger this
// It would be too noisy to disable on a per-callsite basis
#pragma GCC diagnostic ignored "-Wold-style-cast"

using namespace std::chrono_literals;  // for s, ms, etc.

namespace moveit_servo
{
namespace
{
static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_servo.servo_calcs");
constexpr auto ROS_LOG_THROTTLE_PERIOD = std::chrono::milliseconds(3000).count();
static constexpr double STOPPED_VELOCITY_EPS = 1e-4;  // rad/s

// This value is used when configuring the main loop to use SCHED_FIFO scheduling
// We use a slightly lower priority than the ros2_control default in order to reduce jitter
// Reference: https://man7.org/linux/man-pages/man2/sched_setparam.2.html
int const THREAD_PRIORITY = 40;
}  // namespace

// Constructor for the class that handles servoing calculations
ServoCalcs::ServoCalcs(const rclcpp::Node::SharedPtr& node,
                       const planning_scene_monitor::PlanningSceneMonitorPtr& planning_scene_monitor,
                       const std::shared_ptr<const servo::ParamListener>& servo_param_listener,
                       CollisionCheck & collision_checker)
  : node_(node)
  , servo_param_listener_(servo_param_listener)
  , servo_params_(servo_param_listener_->get_params())
  , planning_scene_monitor_(planning_scene_monitor)
  , stop_requested_(true)
  , collision_checker_(collision_checker)
  , smoothing_loader_("moveit_core", "online_signal_smoothing::SmoothingBaseClass")

{
  // MoveIt Setup
  current_state_ = planning_scene_monitor_->getStateMonitor()->getCurrentState();
  joint_model_group_ = current_state_->getJointModelGroup(servo_params_.move_group_name);
  if (joint_model_group_ == nullptr)
  {
    RCLCPP_ERROR_STREAM(LOGGER, "Invalid move group name: `" << servo_params_.move_group_name << '`');
    throw std::runtime_error("Invalid move group name");
  }

  // Subscribe to settings
  ee_frame_id_sub_ = node_->create_subscription<std_msgs::msg::String>(
      "~/ee_frame_id", rclcpp::QoS(1),
      [this](const std_msgs::msg::String::ConstSharedPtr& msg) { return eeFrameIdCB(msg); });

  // Subscribe to command topics
  twist_stamped_sub_ = node_->create_subscription<geometry_msgs::msg::TwistStamped>(
      servo_params_.cartesian_command_in_topic, rclcpp::QoS(1).best_effort(),
      [this](const geometry_msgs::msg::TwistStamped::ConstSharedPtr& msg) { return twistStampedCB(msg); });

  joint_cmd_sub_ = node_->create_subscription<control_msgs::msg::JointJog>(
      servo_params_.joint_command_in_topic, rclcpp::QoS(1).best_effort(),
      [this](const control_msgs::msg::JointJog::ConstSharedPtr& msg) { return jointCmdCB(msg); });

  // ROS Server for allowing drift in some dimensions
  drift_dimensions_server_ = node_->create_service<spinbot_msgs::srv::ChangeDriftDimensions>(
      "~/change_drift_dimensions",
      [this](const std::shared_ptr<spinbot_msgs::srv::ChangeDriftDimensions::Request>& req,
             const std::shared_ptr<spinbot_msgs::srv::ChangeDriftDimensions::Response>& res) {
        return changeDriftDimensions(req, res);
      });

  // // Subscribe to the collision_check topic
  // collision_velocity_scale_sub_ = node_->create_subscription<std_msgs::msg::Float64>(
  //     "~/collision_velocity_scale", rclcpp::SystemDefaultsQoS(),
  //     [this](const std_msgs::msg::Float64::ConstSharedPtr& msg) { return collisionVelocityScaleCB(msg); });

  // Publish freshly-calculated joints to the robot.
  // Put the outgoing msg in the right format (trajectory_msgs/JointTrajectory or std_msgs/Float64MultiArray).
  if (servo_params_.command_out_type == "trajectory_msgs/JointTrajectory")
  {
    trajectory_outgoing_cmd_pub_ = node_->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        servo_params_.command_out_topic, rclcpp::SystemDefaultsQoS());
  }
  else if (servo_params_.command_out_type == "std_msgs/Float64MultiArray")
  {
    multiarray_outgoing_cmd_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(
        servo_params_.command_out_topic, rclcpp::SystemDefaultsQoS());
  }
  debug_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/servo_debug", rclcpp::SystemDefaultsQoS());

  // Publish status
  status_pub_ = node_->create_publisher<std_msgs::msg::Int8>(servo_params_.status_topic, rclcpp::SystemDefaultsQoS());

  current_joint_state_.name = joint_model_group_->getActiveJointModelNames();
  num_joints_ = current_joint_state_.name.size();
  current_joint_state_.position.resize(num_joints_);
  current_joint_state_.velocity.resize(num_joints_);
  delta_theta_.setZero(num_joints_);

  for (std::size_t i = 0; i < num_joints_; ++i)
  {
    // A map for the indices of incoming joint commands
    joint_state_name_map_[current_joint_state_.name[i]] = i;
  }

  // Load the smoothing plugin
  try
  {
    smoother_ = smoothing_loader_.createUniqueInstance(servo_params_.smoothing_filter_plugin_name);
  }
  catch (pluginlib::PluginlibException& ex)
  {
    RCLCPP_ERROR(LOGGER, "Exception while loading the smoothing plugin '%s': '%s'",
                 servo_params_.smoothing_filter_plugin_name.c_str(), ex.what());
    std::exit(EXIT_FAILURE);
  }

  // Initialize the smoothing plugin
  if (!smoother_->initialize(node_, planning_scene_monitor_->getRobotModel(), num_joints_))
  {
    RCLCPP_ERROR(LOGGER, "Smoothing plugin could not be initialized");
    std::exit(EXIT_FAILURE);
  }

  // A matrix of all zeros is used to check whether matrices have been initialized
  Eigen::Matrix3d empty_matrix;
  empty_matrix.setZero();
  tf_moveit_to_ee_frame_ = empty_matrix;
  tf_moveit_to_robot_cmd_frame_ = empty_matrix;

  // Get the IK solver for the group
  ik_solver_ = joint_model_group_->getSolverInstance();
  if (!ik_solver_)
  {
    RCLCPP_WARN(
        LOGGER,
        "No kinematics solver instantiated for group '%s'. Will use inverse Jacobian for servo calculations instead.",
        joint_model_group_->getName().c_str());
  }
  else if (!ik_solver_->supportsGroup(joint_model_group_))
  {
    ik_solver_ = nullptr;
    RCLCPP_WARN(LOGGER,
                "The loaded kinematics plugin does not support group '%s'. Will use inverse Jacobian for servo "
                "calculations instead.",
                joint_model_group_->getName().c_str());
  }

  if (ik_solver_)
    ee_frame_id_ = ik_solver_->getTipFrames().front();
  else
    ee_frame_id_ = joint_model_group_->getLinkModelNames().back();
}

ServoCalcs::~ServoCalcs()
{
  stop();
}

void ServoCalcs::start()
{
  // Stop the thread if we are currently running
  stop();

  // Set up the "last" published message, in case we need to send it first
  auto initial_joint_trajectory = std::make_unique<trajectory_msgs::msg::JointTrajectory>();
  initial_joint_trajectory->header.stamp = node_->now();
  initial_joint_trajectory->header.frame_id = servo_params_.planning_frame;
  initial_joint_trajectory->joint_names = current_joint_state_.name;
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.time_from_start = rclcpp::Duration::from_seconds(0.0);
  if (servo_params_.publish_joint_positions)
  {
    planning_scene_monitor_->getStateMonitor()->getCurrentState()->copyJointGroupPositions(joint_model_group_,
                                                                                           point.positions);
  }
  if (servo_params_.publish_joint_velocities)
  {
    std::vector<double> velocity(num_joints_);
    point.velocities = velocity;
  }
  if (servo_params_.publish_joint_accelerations)
  {
    // I do not know of a robot that takes acceleration commands.
    // However, some controllers check that this data is non-empty.
    // Send all zeros, for now.
    point.accelerations.resize(num_joints_);
  }
  reloadMovementLimits();
  initial_joint_trajectory->points.push_back(point);
  last_sent_command_ = std::move(initial_joint_trajectory);

  current_state_ = planning_scene_monitor_->getStateMonitor()->getCurrentState();
  current_state_->copyJointGroupPositions(joint_model_group_, current_joint_state_.position);
  current_state_->copyJointGroupVelocities(joint_model_group_, current_joint_state_.velocity);
  // set previous state to same as current state for t = 0
  previous_joint_state_ = current_joint_state_;

  // Check that all links are known to the robot
  auto check_link_is_known = [this](const std::string& frame_name) {
    if (!current_state_->knowsFrameTransform(frame_name))
    {
      throw std::runtime_error{ "Unknown frame: " + frame_name };
    }
  };
  check_link_is_known(servo_params_.planning_frame);
  check_link_is_known(servo_params_.ee_frame_name);
  check_link_is_known(servo_params_.robot_link_command_frame);

  tf_moveit_to_ee_frame_ = current_state_->getGlobalLinkTransform(servo_params_.planning_frame).inverse() *
                           current_state_->getGlobalLinkTransform(servo_params_.ee_frame_name);
  tf_moveit_to_robot_cmd_frame_ = current_state_->getGlobalLinkTransform(servo_params_.planning_frame).inverse() *
                                  current_state_->getGlobalLinkTransform(servo_params_.robot_link_command_frame);

  // Always reset the low-pass filters when first starting servo
  resetLowPassFilters(current_joint_state_);

  stop_requested_ = false;
  // thread_ = std::thread([this] { mainCalcLoop(); });
  thread_ = std::thread([this] {
    // Check if a realtime kernel is installed. Set a higher thread priority, if so
    if (realtime_tools::has_realtime_kernel())
    {
      if (!realtime_tools::configure_sched_fifo(THREAD_PRIORITY))
      {
        RCLCPP_WARN(LOGGER, "Could not enable FIFO RT scheduling policy");
      }
    }
    else
    {
      RCLCPP_INFO(LOGGER, "RT kernel is recommended for better performance");
    }
    mainCalcLoop();
  });
  new_input_cmd_ = false;
}

void ServoCalcs::stop()
{
  // Request stop
  stop_requested_ = true;

  // Notify condition variable in case the thread is blocked on it
  {
    // scope so the mutex is unlocked after so the thread can continue
    // and therefore be joinable
    const std::lock_guard<std::mutex> lock(main_loop_mutex_);
    new_input_cmd_ = false;
    input_cv_.notify_all();
  }

  // Join the thread
  if (thread_.joinable())
  {
    thread_.join();
  }
}

void ServoCalcs::updateParams()
{
  if (servo_param_listener_->is_old(servo_params_))
  {
    auto params = servo_param_listener_->get_params();
    if (params.override_velocity_scaling_factor != servo_params_.override_velocity_scaling_factor)
    {
      RCLCPP_INFO_STREAM(LOGGER, "override_velocity_scaling_factor changed to : "
                                     << std::to_string(params.override_velocity_scaling_factor));
    }

    if (params.robot_link_command_frame != servo_params_.robot_link_command_frame)
    {
      if (current_state_->knowsFrameTransform(params.robot_link_command_frame))
      {
        RCLCPP_INFO_STREAM(LOGGER, "robot_link_command_frame changed to : " << params.robot_link_command_frame);
      }
      else
      {
        RCLCPP_ERROR_STREAM(LOGGER, "Failed to change robot_link_command_frame. Passed frame '"
                                        << params.robot_link_command_frame
                                        << "' is unknown, will keep using old command frame.");
        // Replace frame in new param set with old frame value
        // TODO : Is there a better behaviour here ?
        params.robot_link_command_frame = servo_params_.robot_link_command_frame;
      }
    }
    servo_params_ = params;

    if (params.movement_limits_file != servo_params_.movement_limits_file) {
      reloadMovementLimits();
    }
  }
}

void ServoCalcs::reloadMovementLimits()
{
  static const double inf = std::numeric_limits<double>::infinity();
  auto clear_limits = [this]() {
    movement_limits_.joint_limits.clear();
    movement_limits_.ee_pos_limits = Eigen::AlignedBox3d(Eigen::Vector3d(-inf, -inf, -inf), Eigen::Vector3d(inf, inf, inf));
    movement_limits_.max_ee_velocity = inf;
  };
  if (servo_params_.movement_limits_file.empty()) {
    clear_limits();
    return;
  }
  RCLCPP_INFO(LOGGER, "Loading movement_limits_file %s", servo_params_.movement_limits_file.c_str());
  try {
    YAML::Node limits_yaml = YAML::LoadFile(servo_params_.movement_limits_file);
    if (!limits_yaml) {
      RCLCPP_ERROR(LOGGER, "movement_limits_file invalid");
      clear_limits();
      return;
    }

    for (auto joint: limits_yaml) {
      if (joint.first.as<std::string>() == "tcp") {
        auto tcp_yaml = joint.second;
        if (auto limit = tcp_yaml["max_velocity"])
          movement_limits_.max_ee_velocity = limit.as<double>();
        Eigen::Vector3d min_pos(-inf, -inf, -inf);
        Eigen::Vector3d max_pos(-inf, -inf, -inf);
        if (auto limit = tcp_yaml["min_x"])
          min_pos[0] = limit.as<double>();
        if (auto limit = tcp_yaml["min_y"])
          min_pos[1] = limit.as<double>();
        if (auto limit = tcp_yaml["min_z"])
          min_pos[2] = limit.as<double>();
        if (auto limit = tcp_yaml["max_x"])
          max_pos[0] = limit.as<double>();
        if (auto limit = tcp_yaml["max_y"])
          max_pos[1] = limit.as<double>();
        if (auto limit = tcp_yaml["max_z"])
          max_pos[2] = limit.as<double>();
        movement_limits_.ee_pos_limits = Eigen::AlignedBox3d(min_pos, max_pos);
      }
      else {
        auto joint_yaml = joint.second;
        JointMovementLimits joint_limits;
        if (auto limit = joint_yaml["max_velocity"])
          joint_limits.max_velocity = limit.as<double>();
        if (auto limit = joint_yaml["max_acceleration"])
          joint_limits.max_acceleration = limit.as<double>();
        movement_limits_.joint_limits[joint.first.as<std::string>()] = joint_limits;
        RCLCPP_INFO(LOGGER, "Adding joint limit %s vel %lf acc %lf", joint.first.as<std::string>().c_str(), joint_limits.max_velocity, joint_limits.max_acceleration);
      }
    }
  }
  catch (const YAML::Exception &e) {
    RCLCPP_ERROR(LOGGER, "YAML error when loading movement limits: %s", e.what());
    clear_limits();
  }
}

void ServoCalcs::mainCalcLoop()
{
  rclcpp::WallRate rate(1.0 / servo_params_.publish_period);

  while (rclcpp::ok() && !stop_requested_)
  {
    // lock the input state mutex
    std::unique_lock<std::mutex> main_loop_lock(main_loop_mutex_);

    // Check if any parameters changed
    if (servo_params_.enable_parameter_update)
    {
      updateParams();
    }

    // low latency mode -- begin calculations as soon as a new command is received.
    if (servo_params_.low_latency_mode)
    {
      input_cv_.wait(main_loop_lock, [this] { return (new_input_cmd_ || stop_requested_); });
    }

    // reset new_input_cmd_ flag
    new_input_cmd_ = false;

    // run servo calcs
    const auto start_time = node_->now();
    calculateSingleIteration();
    const auto run_duration = node_->now() - start_time;

    // Log warning when the run duration was longer than the period
    if (run_duration.seconds() > servo_params_.publish_period)
    {
      rclcpp::Clock& clock = *node_->get_clock();
      RCLCPP_WARN_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD,
                                  "run_duration: " << run_duration.seconds() << " (" << servo_params_.publish_period
                                                   << ')');
    }

    // normal mode, unlock input mutex and wait for the period of the loop
    if (!servo_params_.low_latency_mode)
    {
      main_loop_lock.unlock();
      rate.sleep();
    }
  }
}

void ServoCalcs::calculateSingleIteration()
{
  // Publish status each loop iteration
  auto status_msg = std::make_unique<std_msgs::msg::Int8>();
  status_msg->data = static_cast<int8_t>(status_);
  status_pub_->publish(std::move(status_msg));

  // After we publish, status, reset it back to no warnings
  status_ = StatusCode::NO_WARNING;

  // Always update the joints and end-effector transform for 2 reasons:
  // 1) in case the getCommandFrameTransform() method is being used
  // 2) so the low-pass filters are up to date and don't cause a jump
  // Get the latest joint group positions
  // current_state_ = planning_scene_monitor_->getStateMonitor()->getCurrentState();
  // current_state_->copyJointGroupPositions(joint_model_group_, current_joint_state_.position);
  // current_state_->copyJointGroupVelocities(joint_model_group_, current_joint_state_.velocity);
  auto real_state = planning_scene_monitor_->getStateMonitor()->getCurrentState();
  auto real_state_msg = current_joint_state_;
  real_state->copyJointGroupPositions(joint_model_group_, real_state_msg.position);
  real_state->copyJointGroupVelocities(joint_model_group_, real_state_msg.velocity);
  for (size_t i = 0; i < real_state_msg.position.size(); i++) {
    current_joint_state_.position[i] =
        (1.0 - servo_params_.joint_state_aligning_factor)*current_joint_state_.position[i] +
        servo_params_.joint_state_aligning_factor*real_state_msg.position[i];
    current_joint_state_.velocity[i] =
        (1.0 - servo_params_.joint_state_aligning_factor)*current_joint_state_.velocity[i] +
        servo_params_.joint_state_aligning_factor*real_state_msg.velocity[i];
  }
  current_state_->setJointGroupPositions(joint_model_group_, current_joint_state_.position);
  current_state_->setJointGroupVelocities(joint_model_group_, current_joint_state_.velocity);

  // copy current state to temp state to use for calculating next state
  // This is done so that current_joint_state_ is preserved and can be used as backup.
  // All computations related to computing state q(t + dt) acts only on next_joint_state_ variable.
  next_joint_state_ = current_joint_state_;

  if (latest_twist_stamped_)
    twist_stamped_cmd_ = *latest_twist_stamped_;
  if (latest_joint_cmd_)
    joint_servo_cmd_ = *latest_joint_cmd_;

  // Check for stale cmds
  twist_command_is_stale_ = ((node_->now() - latest_twist_command_stamp_) >=
                             rclcpp::Duration::from_seconds(servo_params_.incoming_command_timeout));
  joint_command_is_stale_ = ((node_->now() - latest_joint_command_stamp_) >=
                             rclcpp::Duration::from_seconds(servo_params_.incoming_command_timeout));

  // Get the transform from MoveIt planning frame to servoing command frame
  // Calculate this transform to ensure it is available via C++ API
  // We solve (planning_frame -> base -> robot_link_command_frame)
  // by computing (base->planning_frame)^-1 * (base->robot_link_command_frame)
  tf_moveit_to_robot_cmd_frame_ = current_state_->getGlobalLinkTransform(servo_params_.planning_frame).inverse() *
                                  current_state_->getGlobalLinkTransform(servo_params_.robot_link_command_frame);

  // Calculate the transform from MoveIt planning frame to End Effector frame
  // Calculate this transform to ensure it is available via C++ API
  tf_moveit_to_ee_frame_ = current_state_->getGlobalLinkTransform(servo_params_.planning_frame).inverse() *
                           current_state_->getGlobalLinkTransform(servo_params_.ee_frame_name);

  // Don't end this function without updating the filters
  updated_filters_ = false;

  // If waiting for initial servo commands, just keep the low-pass filters up to date with current
  // joints so a jump doesn't occur when restarting
  if (wait_for_servo_commands_)
  {
    resetLowPassFilters(current_joint_state_);

    // Check if there are any new commands with valid timestamp
    wait_for_servo_commands_ =
        twist_stamped_cmd_.header.stamp == rclcpp::Time(0.) && joint_servo_cmd_.header.stamp == rclcpp::Time(0.);

    // Early exit
    return;
  }

  // If not waiting for initial command,
  // Do servoing calculations only if the robot should move, for efficiency
  // Create new outgoing joint trajectory command message
  auto joint_trajectory = std::make_unique<trajectory_msgs::msg::JointTrajectory>();

  // Prioritize cartesian servoing above joint servoing
  // Only run commands if not stale
  if (!twist_command_is_stale_)
  {
    if (!cartesianServoCalcs(twist_stamped_cmd_, *joint_trajectory))
    {
      resetLowPassFilters(current_joint_state_);
      return;
    }
  }
  else if (!joint_command_is_stale_)
  {
    if (!jointServoCalcs(joint_servo_cmd_, *joint_trajectory))
    {
      resetLowPassFilters(current_joint_state_);
      return;
    }
  }

  // Skip servoing publication if both types of commands are stale.
  if (twist_command_is_stale_ && joint_command_is_stale_)
  {
    rclcpp::Clock& clock = *node_->get_clock();
    RCLCPP_DEBUG_STREAM_THROTTLE(LOGGER, clock, 100,
                                 "Skipping publishing because incoming commands are stale.");
    filteredHalt(*joint_trajectory);
  }

  // Clear out position commands if user did not request them (can cause interpolation issues)
  if (!servo_params_.publish_joint_positions)
  {
    joint_trajectory->points[0].positions.clear();
  }
  // Likewise for velocity and acceleration
  if (!servo_params_.publish_joint_velocities)
  {
    joint_trajectory->points[0].velocities.clear();
  }
  if (!servo_params_.publish_joint_accelerations)
  {
    joint_trajectory->points[0].accelerations.clear();
  }

  // Put the outgoing msg in the right format
  // (trajectory_msgs/JointTrajectory or std_msgs/Float64MultiArray).
  if (servo_params_.command_out_type == "trajectory_msgs/JointTrajectory")
  {
    // When a joint_trajectory_controller receives a new command, a stamp of 0 indicates "begin immediately"
    // See http://wiki.ros.org/joint_trajectory_controller#Trajectory_replacement
    joint_trajectory->header.stamp = rclcpp::Time(0);
    *last_sent_command_ = *joint_trajectory;
    trajectory_outgoing_cmd_pub_->publish(std::move(joint_trajectory));
  }
  else if (servo_params_.command_out_type == "std_msgs/Float64MultiArray")
  {
    auto joints = std::make_unique<std_msgs::msg::Float64MultiArray>();
    if (servo_params_.publish_joint_positions && !joint_trajectory->points.empty())
    {
      joints->data = joint_trajectory->points[0].positions;
    }
    else if (servo_params_.publish_joint_velocities && !joint_trajectory->points.empty())
    {
      joints->data = joint_trajectory->points[0].velocities;
    }
    
    *last_sent_command_ = *joint_trajectory;
    multiarray_outgoing_cmd_pub_->publish(std::move(joints));

    auto debug_msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
    debug_msg->data = debug_data_;
    debug_pub_->publish(std::move(debug_msg));
  }

  // Update the filters if we haven't yet
  if (!updated_filters_)
    resetLowPassFilters(current_joint_state_);
}

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &stream, const std::vector<double> &vec)
{
  for (auto &v : vec)
    stream << v << "\n";
  return stream;
}

namespace
{
double velocityLimitsScalingFactor(const moveit::core::JointModelGroup* joint_model_group, const Eigen::VectorXd& velocity)
{
  std::size_t joint_delta_index{ 0 };
  double velocity_scaling_factor{ 1.0 };
  for (const moveit::core::JointModel* joint : joint_model_group->getActiveJointModels())
  {
    const auto& bounds = joint->getVariableBounds(joint->getName());
    if (bounds.velocity_bounded_ && velocity(joint_delta_index) != 0.0)
    {
      const double unbounded_velocity = velocity(joint_delta_index);
      // Clamp each joint velocity to a joint specific [min_velocity, max_velocity] range.
      const auto bounded_velocity = std::min(std::max(unbounded_velocity, bounds.min_velocity_), bounds.max_velocity_);
      // if (velocity.any())
      //   RCLCPP_INFO(LOGGER, "uv: %lf, min: %lf, max: %lf, bv: %lf", unbounded_velocity, bounds.min_velocity_, bounds.max_velocity_, bounded_velocity);
      velocity_scaling_factor = std::min(velocity_scaling_factor, bounded_velocity / unbounded_velocity);
    }
    ++joint_delta_index;
  }

  return velocity_scaling_factor;
}

}  // namespace

// Perform the servoing calculations
bool ServoCalcs::cartesianServoCalcs(geometry_msgs::msg::TwistStamped& cmd,
                                     trajectory_msgs::msg::JointTrajectory& joint_trajectory)
{
  // Check for nan's in the incoming command
  if (!checkValidCommand(cmd))
    return false;

  // Transform the command to the MoveGroup planning frame
  if (cmd.header.frame_id.empty())
  {
    RCLCPP_WARN_STREAM_THROTTLE(LOGGER, *node_->get_clock(), ROS_LOG_THROTTLE_PERIOD,
                                "No frame specified for command, will use planning_frame: "
                                    << servo_params_.planning_frame);
    cmd.header.frame_id = servo_params_.planning_frame;
  }
  if (cmd.header.frame_id != servo_params_.planning_frame)
  {
    transformTwistToPlanningFrame(cmd, servo_params_.planning_frame, current_state_);
  }

  Eigen::VectorXd delta_x = scaleCartesianCommand(cmd);

  bool use_inv_jacobian = use_inv_jacobian_;
  // if (!use_inv_jacobian
  //   && std::find(drift_dimensions_.begin(), drift_dimensions_.end(), true) == drift_dimensions_.end())
  // {
  //   RCLCPP_WARN_ONCE(LOGGER, "IK solving not supported with drift dimensions, using inverse jacobian."
  //     "This will be printed only once.");
  //   use_inv_jacobian = true;
  // }

  // Convert from cartesian commands to joint commands
  // Use an IK solver plugin if we have one, otherwise use inverse Jacobian.
  // Also use inverse Jacobian if drift dimensions are used, since IK solvers don't support them

  Eigen::MatrixXd jacobian_full = current_state_->getJacobian(joint_model_group_);
  Eigen::MatrixXd jacobian = jacobian_full;
  Eigen::VectorXd delta_x_full = delta_x;

  double delta_x_norm_weighted = 0;
  for (size_t i = 0; i < drift_dimensions_.size(); i++)
    if (!drift_dimensions_[i]) {
      double dim = delta_x_full[i]*servo_params_.drift_speed_correction_nondrifting_dimension_multipliers[i];
      delta_x_norm_weighted += dim*dim;
    }
  delta_x_norm_weighted = sqrt(delta_x_norm_weighted);
  double best_score = std::numeric_limits<double>::infinity();
  auto robot_state = *current_state_;
  double score_lookahead_interval = (servo_params_.ik_lookahead_seconds + servo_params_.publish_period) / servo_params_.publish_period / 2;
  std::string solution_name = "";

  const Eigen::Isometry3d base_to_tip_frame_transform =
      current_state_->getGlobalLinkTransform(ik_solver_->getBaseFrame()).inverse() *
      current_state_->getGlobalLinkTransform(ee_frame_id_);
  ik_base_to_tip_frame_ = base_to_tip_frame_transform;

  if (!use_inv_jacobian)
  {
    double lookahead_interval = servo_params_.ik_lookahead_seconds / servo_params_.publish_period;

    geometry_msgs::msg::Pose next_pose = poseFromCartesianDelta(delta_x, base_to_tip_frame_transform, lookahead_interval);

    // // for debug logging only //
    // if (delta_x.any()) {
    //   Eigen::Isometry3d next_pose_eigen;
    //   tf2::fromMsg(next_pose, next_pose_eigen);
    //   RCLCPP_INFO_STREAM(LOGGER, "\n\n\n\n\n\n\n\n\ndelta_x\n" << delta_x << "\nnext_pose\n" << next_pose_eigen.matrix());
    // }
    // ////////////////////////////

    removeDriftDimensions(jacobian, delta_x);

    // setup for IK call
    moveit_msgs::msg::MoveItErrorCodes err;
    kinematics::KinematicsQueryOptions opts;
    opts.return_approximate_solution = true;
    const ik_common::DynamicallyAdjustableIK *daik = dynamic_cast<const ik_common::DynamicallyAdjustableIK *>(ik_solver_.get());
    if (daik)
      daik->setOrientationVsPositionWeight(servo_params_.ik_rotation_error_multiplier);
    const auto &tip_frames = ik_solver_->getTipFrames();
    std::vector<geometry_msgs::msg::Pose> tip_frame_target_poses;
    std::vector<double> solution(num_joints_);
    for (auto &f : tip_frames) {
      if (f == ee_frame_id_)
        tip_frame_target_poses.push_back(next_pose);
      else {
        tip_frame_target_poses.emplace_back();
        tip_frame_target_poses.back().orientation.w = std::numeric_limits<double>::infinity();
      }
    }
    if (ik_solver_->searchPositionIK(tip_frame_target_poses, current_joint_state_.position, servo_params_.publish_period / 2.0,
                                     std::vector<double>(), solution, kinematics::KinematicsBase::IKCallbackFn(), err, opts))
    {
      // find the difference in joint positions that will get us to the desired pose
      for (size_t i = 0; i < num_joints_; ++i)
      {
        delta_theta_.coeffRef(i) = (solution.at(i) - current_joint_state_.position.at(i))/lookahead_interval;
      }
      
      // // for debug logging only //
      // if (delta_x.any()) {
      //   auto solution_state = robot_state;
      //   solution_state.setJointGroupPositions(joint_model_group_, solution);
      //   auto next_tip_frame = solution_state.getGlobalLinkTransform(ik_solver_->getBaseFrame()).inverse()*
      //       solution_state.getGlobalLinkTransform(ee_frame_id_);
      //   RCLCPP_INFO_STREAM(LOGGER,
      //                      "orig_theta\n" << current_joint_state_.position << "\norig_x\n"
      //                                     << ik_base_to_tip_frame_.matrix() <<
      //                                     "\nsolution_theta\n" << solution << "\nsolution_x\n"
      //                                     << next_tip_frame.matrix() << "\ndelta_theta\n" << delta_theta_
      //                                     << "\ntarget_pose\n" << geometry_msgs::msg::to_yaml(next_pose));
      // }
      // ////////////////////////////

      // if (Eigen::VectorXd(delta_theta_).norm() >= servo_params_.publish_period*0.05) {
      double direction_error;
      best_score = solutionScore(
            delta_theta_, delta_x, delta_x_norm_weighted, jacobian_full, robot_state, score_lookahead_interval, "IK", &direction_error);
      if (direction_error == direction_error) {
        // if (delta_x.any())
        //   RCLCPP_INFO(LOGGER, "dir error: %lf, factor: %lf", direction_error,
        //               pow(1 + direction_error, servo_params_.ik_direction_error_slowdown_factor));
        delta_theta_ /= pow(1 + direction_error, servo_params_.ik_direction_error_slowdown_factor);
      }
      solution_name = "IK";
      // }
    }
    else
    {
      RCLCPP_WARN(LOGGER, "Could not find IK solution for requested motion, got error code %d", err.val);
      // RCLCPP_WARN_THROTTLE(LOGGER, *node_->get_clock(), 500, "Could not find IK solution for requested motion, got error code %d", err.val);
    }

    if (daik && drift_dimensions_[3] && drift_dimensions_[4] && drift_dimensions_[5]) {
      daik->setOrientationVsPositionWeight(0);
      if (ik_solver_->searchPositionIK(tip_frame_target_poses, current_joint_state_.position, servo_params_.publish_period / 2.0,
                                     std::vector<double>(), solution, kinematics::KinematicsBase::IKCallbackFn(), err, opts))
      {
        Eigen::VectorXd delta_theta_pos(jacobian.cols());
        // find the difference in joint positions that will get us to the desired pose
        for (size_t i = 0; i < num_joints_; ++i)
        {
          delta_theta_pos.coeffRef(i) = (solution.at(i) - current_joint_state_.position.at(i))/lookahead_interval;
        }

        // // for debug logging only //
        // if (delta_x.any()) {
        //   robot_state.setJointGroupPositions(joint_model_group_, solution);
        //   auto next_tip_frame = robot_state.getGlobalLinkTransform(ik_solver_->getBaseFrame()).inverse()*
        //                         robot_state.getGlobalLinkTransform(ee_frame_id_);
        //   RCLCPP_INFO_STREAM(LOGGER,
        //                      "IKPOS orig_theta\n" << current_joint_state_.position << "\norig_x\n"
        //                                               << ik_base_to_tip_frame_.matrix() <<
        //                                               "\nsolution_theta\n" << solution << "\nsolution_x\n"
        //                                               << next_tip_frame.matrix());
        // }
        // ////////////////////////////

        // if (delta_theta_pos.norm() >= servo_params_.publish_period*0.05) {
        bool using_ikp = true;
        if (std::isfinite(best_score)) {
          double direction_error;
          double ikp_score = solutionScore(
                delta_theta_pos, delta_x, delta_x_norm_weighted, jacobian_full, robot_state, score_lookahead_interval, "IKP", &direction_error);
          if (direction_error == direction_error) {
            delta_theta_pos /= pow(1 + direction_error, servo_params_.ik_direction_error_slowdown_factor);
          }
          if (ikp_score < best_score)
            best_score = ikp_score;
          else
            using_ikp = false;
        }
        if (using_ikp) {
          delta_theta_ = delta_theta_pos;
          solution_name = "IK POSITIONAL";
        }
        // }
      }
      else
      {
        RCLCPP_WARN(LOGGER, "Could not find IKP solution for requested motion, got error code %d", err.val);
        // RCLCPP_WARN_THROTTLE(LOGGER, *node_->get_clock(), 500, "Could not find IK solution for requested motion, got error code %d", err.val);
      }
    }
  }
  else
    removeDriftDimensions(jacobian, delta_x);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd =
      Eigen::JacobiSVD<Eigen::MatrixXd>(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXd matrix_s = svd.singularValues().asDiagonal();
  Eigen::MatrixXd pseudo_inverse = svd.matrixV() * matrix_s.inverse() * svd.matrixU().transpose();

  bool using_jacobi = true;
  {
    // in case of active drift_dimensions we will compare quality of the IK solution against the pseudo_inverse approach and pick the better solution
    // Jacobi
    Eigen::VectorXd delta_theta_jacobi = pseudo_inverse * delta_x;
    // const StatusCode last_status = status_;
    double singularity_scale_jacobi = velocityScalingFactorForSingularity(joint_model_group_, delta_x, svd, pseudo_inverse,
                                                    servo_params_.hard_stop_singularity_threshold,
                                                    servo_params_.lower_singularity_threshold,
                                                    servo_params_.leaving_singularity_threshold_multiplier,
                                                    current_state_, status_);
    // if (last_status != status_)
    // {
    //   RCLCPP_WARN_STREAM_THROTTLE(LOGGER, *node_->get_clock(), ROS_LOG_THROTTLE_PERIOD, SERVO_STATUS_CODE_MAP.at(status_));
    // }
    delta_theta_jacobi *= singularity_scale_jacobi;
    if (std::isfinite(best_score)) {
      double jacobi_score = solutionScore(
        delta_theta_jacobi, delta_x, delta_x_norm_weighted,
        jacobian_full, robot_state, score_lookahead_interval, "J");
      if (jacobi_score < best_score)
        best_score = jacobi_score;
      else
        using_jacobi = false;
    }
    if (using_jacobi) {
      delta_theta_ = delta_theta_jacobi;
      solution_name = "JACOBI";
    }
  }
  if (solution_name != last_used_solution_source_) {
    last_used_solution_source_ = solution_name;
    RCLCPP_INFO(LOGGER, "USING %s", solution_name.c_str());
  }

  double drift_factor = velocityScalingFactorForDriftDimensions(delta_theta_, jacobian_full, drift_dimensions_,
                                                                servo_params_.drift_speed_correction_drifting_dimension_multipliers,
                                                                servo_params_.drift_speed_correction_nondrifting_dimension_multipliers,
                                                                using_jacobi
                                                                  ? servo_params_.drift_speed_correction_power
                                                                  : servo_params_.drift_speed_correction_power_ik);
  delta_theta_ *= drift_factor;

  // if (delta_x.any())
  //   RCLCPP_INFO_STREAM(LOGGER, "\ndelta_x\n" << delta_x << "\ndelta_theta\n" << delta_theta_);
  bool success = internalServoUpdate(delta_theta_, joint_trajectory, ServoType::CARTESIAN_SPACE);

  debug_data_.clear();
  for (int i = 0; i < delta_theta_.rows(); i++)
    debug_data_.push_back(delta_theta_[i]);
  debug_data_.push_back(8);
  for (int i = 0; i < delta_x_full.rows(); i++)
    debug_data_.push_back(delta_x_full[i]);
  debug_data_.push_back(8);
  debug_data_.push_back(solution_name.length());
  debug_data_.push_back(drift_factor);
  if (!joint_trajectory.points.empty()) {
    debug_data_.push_back(8);
    for (auto v : joint_trajectory.points[0].velocities)
      debug_data_.push_back(v);
    debug_data_.push_back(8);
    for (auto v : joint_trajectory.points[0].positions)
      debug_data_.push_back(v);
    debug_data_.push_back(8);
  }
  debug_data_.push_back(node_->now().seconds());

  return success;
}

double ServoCalcs::solutionScore(
    const Eigen::VectorXd &delta_theta, const Eigen::VectorXd &desired_delta_x, double delta_x_norm_weighted,
    const Eigen::MatrixXd &jacobian_full, moveit::core::RobotState &robot_state, double lookahead_interval, const std::string &name,
    double *direction_error)
{
  double score_drift = 1 - velocityScalingFactorForDriftDimensions(delta_theta, jacobian_full, drift_dimensions_,
                                                                servo_params_.drift_speed_correction_drifting_dimension_multipliers,
                                                                servo_params_.drift_speed_correction_nondrifting_dimension_multipliers,
                                                                1);
  double limits_scale = velocityLimitsScalingFactor(joint_model_group_, delta_theta/servo_params_.publish_period);
  // if (desired_delta_x.any())
  //   RCLCPP_INFO_STREAM(LOGGER, "delta_theta_orig\n" << delta_theta <<
  //     "\nlimits scale " << limits_scale << "\nlookahead interval " << lookahead_interval);

  // Eigen::VectorXd delta_x = jacobian*limits_scale*Eigen::VectorXd(delta_theta);
  Eigen::VectorXd delta_x(6);
  double direction_error_scale = 1;
  if (direction_error) {
    auto next_position = current_joint_state_.position;
    for (size_t i = 0; i < next_position.size(); i++)
      next_position[i] += delta_theta[i]*limits_scale*direction_error_scale*lookahead_interval;
    robot_state.setJointGroupPositions(joint_model_group_, next_position);
    auto next_tip_frame = robot_state.getGlobalLinkTransform(ik_solver_->getBaseFrame()).inverse() *
                          robot_state.getGlobalLinkTransform(ee_frame_id_);
    auto xyz = next_tip_frame.translation() - ik_base_to_tip_frame_.translation();
    auto rodr = Eigen::AngleAxisd(ik_base_to_tip_frame_.rotation().inverse()*next_tip_frame.rotation());
    Eigen::Vector3d axis = ik_base_to_tip_frame_.rotation()*rodr.axis();
    double angle = rodr.angle();

    delta_x << xyz[0], xyz[1], xyz[2], axis[0]*angle, axis[1]*angle, axis[2]*angle;
    delta_x /= lookahead_interval;
    
    double delta_x_proj_multiplier = 0;
    for (size_t i = 0, j = 0; i < drift_dimensions_.size(); i++)
      if (!drift_dimensions_[i]) {
        delta_x_proj_multiplier += desired_delta_x[j]*delta_x[i];
        j++;
      }
    delta_x_proj_multiplier = std::abs(delta_x_proj_multiplier)/desired_delta_x.squaredNorm();

    double dir_error = 0;
    for (size_t i = 0, j = 0; i < drift_dimensions_.size(); i++)
      if (!drift_dimensions_[i]) {
        double error = (delta_x[i] - desired_delta_x[j]*delta_x_proj_multiplier)*servo_params_.drift_speed_correction_nondrifting_dimension_multipliers[i];
        dir_error += error*error;
        j++;
      }
    dir_error = sqrt(dir_error)/(delta_x_norm_weighted*delta_x_proj_multiplier);
    *direction_error = dir_error;
//    RCLCPP_INFO_STREAM(LOGGER, "\ndelta_x\n" << delta_x << "\ndesired_delta_x\n" << desired_delta_x << "\nprojected\n" << desired_delta_x*delta_x_proj_multiplier <<
//      "\nproj: " << delta_x_proj_multiplier << "\n");
//    RCLCPP_INFO(LOGGER, "%s dir error: %lf", name.c_str(), dir_error);
    direction_error_scale = pow(1 + dir_error, -servo_params_.ik_direction_error_slowdown_factor);
  }

  double score_nondrift = 0;
  {
    auto next_position = current_joint_state_.position;
    for (size_t i = 0; i < next_position.size(); i++)
      next_position[i] += delta_theta[i]*limits_scale*direction_error_scale*lookahead_interval;
    robot_state.setJointGroupPositions(joint_model_group_, next_position);
    auto next_tip_frame = robot_state.getGlobalLinkTransform(ik_solver_->getBaseFrame()).inverse() *
                          robot_state.getGlobalLinkTransform(ee_frame_id_);
    auto xyz = next_tip_frame.translation() - ik_base_to_tip_frame_.translation();
    auto rodr = Eigen::AngleAxisd(ik_base_to_tip_frame_.rotation().inverse()*next_tip_frame.rotation());
    Eigen::Vector3d axis = ik_base_to_tip_frame_.rotation()*rodr.axis();
    double angle = rodr.angle();

    delta_x << xyz[0], xyz[1], xyz[2], axis[0]*angle, axis[1]*angle, axis[2]*angle;
    delta_x /= lookahead_interval;
    // RCLCPP_INFO_STREAM(LOGGER,
    //                    name << "\nnext_tip_frame\n" << next_tip_frame.matrix() << "\ncurrent_tip_frame\n" << ik_base_to_tip_frame_.matrix() << "\ndelta_x\n" << delta_x);
  }
  for (size_t i = 0, j = 0; i < drift_dimensions_.size(); i++)
    if (!drift_dimensions_[i]) {
//      RCLCPP_INFO(LOGGER, "i %d, j %d, dxj size %d, dx size %d", (int)i, (int)j, (int)delta_x.rows(), (int)desired_delta_x.rows());
      // double error = (delta_x[j] - desired_delta_x[j])*servo_params_.drift_speed_correction_nondrifting_dimension_multipliers[i];
      double error = (delta_x[i] - desired_delta_x[j])*servo_params_.drift_speed_correction_nondrifting_dimension_multipliers[i];
//      RCLCPP_INFO(LOGGER, "%s error: %lf", name.c_str(), error);
      score_nondrift += error*error;
      j++;
    }
  score_nondrift = sqrt(score_nondrift)/delta_x_norm_weighted;
  double score_theta = limits_scale*direction_error_scale*Eigen::VectorXd(delta_theta).norm();
  double score =
    score_drift*servo_params_.ik_vs_jacobi_drifting_error_weight + 
    score_nondrift*servo_params_.ik_vs_jacobi_nondrifting_error_weight +
    score_theta*servo_params_.ik_vs_jacobi_theta_weight;

  // if (desired_delta_x.any()) {
  //   RCLCPP_INFO_STREAM(LOGGER,
  //                      name << "_delta_theta\n" << delta_theta*limits_scale << "\n" << name << "_delta_x\n" << delta_x
  //                           << "\ndesired_delta_x\n" << desired_delta_x);
  //   RCLCPP_INFO(LOGGER, "\n%s %lf = %lf (%lf) + %lf (%lf) + %lf (%lf)", name.c_str(),
  //               score, score_drift*servo_params_.ik_vs_jacobi_drifting_error_weight, score_drift,
  //               score_nondrift*servo_params_.ik_vs_jacobi_nondrifting_error_weight, score_nondrift,
  //               score_theta*servo_params_.ik_vs_jacobi_theta_weight, score_theta);
  // }
  return score;
}

bool ServoCalcs::jointServoCalcs(const control_msgs::msg::JointJog& cmd,
                                 trajectory_msgs::msg::JointTrajectory& joint_trajectory)
{
  // Check for nan's
  if (!checkValidCommand(cmd))
    return false;

  // Apply user-defined scaling
  delta_theta_ = scaleJointCommand(cmd);

  // Perform internal servo with the command
  return internalServoUpdate(delta_theta_, joint_trajectory, ServoType::JOINT_SPACE);
}

bool ServoCalcs::internalServoUpdate(Eigen::ArrayXd& delta_theta,
                                     trajectory_msgs::msg::JointTrajectory& joint_trajectory,
                                     const ServoType servo_type)
{
  // The order of operations here is:
  // 1. apply velocity scaling for collisions (in the position domain)
  // 2. low-pass filter the position command in applyJointUpdate()
  // 3. calculate velocities in applyJointUpdate()
  // 4. apply velocity limits
  // 5. apply position limits. This is a higher priority than velocity limits, so check it last.

  // Apply collision scaling
  collision_checker_.setWorkspaceBounds(movement_limits_.ee_pos_limits);  // apply tcp position limit
  double collision_scale = collision_checker_.getCollisionVelocityScale(delta_theta);
  if (collision_scale > 0 && collision_scale < 1)
  {
    status_ = StatusCode::DECELERATE_FOR_COLLISION;
    rclcpp::Clock& clock = *node_->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD, SERVO_STATUS_CODE_MAP.at(status_));
  }
  else if (collision_scale == 0)
  {
    status_ = StatusCode::HALT_FOR_COLLISION;
    rclcpp::Clock& clock = *node_->get_clock();
    RCLCPP_ERROR_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD, "Halting for collision!");
  }
  delta_theta *= collision_scale;

  // apply tcp velocity limit
  Eigen::MatrixXd jacobian = current_state_->getJacobian(joint_model_group_);
  double tcp_vel = (jacobian.block(0, 0, jacobian.rows(), 3)*delta_theta.block<3, 1>(0, 0).matrix()).norm()/servo_params_.publish_period;
  if (tcp_vel > movement_limits_.max_ee_velocity) {
    RCLCPP_INFO_THROTTLE(LOGGER, *node_->get_clock(), 2000, "Limiting tcp vel (%lf > %lf)", tcp_vel, movement_limits_.max_ee_velocity);
    delta_theta *= movement_limits_.max_ee_velocity/tcp_vel;
  }

  // apply joint velocity limit
  const auto &joint_model_names = joint_model_group_->getActiveJointModelNames();
  for (int i = 0; i < delta_theta.rows(); i++) {
    const auto &limit_it = movement_limits_.joint_limits.find(joint_model_names[i]);
    if (limit_it != movement_limits_.joint_limits.end()) {
      double v = std::abs(delta_theta[i]/servo_params_.publish_period);
      const auto &limit = limit_it->second.max_velocity;
      if (v > limit) {
        RCLCPP_INFO_THROTTLE(LOGGER, *node_->get_clock(), 2000, "Limiting joint %d %s vel (%lf > %lf)", i, joint_model_names[i].c_str(), v, limit);
        delta_theta *= limit/v;
      }
    }
  }

  // apply joint acceleration limit
  for (int i = 0; i < delta_theta.rows(); i++) {
    const auto &limit_it = movement_limits_.joint_limits.find(joint_model_names[i]);
    if (limit_it != movement_limits_.joint_limits.end()) {
      double next_vel = delta_theta[i]/servo_params_.publish_period;
      const auto &limit = limit_it->second.max_acceleration;
      double current_vel = current_joint_state_.velocity[i];
      double acc = std::abs(next_vel - current_vel)/servo_params_.publish_period;
      if (acc > limit) {
        RCLCPP_INFO_THROTTLE(LOGGER, *node_->get_clock(), 2000, "Limiting joint %d acc (%lf %lf -> %lf > %lf) -> %lf", i, next_vel, current_vel, acc, limit,
                    current_vel + limit*servo_params_.publish_period*(next_vel < current_vel ? -1.0 : 1.0));
        double v = current_vel + limit*servo_params_.publish_period*(next_vel < current_vel ? -1.0 : 1.0);
        Eigen::Map<Eigen::ArrayXd> current_vels(current_joint_state_.velocity.data(), current_joint_state_.velocity.size());

        // vels = current_vels + k*(next_vels - current_vels)
        double k = (v - current_vel)/(next_vel - current_vel);
        delta_theta = (current_vels + k*(delta_theta/servo_params_.publish_period - current_vels))*servo_params_.publish_period;
      }
    }
  }

  // Loop through joints and update them, calculate velocities, and filter
  if (!applyJointUpdate(servo_params_.publish_period, delta_theta, current_joint_state_, next_joint_state_, smoother_))
  {
    RCLCPP_ERROR_STREAM_THROTTLE(LOGGER, *node_->get_clock(), ROS_LOG_THROTTLE_PERIOD,
                                 "Lengths of output and increments do not match.");
    return false;
  }

  // Mark the lowpass filters as updated for this cycle
  updated_filters_ = true;

  // Enforce SRDF velocity limits
  enforceVelocityLimits(joint_model_group_, servo_params_.publish_period, next_joint_state_,
                        servo_params_.override_velocity_scaling_factor);

  // Enforce SRDF position limits, might halt if needed, set prev_vel to 0
  const auto joints_to_halt =
      enforcePositionLimits(next_joint_state_, servo_params_.joint_limit_margin, joint_model_group_);

  if (!joints_to_halt.empty())
  {
    std::ostringstream joint_names;
    std::transform(joints_to_halt.cbegin(), joints_to_halt.cend(), std::ostream_iterator<std::string>(joint_names, ""),
                   [](const auto& joint) { return " '" + joint->getName() + "'"; });

    RCLCPP_WARN_STREAM_THROTTLE(LOGGER, *node_->get_clock(), ROS_LOG_THROTTLE_PERIOD,
                                "Joints" << joint_names.str() << " close to a position limit. Halting.");

    status_ = StatusCode::JOINT_BOUND;
    if ((servo_type == ServoType::JOINT_SPACE && !servo_params_.halt_all_joints_in_joint_mode) ||
        (servo_type == ServoType::CARTESIAN_SPACE && !servo_params_.halt_all_joints_in_cartesian_mode))
    {
      suddenHalt(next_joint_state_, joints_to_halt);
    }
    else
    {
      suddenHalt(next_joint_state_, joint_model_group_->getActiveJointModels());
    }
  }

  // compose outgoing message
  composeJointTrajMessage(next_joint_state_, joint_trajectory);

  // if (delta_theta[1] != 0.0)
  //   RCLCPP_INFO(LOGGER, "dt0: %lf, cs: %lf, ns: %lf", delta_theta[1], current_joint_state_.position[1], next_joint_state_.position[1]);

  previous_joint_state_ = current_joint_state_;
  current_joint_state_ = next_joint_state_;
  return true;
}

void ServoCalcs::resetLowPassFilters(const sensor_msgs::msg::JointState& joint_state)
{
  smoother_->reset(joint_state.position);
  updated_filters_ = true;
}

void ServoCalcs::composeJointTrajMessage(const sensor_msgs::msg::JointState& joint_state,
                                         trajectory_msgs::msg::JointTrajectory& joint_trajectory)
{
  // When a joint_trajectory_controller receives a new command, a stamp of 0 indicates "begin immediately"
  // See http://wiki.ros.org/joint_trajectory_controller#Trajectory_replacement
  joint_trajectory.header.stamp = rclcpp::Time(0);
  joint_trajectory.header.frame_id = servo_params_.planning_frame;
  joint_trajectory.joint_names = joint_state.name;

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.time_from_start = rclcpp::Duration::from_seconds(0.0);  // servo_params_.publish_period);
  if (servo_params_.publish_joint_positions)
    point.positions = joint_state.position;
  if (servo_params_.publish_joint_velocities)
    point.velocities = joint_state.velocity;
  if (servo_params_.publish_joint_accelerations)
  {
    // I do not know of a robot that takes acceleration commands.
    // However, some controllers check that this data is non-empty.
    // Send all zeros, for now.
    std::vector<double> acceleration(num_joints_);
    point.accelerations = acceleration;
  }
  joint_trajectory.points.push_back(point);
}

void ServoCalcs::filteredHalt(trajectory_msgs::msg::JointTrajectory& joint_trajectory)
{
  // Prepare the joint trajectory message to stop the robot
  joint_trajectory.points.clear();
  joint_trajectory.points.emplace_back();
  joint_trajectory.joint_names = joint_model_group_->getActiveJointModelNames();

  // Deceleration algorithm:
  // Set positions to current_joint_state_
  // Filter
  // Calculate velocities
  // Check if velocities are close to zero. Round to zero, if so.
  assert(current_joint_state_.position.size() >= num_joints_);
  joint_trajectory.points[0].positions = current_joint_state_.position;
  // smoother_->doSmoothing(joint_trajectory.points[0].positions);
  bool done_stopping = true;
  if (servo_params_.publish_joint_velocities)
  {
    joint_trajectory.points[0].velocities = std::vector<double>(num_joints_, 0);
    for (std::size_t i = 0; i < num_joints_; ++i)
    {
      joint_trajectory.points[0].velocities.at(i) =
          (joint_trajectory.points[0].positions.at(i) - current_joint_state_.position.at(i)) /
          servo_params_.publish_period;
      // If velocity is very close to zero, round to zero
      if (joint_trajectory.points[0].velocities.at(i) > STOPPED_VELOCITY_EPS)
      {
        done_stopping = false;
      }
    }
    // If every joint is very close to stopped, round velocity to zero
    if (done_stopping)
    {
      std::fill(joint_trajectory.points[0].velocities.begin(), joint_trajectory.points[0].velocities.end(), 0);
    }
  }

  if (servo_params_.publish_joint_accelerations)
  {
    joint_trajectory.points[0].accelerations = std::vector<double>(num_joints_, 0);
    for (std::size_t i = 0; i < num_joints_; ++i)
    {
      joint_trajectory.points[0].accelerations.at(i) =
          (joint_trajectory.points[0].velocities.at(i) - current_joint_state_.velocity.at(i)) /
          servo_params_.publish_period;
    }
  }

  joint_trajectory.points[0].time_from_start = rclcpp::Duration::from_seconds(0.0);  // servo_params_.publish_period);
}

void ServoCalcs::suddenHalt(sensor_msgs::msg::JointState& joint_state,
                            const std::vector<const moveit::core::JointModel*>& joints_to_halt) const
{
  // Set the position to the original position, and velocity to 0 for input joints
  for (const auto& joint_to_halt : joints_to_halt)
  {
    const auto joint_it = std::find(joint_state.name.cbegin(), joint_state.name.cend(), joint_to_halt->getName());
    if (joint_it != joint_state.name.cend())
    {
      const auto joint_index = std::distance(joint_state.name.cbegin(), joint_it);
      joint_state.position.at(joint_index) = current_joint_state_.position.at(joint_index);
      joint_state.velocity.at(joint_index) = 0.0;
    }
  }
}

bool ServoCalcs::checkValidCommand(const control_msgs::msg::JointJog& cmd)
{
  for (double velocity : cmd.velocities)
  {
    if (std::isnan(velocity))
    {
      rclcpp::Clock& clock = *node_->get_clock();
      RCLCPP_WARN_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD,
                                  "nan in incoming command. Skipping this datapoint.");
      return false;
    }
  }
  return true;
}

bool ServoCalcs::checkValidCommand(const geometry_msgs::msg::TwistStamped& cmd)
{
  if (std::isnan(cmd.twist.linear.x) || std::isnan(cmd.twist.linear.y) || std::isnan(cmd.twist.linear.z) ||
      std::isnan(cmd.twist.angular.x) || std::isnan(cmd.twist.angular.y) || std::isnan(cmd.twist.angular.z))
  {
    rclcpp::Clock& clock = *node_->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD,
                                "nan in incoming command. Skipping this datapoint.");
    return false;
  }

  // If incoming commands should be in the range [-1:1], check for |delta|>1
  if (servo_params_.command_in_type == "unitless")
  {
    if ((fabs(cmd.twist.linear.x) > 1) || (fabs(cmd.twist.linear.y) > 1) || (fabs(cmd.twist.linear.z) > 1) ||
        (fabs(cmd.twist.angular.x) > 1) || (fabs(cmd.twist.angular.y) > 1) || (fabs(cmd.twist.angular.z) > 1))
    {
      rclcpp::Clock& clock = *node_->get_clock();
      RCLCPP_WARN_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD,
                                  "Component of incoming command is >1. Skipping this datapoint.");
      return false;
    }
  }

  // Check that the command frame is known
  if (!cmd.header.frame_id.empty() && !current_state_->knowsFrameTransform(cmd.header.frame_id))
  {
    rclcpp::Clock& clock = *node_->get_clock();
    RCLCPP_WARN_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD,
                                "Commanded frame '" << cmd.header.frame_id << "' is unknown, skipping this command");
    return false;
  }

  return true;
}

// Scale the incoming jog command. Returns a vector of position deltas
Eigen::VectorXd ServoCalcs::scaleCartesianCommand(const geometry_msgs::msg::TwistStamped& command)
{
  Eigen::VectorXd result(6);
  result.setZero();  // Or the else case below leads to misery

  // Apply user-defined scaling if inputs are unitless [-1:1]
  if (servo_params_.command_in_type == "unitless")
  {
    result[0] = servo_params_.linear_scale * servo_params_.publish_period * command.twist.linear.x;
    result[1] = servo_params_.linear_scale * servo_params_.publish_period * command.twist.linear.y;
    result[2] = servo_params_.linear_scale * servo_params_.publish_period * command.twist.linear.z;
    result[3] = servo_params_.rotational_scale * servo_params_.publish_period * command.twist.angular.x;
    result[4] = servo_params_.rotational_scale * servo_params_.publish_period * command.twist.angular.y;
    result[5] = servo_params_.rotational_scale * servo_params_.publish_period * command.twist.angular.z;
  }
  // Otherwise, commands are in m/s and rad/s
  else if (servo_params_.command_in_type == "speed_units")
  {
    result[0] = command.twist.linear.x * servo_params_.publish_period;
    result[1] = command.twist.linear.y * servo_params_.publish_period;
    result[2] = command.twist.linear.z * servo_params_.publish_period;
    result[3] = command.twist.angular.x * servo_params_.publish_period;
    result[4] = command.twist.angular.y * servo_params_.publish_period;
    result[5] = command.twist.angular.z * servo_params_.publish_period;
  }
  else
  {
    rclcpp::Clock& clock = *node_->get_clock();
    RCLCPP_ERROR_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD, "Unexpected command_in_type");
  }

  return result;
}

Eigen::VectorXd ServoCalcs::scaleJointCommand(const control_msgs::msg::JointJog& command)
{
  Eigen::VectorXd result(num_joints_);
  result.setZero();

  std::size_t c;
  for (std::size_t m = 0; m < command.joint_names.size(); ++m)
  {
    try
    {
      c = joint_state_name_map_.at(command.joint_names[m]);
    }
    catch (const std::out_of_range& e)
    {
      rclcpp::Clock& clock = *node_->get_clock();
      RCLCPP_DEBUG_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD, "Ignoring joint " << command.joint_names[m]);
      continue;
    }
    // Apply user-defined scaling if inputs are unitless [-1:1]
    if (servo_params_.command_in_type == "unitless")
    {
      result[c] = command.velocities[m] * servo_params_.joint_scale * servo_params_.publish_period;
      // Otherwise, commands are in m/s and rad/s
    }
    else if (servo_params_.command_in_type == "speed_units")
    {
      result[c] = command.velocities[m] * servo_params_.publish_period;
    }
    else
    {
      rclcpp::Clock& clock = *node_->get_clock();
      RCLCPP_ERROR_STREAM_THROTTLE(LOGGER, clock, ROS_LOG_THROTTLE_PERIOD,
                                   "Unexpected command_in_type, check yaml file.");
    }
  }

  return result;
}

void ServoCalcs::removeDimension(Eigen::MatrixXd& jacobian, Eigen::VectorXd& delta_x, unsigned int row_to_remove) const
{
  unsigned int num_rows = jacobian.rows() - 1;
  unsigned int num_cols = jacobian.cols();

  if (row_to_remove < num_rows)
  {
    jacobian.block(row_to_remove, 0, num_rows - row_to_remove, num_cols) =
        jacobian.block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);
    delta_x.segment(row_to_remove, num_rows - row_to_remove) =
        delta_x.segment(row_to_remove + 1, num_rows - row_to_remove);
  }
  jacobian.conservativeResize(num_rows, num_cols);
  delta_x.conservativeResize(num_rows);
}

void ServoCalcs::removeDriftDimensions(Eigen::MatrixXd& matrix, Eigen::VectorXd& delta_x)
{
  // May allow some dimensions to drift, based on drift_dimensions
  // i.e. take advantage of task redundancy.
  // Remove the Jacobian rows corresponding to True in the vector drift_dimensions
  // Work backwards through the 6-vector so indices don't get out of order
  for (auto dimension = matrix.rows() - 1; dimension >= 0; --dimension)
  {
    if (drift_dimensions_[dimension] && matrix.rows() > 1)
    {
      removeDimension(matrix, delta_x, dimension);
    }
  }
}

bool ServoCalcs::getCommandFrameTransform(Eigen::Isometry3d& transform)
{
  const std::lock_guard<std::mutex> lock(main_loop_mutex_);
  transform = tf_moveit_to_robot_cmd_frame_;

  // All zeros means the transform wasn't initialized, so return false
  return !transform.matrix().isZero(0);
}

bool ServoCalcs::getCommandFrameTransform(geometry_msgs::msg::TransformStamped& transform)
{
  const std::lock_guard<std::mutex> lock(main_loop_mutex_);
  // All zeros means the transform wasn't initialized, so return false
  if (tf_moveit_to_robot_cmd_frame_.matrix().isZero(0))
  {
    return false;
  }

  transform = convertIsometryToTransform(tf_moveit_to_robot_cmd_frame_, servo_params_.planning_frame,
                                         servo_params_.robot_link_command_frame);
  return true;
}

bool ServoCalcs::getEEFrameTransform(Eigen::Isometry3d& transform)
{
  const std::lock_guard<std::mutex> lock(main_loop_mutex_);
  transform = tf_moveit_to_ee_frame_;

  // All zeros means the transform wasn't initialized, so return false
  return !transform.matrix().isZero(0);
}

bool ServoCalcs::getEEFrameTransform(geometry_msgs::msg::TransformStamped& transform)
{
  const std::lock_guard<std::mutex> lock(main_loop_mutex_);
  // All zeros means the transform wasn't initialized, so return false
  if (tf_moveit_to_ee_frame_.matrix().isZero(0))
  {
    return false;
  }

  transform =
      convertIsometryToTransform(tf_moveit_to_ee_frame_, servo_params_.planning_frame, servo_params_.ee_frame_name);
  return true;
}

void ServoCalcs::eeFrameIdCB(const std_msgs::msg::String::ConstSharedPtr& msg)
{
  if (ik_solver_) {
    const auto &tip_frames = ik_solver_->getTipFrames();
    if (std::find(tip_frames.begin(), tip_frames.end(), msg->data) == tip_frames.end()) {
      RCLCPP_ERROR_STREAM(LOGGER, "End effector frame " << msg->data << " not found among tip frames. Ignoring. Valid values are: " <<
                                                        (Eigen::Map<const Eigen::Array<std::string, 1, Eigen::Dynamic>>(tip_frames.data(), tip_frames.size())));
      return;
    }
  }
  else if (!joint_model_group_->hasLinkModel(msg->data)) {
    const auto &link_models = joint_model_group_->getLinkModelNames();
    RCLCPP_ERROR_STREAM(LOGGER, "End effector frame " << msg->data << " not found among tip frames. Ignoring. Valid values are: " <<
                                                      (Eigen::Map<const Eigen::Array<std::string, 1, Eigen::Dynamic>>(link_models.data(), link_models.size())));
    return;
  }
  ee_frame_id_ = msg->data;
  RCLCPP_INFO(LOGGER, "Set ee_frame_id to %s", ee_frame_id_.c_str());
}

void ServoCalcs::twistStampedCB(const geometry_msgs::msg::TwistStamped::ConstSharedPtr& msg)
{
  const std::lock_guard<std::mutex> lock(main_loop_mutex_);
  latest_twist_stamped_ = msg;

  if (msg->header.stamp != rclcpp::Time(0.))
    latest_twist_command_stamp_ = msg->header.stamp;
  // RCLCPP_INFO(LOGGER, "received twist %lf", rclcpp::Time(msg->header.stamp).seconds());

  // notify that we have a new input
  new_input_cmd_ = true;
  input_cv_.notify_all();
}

void ServoCalcs::jointCmdCB(const control_msgs::msg::JointJog::ConstSharedPtr& msg)
{
  const std::lock_guard<std::mutex> lock(main_loop_mutex_);
  latest_joint_cmd_ = msg;

  if (msg->header.stamp != rclcpp::Time(0.))
    latest_joint_command_stamp_ = msg->header.stamp;

  // notify that we have a new input
  new_input_cmd_ = true;
  input_cv_.notify_all();
}

void ServoCalcs::collisionVelocityScaleCB(const std_msgs::msg::Float64::ConstSharedPtr& msg)
{
  collision_velocity_scale_ = msg->data;
}

void ServoCalcs::changeDriftDimensions(const std::shared_ptr<spinbot_msgs::srv::ChangeDriftDimensions::Request>& req,
                                       const std::shared_ptr<spinbot_msgs::srv::ChangeDriftDimensions::Response>& res)
{
  std::unique_lock<std::mutex> main_loop_lock(main_loop_mutex_);
  drift_dimensions_[0] = req->drift_x_translation;
  drift_dimensions_[1] = req->drift_y_translation;
  drift_dimensions_[2] = req->drift_z_translation;
  drift_dimensions_[3] = req->drift_x_rotation;
  drift_dimensions_[4] = req->drift_y_rotation;
  drift_dimensions_[5] = req->drift_z_rotation;

  res->success = true;
}

}  // namespace moveit_servo
