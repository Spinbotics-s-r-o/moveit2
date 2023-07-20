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

/*      Title     : collision_check.cpp
 *      Project   : moveit_servo
 *      Created   : 1/11/2019
 *      Author    : Brian O'Neil, Andy Zelenak, Blake Anderson
 */

#include <std_msgs/msg/float64.hpp>

#include <moveit_servo/collision_check.h>
// #include <moveit_servo/make_shared_from_pool.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_servo.collision_check");
static const double MIN_RECOMMENDED_COLLISION_RATE = 10;
constexpr size_t ROS_LOG_THROTTLE_PERIOD = 30 * 1000;  // Milliseconds to throttle logs inside loops

namespace moveit_servo
{
// Constructor for the class that handles collision checking
CollisionCheck::CollisionCheck(const rclcpp::Node::SharedPtr& node,
                               const planning_scene_monitor::PlanningSceneMonitorPtr& planning_scene_monitor,
                               const std::shared_ptr<const servo::ParamListener>& servo_param_listener)
  : node_(node)
  , servo_param_listener_(servo_param_listener)
  , servo_params_(std::make_shared<servo::Params>(servo_param_listener_->get_params()))
  , planning_scene_monitor_(planning_scene_monitor)
  , self_velocity_scale_coefficient_(-log(0.001) / servo_params_->self_collision_proximity_threshold)
  , scene_velocity_scale_coefficient_(-log(0.001) / servo_params_->scene_collision_proximity_threshold)
  , workspace_bounds_(Eigen::Vector3d(-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()),
                      Eigen::Vector3d(std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()))
{
  // Init collision request
  collision_request_.group_name = servo_params_->move_group_name;
  collision_request_.distance = true;  // enable distance-based collision checking
  collision_request_.contacts = false;  // Record the names of collision pairs

  if (servo_params_->collision_check_rate < MIN_RECOMMENDED_COLLISION_RATE)
  {
    auto& clk = *node_->get_clock();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    RCLCPP_WARN_STREAM_THROTTLE(LOGGER, clk, ROS_LOG_THROTTLE_PERIOD,
                                "Collision check rate is low, increase it in yaml file if CPU allows");
#pragma GCC diagnostic pop
  }

  // ROS pubs/subs
  collision_velocity_scale_pub_ =
      node_->create_publisher<std_msgs::msg::Float64>("~/collision_velocity_scale", rclcpp::SystemDefaultsQoS());
}

planning_scene_monitor::LockedPlanningSceneRO CollisionCheck::getLockedPlanningSceneRO() const
{
  return planning_scene_monitor::LockedPlanningSceneRO(planning_scene_monitor_);
}

double CollisionCheck::getCollisionVelocityScale(const Eigen::ArrayXd& delta_theta) const
{
  // Update the latest parameters
  if (servo_params_->enable_parameter_update && servo_param_listener_->is_old(*servo_params_))
  {
    *servo_params_ = servo_param_listener_->get_params();
  }
  const auto &servo_params = *servo_params_;

  if (!servo_params.check_collisions)
    return 1.0;
  // Update to the latest current state
  auto current_state = planning_scene_monitor_->getStateMonitor()->getCurrentState();
  current_state->updateCollisionBodyTransforms();
  bool collision_detected = false;

  // Do a timer-safe distance-based collision detection
  collision_detection::CollisionResult collision_result;
  collision_result.clear();
  getLockedPlanningSceneRO()->getCollisionEnv()->checkRobotCollision(collision_request_, collision_result,
                                                                     *current_state);
  double scene_collision_distance = collision_result.distance;
  collision_detected |= collision_result.collision;
  // collision_result.print();

  auto group = current_state->getJointModelGroup(servo_params_->move_group_name);
  Eigen::Isometry3d ee_trans = current_state->getGlobalLinkTransform(group->getLinkModelNames().back());
  double col_prox_threshold = servo_params.scene_collision_proximity_threshold;
  Eigen::Vector3d inset(col_prox_threshold, col_prox_threshold, col_prox_threshold);
  Eigen::AlignedBox3d inset_bounds(workspace_bounds_.min() + inset, workspace_bounds_.max() - inset);
  double ws_bounds_distance = -workspace_bounds_.exteriorDistance(ee_trans.translation()) + col_prox_threshold;
  scene_collision_distance = std::min(scene_collision_distance, ws_bounds_distance);

  collision_result.clear();
  // Self-collisions and scene collisions are checked separately so different thresholds can be used
  getLockedPlanningSceneRO()->getCollisionEnvUnpadded()->checkSelfCollision(
      collision_request_, collision_result, *current_state, getLockedPlanningSceneRO()->getAllowedCollisionMatrix());
  double self_collision_distance = collision_result.distance;
  collision_detected |= collision_result.collision;
  // collision_result.print();

  double velocity_scale = 1;
  // If we're definitely in collision, stop immediately
  if (collision_detected)
  {
    velocity_scale = 0;
  }
  else
  {
    // If we are far from a collision, velocity_scale should be 1.
    // If we are very close to a collision, velocity_scale should be ~zero.
    // When scene_collision_proximity_threshold is breached, start decelerating exponentially.
    if (scene_collision_distance < servo_params.scene_collision_proximity_threshold)
    {
      // velocity_scale = e ^ k * (collision_distance - threshold)
      // k = - ln(0.001) / collision_proximity_threshold
      // velocity_scale should equal one when collision_distance is at collision_proximity_threshold.
      // velocity_scale should equal 0.001 when collision_distance is at zero.
      velocity_scale = std::min(velocity_scale,
                                exp(scene_velocity_scale_coefficient_ *
                                     (scene_collision_distance - servo_params.scene_collision_proximity_threshold)));
    }

    if (self_collision_distance < servo_params.self_collision_proximity_threshold)
    {
      velocity_scale =
          std::min(velocity_scale, exp(self_velocity_scale_coefficient_ *
                                       (self_collision_distance - servo_params.self_collision_proximity_threshold)));
    }

    if (velocity_scale != 1) {
      // Update to the latest current state
      Eigen::VectorXd positions;
      current_state->copyJointGroupPositions(servo_params.move_group_name, positions);
      current_state->setJointGroupPositions(servo_params.move_group_name, positions + delta_theta.matrix());
      current_state->updateCollisionBodyTransforms();
      collision_detected = false;

      // Do a timer-safe distance-based collision detection
      collision_result.clear();
      getLockedPlanningSceneRO()->getCollisionEnv()->checkRobotCollision(collision_request_, collision_result,
                                                                         *current_state);
      scene_collision_distance = collision_result.distance;
      collision_detected |= collision_result.collision;
      // collision_result.print();
      auto group = current_state->getJointModelGroup(servo_params_->move_group_name);
      Eigen::Isometry3d ee_trans = current_state->getGlobalLinkTransform(group->getLinkModelNames().back());
      double col_prox_threshold = servo_params.scene_collision_proximity_threshold;
      Eigen::Vector3d inset(col_prox_threshold, col_prox_threshold, col_prox_threshold);
      Eigen::AlignedBox3d inset_bounds(workspace_bounds_.min() + inset, workspace_bounds_.max() - inset);
      double ws_bounds_distance = -workspace_bounds_.exteriorDistance(ee_trans.translation()) + col_prox_threshold;
      scene_collision_distance = std::min(scene_collision_distance, ws_bounds_distance);

      collision_result.clear();
      // Self-collisions and scene collisions are checked separately so different thresholds can be used
      getLockedPlanningSceneRO()->getCollisionEnvUnpadded()->checkSelfCollision(
          collision_request_, collision_result, *current_state, getLockedPlanningSceneRO()->getAllowedCollisionMatrix());
      self_collision_distance = collision_result.distance;
      collision_detected |= collision_result.collision;
      // collision_result.print();

      double future_velocity_scale = 1;
      // If we're definitely in collision, stop immediately
      if (collision_detected)
        future_velocity_scale = 0;
      else {
        // If we are far from a collision, future_velocity_scale should be 1.
        // If we are very close to a collision, future_velocity_scale should be ~zero.
        // When scene_collision_proximity_threshold is breached, start decelerating exponentially.
        if (scene_collision_distance < servo_params.scene_collision_proximity_threshold) {
          // future_velocity_scale = e ^ k * (collision_distance - threshold)
          // k = - ln(0.001) / collision_proximity_threshold
          // future_velocity_scale should equal one when collision_distance is at collision_proximity_threshold.
          // future_velocity_scale should equal 0.001 when collision_distance is at zero.
          future_velocity_scale = std::min(future_velocity_scale,
                                           exp(scene_velocity_scale_coefficient_ *
                                         (scene_collision_distance -
                                          servo_params.scene_collision_proximity_threshold)));
        }

        if (self_collision_distance < servo_params.self_collision_proximity_threshold) {
          future_velocity_scale =
              std::min(future_velocity_scale, exp(self_velocity_scale_coefficient_ *
                                                  (self_collision_distance -
                                            servo_params.self_collision_proximity_threshold)));
        }
      }
      if (future_velocity_scale > velocity_scale)
        velocity_scale = std::min(1.0, velocity_scale + servo_params.leaving_collision_velocity_boost);
    }
  }

  // Publish collision velocity scaling message.
  {
    auto msg = std::make_unique<std_msgs::msg::Float64>();
    msg->data = velocity_scale;
    collision_velocity_scale_pub_->publish(std::move(msg));
  }
  return velocity_scale;
}

void CollisionCheck::setWorkspaceBounds(Eigen::AlignedBox3d workspace_bounds)
{
  workspace_bounds_ = workspace_bounds;
}

}  // namespace moveit_servo
