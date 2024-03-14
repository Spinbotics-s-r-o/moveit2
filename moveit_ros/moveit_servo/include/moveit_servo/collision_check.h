/*******************************************************************************
 * Title     : collision_check.h
 * Project   : moveit_servo
 * Created   : 1/11/2019
 * Author    : Brian O'Neil, Andy Zelenak, Blake Anderson
 *
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

#pragma once

#include <mutex>

#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
// Auto-generated
#include <moveit_servo_lib_parameters.hpp>

namespace moveit_servo
{
class CollisionCheck
{
public:
  /** \brief Constructor
   *  \param parameters: common settings of moveit_servo
   *  \param planning_scene_monitor: PSM should have scene monitor and state monitor
   *                                 already started when passed into this class
   */
  CollisionCheck(const rclcpp::Node::SharedPtr& node,
                 const planning_scene_monitor::PlanningSceneMonitorPtr& planning_scene_monitor,
                 const std::shared_ptr<const servo::ParamListener>& servo_param_listener);

  ~CollisionCheck()
  {
  }

  double getCollisionVelocityScale(const Eigen::ArrayXd& delta_theta) const;

  void setWorkspaceBounds(Eigen::AlignedBox3d workspace_bounds);

private:
  // Pointer to the ROS node
  const std::shared_ptr<rclcpp::Node> node_;

  // Servo parameters
  const std::shared_ptr<const servo::ParamListener> servo_param_listener_;
  std::shared_ptr<servo::Params> servo_params_;

  // Pointer to the collision environment
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

  const double self_velocity_scale_coefficient_;
  const double scene_velocity_scale_coefficient_;
  Eigen::AlignedBox3d workspace_bounds_;

  // collision request
  collision_detection::DistanceRequest distance_request_;

  // ROS
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr collision_velocity_scale_pub_;
};
}  // namespace moveit_servo
