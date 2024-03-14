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
  , self_velocity_scale_coefficient_(servo_params_->collision_velocity_penalty_coefficient)  //-log(0.001) / servo_params_->self_collision_proximity_threshold)
  , scene_velocity_scale_coefficient_(servo_params_->collision_velocity_penalty_coefficient)  //-log(0.001) / servo_params_->scene_collision_proximity_threshold)
  , workspace_bounds_(Eigen::Vector3d(-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()),
                      Eigen::Vector3d(std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()))
{
  // Init collision request
  distance_request_.type = collision_detection::DistanceRequestType::SINGLE;
  distance_request_.group_name = servo_params_->move_group_name;
  distance_request_.enable_nearest_points = true;
  distance_request_.enable_signed_distance = true;
  distance_request_.compute_gradient = true;
  distance_request_.max_contacts_per_body = 100;

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

  // Do a timer-safe distance-based collision detection
  collision_detection::CollisionResult collision_result;
  collision_result.clear();
  planning_scene_monitor::LockedPlanningSceneRO scene(planning_scene_monitor_);
  auto current_state = scene->getCurrentState();
  current_state.updateCollisionBodyTransforms();
  auto distance_request = distance_request_;
  distance_request.acm = &scene->getAllowedCollisionMatrix();
  distance_request.enableGroup(scene->getRobotModel());
  collision_detection::DistanceResult distance_result;
  distance_result.clear();

  scene->getCollisionEnvUnpadded()->distanceRobot(distance_request, distance_result, current_state);
  double scene_collision_distance = distance_result.minimum_distance.distance;
  auto distances = distance_result.distances;

  auto group = current_state.getJointModelGroup(servo_params_->move_group_name);
  Eigen::Isometry3d ee_trans = current_state.getGlobalLinkTransform(group->getLinkModelNames().back());
  double col_prox_threshold = servo_params.scene_collision_proximity_threshold;
  Eigen::Vector3d inset(col_prox_threshold, col_prox_threshold, col_prox_threshold);
  Eigen::AlignedBox3d inset_bounds(workspace_bounds_.min() + inset, workspace_bounds_.max() - inset);
  double ws_bounds_distance = -workspace_bounds_.exteriorDistance(ee_trans.translation()) + col_prox_threshold;
  scene_collision_distance = std::min(scene_collision_distance, ws_bounds_distance);

  distance_result.clear();
  // Self-collisions and scene collisions are checked separately so different thresholds can be used
  scene->getCollisionEnvUnpadded()->distanceSelf(distance_request, distance_result, current_state);
  double self_collision_distance = distance_result.minimum_distance.distance;
  for (auto &dist : distance_result.distances)
    distances[dist.first].insert(distances[dist.first].end(), dist.second.begin(), dist.second.end());

  double velocity_scale = 1;

  if (scene_collision_distance < servo_params.scene_collision_proximity_threshold || self_collision_distance < servo_params.self_collision_proximity_threshold) {
    const moveit::core::JointModel* root_joint_model = group->getJointModels()[0];  // group->getJointRoots()[0];
    const moveit::core::LinkModel* root_link_model = root_joint_model->getParentLinkModel();
    // getGlobalLinkTransform() returns a valid isometry by contract
    Eigen::Isometry3d reference_transform =
        root_link_model ? current_state.getGlobalLinkTransform(root_link_model) : Eigen::Isometry3d::Identity();
    // std::ostringstream ss;  // for debug logging
    // ss << "\ndelta_theta: " << delta_theta.transpose();

    planning_scene::PlanningScenePtr next_scene;
    collision_detection::DistanceMap next_distances;
    auto robot_model = scene->getRobotModel();
    Eigen::MatrixXd jacobian;

    for (auto &dist : distances) {
      for (auto &d : dist.second) {
        // getLinkModel produces warning if not checked using hasLinkModel
        auto first_link = robot_model->hasLinkModel(d.link_names[0]) ? robot_model->getLinkModel(d.link_names[0]) : nullptr;
        auto second_link = robot_model->hasLinkModel(d.link_names[1]) ? robot_model->getLinkModel(d.link_names[1]) : nullptr;

        bool is_self_collision = first_link && second_link;
        double proximity_threshold = is_self_collision ? servo_params.self_collision_proximity_threshold : servo_params.scene_collision_proximity_threshold;

        if (d.distance >= proximity_threshold)
          continue;
        // RCLCPP_INFO(LOGGER, "Proximity detected between %s and %s at distance %f (%s %lx %s %lx)", d.link_names[0].c_str(), d.link_names[1].c_str(), d.distance,
        //             d.link_names[0].c_str(), (long)first_link, d.link_names[1].c_str(), (long)second_link);
        double velocity_scale_coefficient = is_self_collision ? self_velocity_scale_coefficient_ : scene_velocity_scale_coefficient_;

        Eigen::Vector3d first_dir = Eigen::Vector3d::Zero();
        if (first_link) {
          Eigen::Isometry3d link_transform = current_state.getGlobalLinkTransform(first_link);
          current_state.getJacobian(group, first_link, link_transform.inverse()*d.nearest_points[0], jacobian, false);
          first_dir = reference_transform.linear()*(jacobian*delta_theta.matrix()).block<3, 1>(0, 0);
          // ss << "\nfcp " << d.link_names[0] << ": " << (link_transform.inverse()*d.nearest_points[0]).transpose()
          //    << "\nglobal:" << d.nearest_points[0].transpose() << "; dir: " << first_dir.transpose() << " (" << (jacobian*delta_theta.matrix()).block<3, 1>(0, 0).transpose() << ")";
        }
        Eigen::Vector3d second_dir = Eigen::Vector3d::Zero();
        if (second_link) {
          Eigen::Isometry3d link_transform = current_state.getGlobalLinkTransform(second_link);
          current_state.getJacobian(group, second_link, link_transform.inverse()*d.nearest_points[1], jacobian, false);
          second_dir = reference_transform.linear()*(jacobian*delta_theta.matrix()).block<3, 1>(0, 0);
          // ss << "\nscp: " << d.link_names[1] << ": " << (link_transform.inverse()*d.nearest_points[1]).transpose()
          //    << "\nglobal:" << d.nearest_points[1].transpose() << "; dir: " << second_dir.transpose() << " (" << (jacobian*delta_theta.matrix()).block<3, 1>(0, 0).transpose() << ")";
        }
        Eigen::Vector3d approach_dir = first_dir - second_dir;
        double approach_ratio = 0.0;
        const double epsilon = std::numeric_limits<double>::epsilon();
        if (approach_dir.squaredNorm() > 0.0) {
          if (d.normal.squaredNorm() <= epsilon) {
            if (!next_scene) {
              auto next_scene = scene->diff();
              auto &next_state = next_scene->getCurrentStateNonConst();
              Eigen::VectorXd positions;
              next_state.copyJointGroupPositions(servo_params.move_group_name, positions);
              next_state.setJointGroupPositions(servo_params.move_group_name, positions + delta_theta.matrix());
              next_state.updateCollisionBodyTransforms();
              distance_result.clear();
              next_scene->getCollisionEnvUnpadded()->distanceRobot(distance_request, distance_result, next_state);
              next_distances = distance_result.distances;
              distance_result.clear();
              next_scene->getCollisionEnvUnpadded()->distanceSelf(distance_request, distance_result, next_state);
              for (auto &dist : distance_result.distances)
                next_distances[dist.first].insert(next_distances[dist.first].end(), dist.second.begin(), dist.second.end());
            }
            auto &ndist = next_distances[dist.first];
            if (ndist.empty()) {
              d.normal = -approach_dir.normalized();
              d.distance = std::abs(d.distance);
            }
            else {
              auto nd = std::min_element(ndist.begin(), ndist.end(),
                                         [&d](const auto &a, const auto &b) {
                return (a.nearest_points[0] - d.nearest_points[0]).squaredNorm() < (b.nearest_points[0] - d.nearest_points[0]).squaredNorm();
              });

              // Eigen::VectorXd positions;
              // current_state.copyJointGroupPositions(servo_params.move_group_name, positions);
              // ss << "\nCurr positions: " << positions.transpose();
              // next_scene->getCurrentState().copyJointGroupPositions(servo_params.move_group_name, positions);
              // ss << "\nNext positions: " << positions.transpose();
              // ss << "\nUsing next (previous d had normal " << d.normal.transpose() << ", dist " << d.distance << ")";
              d.normal = nd->normal;
              d.distance = nd->distance;
            }
          }
          approach_ratio = d.normal.squaredNorm() > epsilon ? d.normal.dot(approach_dir)/approach_dir.norm() : 1.0;
        }

        double vel_scale = pow(std::max(0.0, d.distance)/proximity_threshold, velocity_scale_coefficient); // exp(velocity_scale_coefficient * (d.distance - proximity_threshold));
        double leave_boost = servo_params.leaving_collision_velocity_boost + std::max(0.0, -approach_ratio)*(1.0 - servo_params.leaving_collision_velocity_boost);
        vel_scale = std::min(vel_scale < leave_boost ? leave_boost : vel_scale,
                             approach_ratio <= epsilon ? 1.0 : vel_scale/std::max(epsilon, approach_ratio));
        // ss << "\nFirst " << d.link_names[0] << ": " << first_dir.transpose() <<
        //       "\nSecond " << d.link_names[1] << ": " << second_dir.transpose();
        // ss << "\nNormal " << d.normal.transpose() << "; dist " << d.distance;
        // ss << "\nApproach dir: " << approach_dir.transpose() << "\nApproach ratio: " << approach_ratio << "\nVelocity scale: " << vel_scale << "\n";
        velocity_scale = std::min(velocity_scale, vel_scale);
      }
    }
    // if (/*velocity_scale >= 0.001 && */delta_theta.matrix().norm() >= 0.001)
    //   RCLCPP_INFO_STREAM(LOGGER, ss.str());
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
