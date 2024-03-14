// Copyright 2022 PickNik Inc.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the PickNik Inc. nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/* Author    : Andy Zelenak
   Desc      : Free functions. We keep them in a separate translation unit to reduce .o filesize
   Title     : utilities.cpp
   Project   : moveit_servo
*/

#include <moveit_servo/utilities.h>

// Disable -Wold-style-cast because all _THROTTLE macros trigger this
// It would be too noisy to disable on a per-callsite basis
#pragma GCC diagnostic ignored "-Wold-style-cast"

namespace moveit_servo
{

/** \brief Helper function for converting Eigen::Isometry3d to geometry_msgs/TransformStamped **/
geometry_msgs::msg::TransformStamped convertIsometryToTransform(const Eigen::Isometry3d& eigen_tf,
                                                                const std::string& parent_frame,
                                                                const std::string& child_frame)
{
  geometry_msgs::msg::TransformStamped output = tf2::eigenToTransform(eigen_tf);
  output.header.frame_id = parent_frame;
  output.child_frame_id = child_frame;

  return output;
}

double velocityScalingFactorForSingularity(const moveit::core::JointModelGroup* joint_model_group,
                                           const Eigen::VectorXd& commanded_twist,
                                           const Eigen::JacobiSVD<Eigen::MatrixXd>& svd,
                                           const Eigen::MatrixXd& pseudo_inverse,
                                           const double hard_stop_singularity_threshold,
                                           const double lower_singularity_threshold,
                                           const double leaving_singularity_threshold_multiplier,
                                           const moveit::core::RobotStateConstPtr& current_state, StatusCode& status)
{
  double velocity_scale = 1;
  std::size_t num_dimensions = commanded_twist.size();

  // Find the direction away from nearest singularity.
  // The last column of U from the SVD of the Jacobian points directly toward or away from the singularity.
  // The sign can flip at any time, so we have to do some extra checking.
  // Look ahead to see if the Jacobian's condition will decrease.
  Eigen::VectorXd vector_toward_singularity = svd.matrixU().col(svd.matrixU().cols() - 1);

  double ini_condition = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

  // This singular vector tends to flip direction unpredictably. See R. Bro,
  // "Resolving the Sign Ambiguity in the Singular Value Decomposition".
  // Look ahead to see if the Jacobian's condition will decrease in this
  // direction. Start with a scaled version of the singular vector
  Eigen::VectorXd delta_x(num_dimensions);
  double scale = 100;
  delta_x = vector_toward_singularity / scale;

  // Calculate a small change in joints
  Eigen::VectorXd new_theta;
  current_state->copyJointGroupPositions(joint_model_group, new_theta);
  new_theta += pseudo_inverse * delta_x;
  auto robot_state = *current_state;
  robot_state.setJointGroupPositions(joint_model_group, new_theta);
  Eigen::MatrixXd new_jacobian = robot_state.getJacobian(joint_model_group);

  Eigen::JacobiSVD<Eigen::MatrixXd> new_svd(new_jacobian);
  double new_condition = new_svd.singularValues()(0) / new_svd.singularValues()(new_svd.singularValues().size() - 1);
  // If new_condition < ini_condition, the singular vector does point towards a
  // singularity. Otherwise, flip its direction.
  if (ini_condition >= new_condition)
  {
    vector_toward_singularity *= -1;
  }

  // If this dot product is positive, we're moving toward singularity
  double dot = vector_toward_singularity.dot(commanded_twist);
  // see https://github.com/ros-planning/moveit2/pull/620#issuecomment-1201418258 for visual explanation of algorithm
  double upper_threshold = dot > 0 ? hard_stop_singularity_threshold :
                                     (hard_stop_singularity_threshold - lower_singularity_threshold) *
                                             leaving_singularity_threshold_multiplier +
                                         lower_singularity_threshold;
  if ((ini_condition > lower_singularity_threshold) && (ini_condition < upper_threshold))
  {
    velocity_scale =
        1. - (ini_condition - lower_singularity_threshold) / (upper_threshold - lower_singularity_threshold);
    status =
        dot > 0 ? StatusCode::DECELERATE_FOR_APPROACHING_SINGULARITY : StatusCode::DECELERATE_FOR_LEAVING_SINGULARITY;
  }

  // Very close to singularity, so halt.
  else if (ini_condition >= upper_threshold)
  {
    velocity_scale = 0;
    status = StatusCode::HALT_FOR_SINGULARITY;
  }

  return velocity_scale;
}

double velocityScalingFactorForSingularity(const Eigen::VectorXd& delta_theta,
                                           const Eigen::MatrixXd& jacobian,
                                           const Eigen::VectorXd& commanded_twist,
                                           const double hard_stop_singularity_threshold,
                                           const double lower_singularity_threshold,
                                           const double leaving_singularity_threshold_multiplier,
                                           const moveit::core::JointModelGroup* joint_model_group,
                                           const moveit::core::RobotStateConstPtr& current_state, StatusCode& status)
{
  double velocity_scale = 1;
  Eigen::VectorXd delta_x = jacobian * delta_theta;
  double compliance_factor = delta_x.dot(commanded_twist)/std::max(std::numeric_limits<double>::epsilon(), sqrt(delta_x.squaredNorm()*commanded_twist.squaredNorm()));
  delta_x *= std::max(0.0, compliance_factor);
  double ini_condition = sqrt(std::max(std::numeric_limits<double>::epsilon(), delta_theta.squaredNorm())/delta_x.squaredNorm());
  if (commanded_twist.norm() > 0.0003)
    RCLCPP_INFO_STREAM(rclcpp::get_logger("servo_utils"), "ini_condition: " << ini_condition << " compliance_factor: " << compliance_factor <<
                       "\ndelta_x: " << delta_x.transpose() << "\ncommanded_twist: " << commanded_twist.transpose() << "\ndelta_theta: " << delta_theta.transpose() << "\n");

  Eigen::VectorXd new_theta;
  current_state->copyJointGroupPositions(joint_model_group, new_theta);
  new_theta += delta_theta;
  auto robot_state = *current_state;
  robot_state.setJointGroupPositions(joint_model_group, new_theta);
  Eigen::MatrixXd new_jacobian = robot_state.getJacobian(joint_model_group);
  Eigen::VectorXd new_delta_x = new_jacobian * delta_theta;
  double new_compliance_factor = new_delta_x.dot(commanded_twist)/std::max(std::numeric_limits<double>::epsilon(), sqrt(new_delta_x.squaredNorm()*commanded_twist.squaredNorm()));
  new_delta_x *= std::max(0.0, new_compliance_factor);
  double new_condition = sqrt(std::max(std::numeric_limits<double>::epsilon(), delta_theta.squaredNorm())/new_delta_x.squaredNorm());
  bool approaching_singularity = new_condition >= ini_condition;
  if (commanded_twist.norm() > 0.0003)
    RCLCPP_INFO_STREAM(rclcpp::get_logger("servo_utils"), "new_condition: " << new_condition << " new_compliance_factor: " << new_compliance_factor <<
                       "\nnew_delta_x: " << new_delta_x.transpose());

  // see https://github.com/ros-planning/moveit2/pull/620#issuecomment-1201418258 for visual explanation of algorithm
  double upper_threshold = approaching_singularity ? hard_stop_singularity_threshold :
                           (hard_stop_singularity_threshold - lower_singularity_threshold) *
                           leaving_singularity_threshold_multiplier +
                           lower_singularity_threshold;
  if ((ini_condition > lower_singularity_threshold) && (ini_condition < upper_threshold))
  {
    velocity_scale =
        1. - (ini_condition - lower_singularity_threshold) / (upper_threshold - lower_singularity_threshold);
    status =
        approaching_singularity ? StatusCode::DECELERATE_FOR_APPROACHING_SINGULARITY : StatusCode::DECELERATE_FOR_LEAVING_SINGULARITY;
  }
  else if (ini_condition >= upper_threshold)
  {
    velocity_scale = 0;
    status = StatusCode::HALT_FOR_SINGULARITY;
  }
  return velocity_scale;
}

bool applyJointUpdate(const double publish_period, const Eigen::ArrayXd& delta_theta,
                      const sensor_msgs::msg::JointState& previous_joint_state,
                      sensor_msgs::msg::JointState& next_joint_state,
                      pluginlib::UniquePtr<online_signal_smoothing::SmoothingBaseClass>& smoother)
{
  // All the sizes must match
  if (next_joint_state.position.size() != static_cast<std::size_t>(delta_theta.size()) ||
      next_joint_state.velocity.size() != next_joint_state.position.size())
  {
    return false;
  }

  for (std::size_t i = 0; i < next_joint_state.position.size(); ++i)
  {
    // Increment joint
    next_joint_state.position[i] += delta_theta[i];
  }

//  smoother->doSmoothing(next_joint_state.position);  // TODO: create a void smoother and uncomment

  // Lambda that calculates velocity using central difference.
  // (q(t + dt) - q(t - dt)) / ( 2 * dt )
  auto compute_velocity = [&](const double next_pos, const double previous_pos) {
//    return (next_pos - previous_pos) / (2 * publish_period);
    return (next_pos - previous_pos) / publish_period;  // we're using a simpl formula: (q(t + dt) - q(t)) / ( dt )
  };

  // Transform that applies the lambda to all joints.
  // next_joint_state contains the future position q(t + dt)
  // previous_joint_state_ contains past position q(t - dt)
  std::transform(next_joint_state.position.begin(), next_joint_state.position.end(),
                 previous_joint_state.position.begin(), next_joint_state.velocity.begin(), compute_velocity);

  return true;
}

void transformTwistToPlanningFrame(geometry_msgs::msg::TwistStamped& cmd, const std::string& planning_frame,
                                   const moveit::core::RobotStatePtr& current_state)
{
  // We solve (planning_frame -> base -> cmd.header.frame_id)
  // by computing (base->planning_frame)^-1 * (base->cmd.header.frame_id)
  const Eigen::Isometry3d tf_moveit_to_incoming_cmd_frame =
      current_state->getGlobalLinkTransform(planning_frame).inverse() *
      current_state->getGlobalLinkTransform(cmd.header.frame_id);

  // Apply the transform to linear and angular velocities
  // v' = R * v  and w' = R * w
  Eigen::Vector3d translation_vector(cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z);
  Eigen::Vector3d angular_vector(cmd.twist.angular.x, cmd.twist.angular.y, cmd.twist.angular.z);
  translation_vector = tf_moveit_to_incoming_cmd_frame.linear() * translation_vector;
  angular_vector = tf_moveit_to_incoming_cmd_frame.linear() * angular_vector;

  // Update the values of the original command message to reflect the change in frame
  cmd.header.frame_id = planning_frame;
  cmd.twist.linear.x = translation_vector(0);
  cmd.twist.linear.y = translation_vector(1);
  cmd.twist.linear.z = translation_vector(2);
  cmd.twist.angular.x = angular_vector(0);
  cmd.twist.angular.y = angular_vector(1);
  cmd.twist.angular.z = angular_vector(2);
}

Eigen::Isometry3d poseFromCartesianDelta(const Eigen::VectorXd& delta_x,
                                                const Eigen::Isometry3d& base_to_tip_frame_transform,
                                                double lookahead_interval)
{
  // get a transformation matrix with the desired position change &
  Eigen::Isometry3d tf_pos_delta(Eigen::Isometry3d::Identity());
  tf_pos_delta.translate(Eigen::Vector3d(delta_x[0], delta_x[1], delta_x[2]) * lookahead_interval);

  // get a transformation matrix with desired orientation change
  Eigen::Vector3d delta_rot(delta_x[3], delta_x[4], delta_x[5]);
  double angle = delta_rot.norm();
  Eigen::Vector3d axis = angle == 0 ? Eigen::Vector3d(Eigen::Vector3d::UnitZ()) : Eigen::Vector3d(delta_rot/angle);
  Eigen::Isometry3d tf_rot_delta(Eigen::Isometry3d::Identity());
  tf_rot_delta.rotate(Eigen::AngleAxisd(angle * lookahead_interval, axis));

  auto tf_result = tf_pos_delta * base_to_tip_frame_transform;
  tf_result.linear() = tf_rot_delta.linear() * tf_result.linear();
  return tf_result;
}

Eigen::VectorXd cartesianDeltaFromPoses(const Eigen::Isometry3d& pose_start, const Eigen::Isometry3d& pose_end,
                                          double lookahead_interval)
{
  Eigen::Vector3d pos_diff = (pose_end.translation() - pose_start.translation())/lookahead_interval;
  Eigen::AngleAxisd rot_diff(pose_start.linear().inverse()*pose_end.linear());
  rot_diff.axis() = pose_start.linear()*rot_diff.axis();

  Eigen::VectorXd delta_x(6);
  delta_x << pos_diff, (rot_diff.angle()/lookahead_interval * rot_diff.axis());
  return delta_x;
}

Eigen::Isometry3d closestPoseOnLine(const Eigen::Isometry3d& line_start, const Eigen::VectorXd& line_delta,
                                    const Eigen::Isometry3d& pose)
{
  Eigen::VectorXd diff_vector = cartesianDeltaFromPoses(line_start, pose, 1.0);
  double projection = line_delta.dot(diff_vector);
  return poseFromCartesianDelta(projection * line_delta, line_start, 1.0);
}

double getVelocityScalingFactor(const moveit::core::JointModelGroup* joint_model_group, const Eigen::VectorXd& velocity)
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
      velocity_scaling_factor = std::min(velocity_scaling_factor, bounded_velocity / unbounded_velocity);
    }
    ++joint_delta_index;
  }

  return velocity_scaling_factor;
}

void enforceVelocityLimits(const moveit::core::JointModelGroup* joint_model_group, const double publish_period,
                           sensor_msgs::msg::JointState& joint_state, const double override_velocity_scaling_factor)
{
  // Get the velocity scaling factor
  Eigen::VectorXd velocity =
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(joint_state.velocity.data(), joint_state.velocity.size());
  double velocity_scaling_factor = override_velocity_scaling_factor;
  // if the override velocity scaling factor is approximately zero then the user is not overriding the value.
  if (override_velocity_scaling_factor < 0.01)
    velocity_scaling_factor = getVelocityScalingFactor(joint_model_group, velocity);

  // Take a smaller step if the velocity scaling factor is less than 1
  if (velocity_scaling_factor < 1)
  {
    Eigen::VectorXd velocity_residuals = (1 - velocity_scaling_factor) * velocity;
    Eigen::VectorXd positions =
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(joint_state.position.data(), joint_state.position.size());
    positions -= velocity_residuals * publish_period;

    velocity *= velocity_scaling_factor;
    // Back to sensor_msgs type
    joint_state.velocity = std::vector<double>(velocity.data(), velocity.data() + velocity.size());
    joint_state.position = std::vector<double>(positions.data(), positions.data() + positions.size());
  }
}

std::vector<const moveit::core::JointModel*>
enforcePositionLimits(sensor_msgs::msg::JointState& joint_state, const double joint_limit_margin,
                      const moveit::core::JointModelGroup* joint_model_group)
{
  // Halt if we're past a joint margin and joint velocity is moving even farther past
  double joint_angle = 0;
  std::vector<const moveit::core::JointModel*> joints_to_halt;
  for (auto joint : joint_model_group->getActiveJointModels())
  {
    for (std::size_t c = 0; c < joint_state.name.size(); ++c)
    {
      // Use the most recent robot joint state
      if (joint_state.name[c] == joint->getName())
      {
        joint_angle = joint_state.position.at(c);
        break;
      }
    }

    if (!joint->satisfiesPositionBounds(&joint_angle, -joint_limit_margin))
    {
      const std::vector<moveit_msgs::msg::JointLimits>& limits = joint->getVariableBoundsMsg();

      // Joint limits are not defined for some joints. Skip them.
      if (!limits.empty())
      {
        // Check if pending velocity command is moving in the right direction
        auto joint_itr = std::find(joint_state.name.begin(), joint_state.name.end(), joint->getName());
        auto joint_idx = std::distance(joint_state.name.begin(), joint_itr);

        if ((joint_state.velocity.at(joint_idx) < 0 && (joint_angle < (limits[0].min_position + joint_limit_margin))) ||
            (joint_state.velocity.at(joint_idx) > 0 && (joint_angle > (limits[0].max_position - joint_limit_margin))))
        {
          joints_to_halt.push_back(joint);
        }
      }
    }
  }
  return joints_to_halt;
}

/** \brief Calculate a velocity scaling factor, due to existence of movements in drifting dimensions
 * @param[in] delta_theta          The commanded joint speeds
 * @param[in] jacobian             The Jacobian
 * @param[in] drift_dimensions     The drift dimensions which were ignored in delta_theta computation but do have an impact here
 * @param[in] drifting_dimension_multipliers    Multiplier applied to each of the drifting dimensions
 * @param[in] nondrifting_dimension_multipliers Multiplier applied to each of the non-drifting dimensions
 * @param[in] scaling_factor_power Power to be applied on final non-drifting/total speed ratio
 */
double velocityScalingFactorForDriftDimensions(const Eigen::VectorXd& delta_theta,
                                              const Eigen::MatrixXd& jacobian,
                                              const std::array<bool, 6> &drift_dimensions,
                                              const std::vector<double> &drifting_dimension_multipliers,
                                              const std::vector<double> &nondrifting_dimension_multipliers,
                                              const double scaling_factor_power,
                                              const std::array<bool, 6> &ignored_dimensions)
{
  if (scaling_factor_power == 0)  // this would always result in 1, skip computations
    return 1;
  const Eigen::VectorXd delta_x = jacobian*delta_theta;

  // compute speeds
  double nondrifting_velocity_sqr = 0;
  double total_velocity_sqr = 0;
  for (size_t i = 0; i < drift_dimensions.size(); i++) {
    if (ignored_dimensions[i])
      continue;
    double multiplied_velocity = delta_x[i]*(drift_dimensions[i] ? drifting_dimension_multipliers[i] : nondrifting_dimension_multipliers[i]);
    double multiplied_velocity_sqr = multiplied_velocity*multiplied_velocity;
    if (!drift_dimensions[i])
      nondrifting_velocity_sqr += multiplied_velocity_sqr;
    total_velocity_sqr += multiplied_velocity_sqr;
  }
  if (total_velocity_sqr == 0)  // no movement of end effector at all -> no need to constrain anything
    return 1;
  return pow(nondrifting_velocity_sqr/total_velocity_sqr, scaling_factor_power/2);  // pow(sqrt(x), p) is the same as pow(x, p/2)
}

}  // namespace moveit_servo
