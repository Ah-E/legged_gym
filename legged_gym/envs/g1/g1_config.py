# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 16
        num_observations = 225
        num_actions = 23

    
    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.9] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0

            #left leg
            'left_hip_pitch_joint': .0,
            'left_hip_roll_joint': .0,
            'left_hip_yaw_joint': .0,
            'left_knee_joint': .0,
            'left_ankle_pitch_joint': .0,
            'left_ankle_roll_joint': .0,

            # 'left_hip_pitch_joint': ,
            # 'left_hip_roll_joint': ,
            # 'left_hip_yaw_joint': ,
            # 'left_knee_joint':  ,
            # 'left_ankle_pitch_joint': ,
            # 'left_ankle_roll_joint': ,


            #right leg
            'right_hip_pitch_joint': 0.,
            'right_hip_roll_joint': 0,
            'right_hip_yaw_joint': 0.,
            'right_knee_joint': .0,
            'right_ankle_pitch_joint': .0,
            'right_ankle_roll_joint': .0,

            'torso_joint': .0,
            
            #left arm
            'left_shoulder_pitch_joint': .0,
            'left_shoulder_roll_joint': .0,
            'left_shoulder_yaw_joint': .0,
            'left_elbow_pitch_joint': .0,
            'left_elbow_roll_joint': .0,
            #right arm
            'right_shoulder_pitch_joint': .0,
            'right_shoulder_roll_joint': .0,
            'right_shoulder_yaw_joint': .0,
            'right_elbow_pitch_joint': 0,
            'right_elbow_roll_joint': .0,

            #hand
            # 'left_zero_joint': .0,
            # 'left_one_joint': .0,
            # 'left_two_joint': .0,
            # 'left_three_joint': .0,
            # 'left_four_joint': .0,
            # 'left_five_joint': .0,
            # 'left_six_joint': .0,
            
            #hand
            # 'right_zero_joint': .0,
            # 'right_one_joint': .0,
            # 'right_two_joint': .0,
            # 'right_three_joint': .0,
            # 'right_four_joint': .0,
            # 'right_five_joint': .0,
            # 'right_six_joint': .0,

        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {       
            'torso_joint': 200.0,

            'left_hip_pitch_joint': 200.0,
            'left_hip_roll_joint': 200.0,
            'left_hip_yaw_joint': 200.0,
            'left_knee_joint': 200.0,
            'left_ankle_pitch_joint': 50.0,
            'left_ankle_roll_joint': 50.0,

            'left_shoulder_pitch_joint': 200.0,
            'left_shoulder_roll_joint': 200.0,
            'left_shoulder_yaw_joint': 200.0,
            'left_elbow_pitch_joint': 200.0,
            'left_elbow_roll_joint': 200.0,

            # 'left_zero_joint': 100.0,
            # 'left_one_joint': 100.0,
            # 'left_two_joint': 100.0,
            # 'left_three_joint': 100.0,
            # 'left_four_joint': 100.0,
            # 'left_five_joint': 100.0,
            # 'left_six_joint': 100.0,

            'right_hip_pitch_joint': 200.0,
            'right_hip_roll_joint': 200.0,
            'right_hip_yaw_joint': 200.0,
            'right_knee_joint': 200.0,
            'right_ankle_pitch_joint': 50.0,
            'right_ankle_roll_joint': 50.0,

            'right_shoulder_pitch_joint': 200.0,
            'right_shoulder_roll_joint': 200.0,
            'right_shoulder_yaw_joint': 200.0,
            'right_elbow_pitch_joint': 200.0,
            'right_elbow_roll_joint': 200.0,

            # 'right_zero_joint': 100.0,
            # 'right_one_joint': 100.0,
            # 'right_two_joint': 100.0,
            # 'right_three_joint': 100.0,
            # 'right_four_joint': 100.0,
            # 'right_five_joint': 100.0,
            # 'right_six_joint': 100.0 
            }  # [N*m/rad]
        
        damping = { 
            'torso_joint': 4.0,

            'left_hip_pitch_joint': 4.0,
            'left_hip_roll_joint': 4.0,
            'left_hip_yaw_joint': 4.0,
            'left_knee_joint': 4.0,
            'left_ankle_pitch_joint': 0.2,
            'left_ankle_roll_joint': 0.2,

            'left_shoulder_pitch_joint': 4.0,
            'left_shoulder_roll_joint': 4.0,
            'left_shoulder_yaw_joint': 4.0,
            'left_elbow_pitch_joint': 0.2,
            'left_elbow_roll_joint': 0.2,

            # 'left_zero_joint': 3.0,
            # 'left_one_joint': 3.0,
            # 'left_two_joint': 3.0,
            # 'left_three_joint': 3.0,
            # 'left_four_joint': 3.0,
            # 'left_five_joint': 3.0,
            # 'left_six_joint': 3.0,

            'right_hip_pitch_joint': 4.0,
            'right_hip_roll_joint': 4.0,
            'right_hip_yaw_joint': 4.0,
            'right_knee_joint': 4.0,
            'right_ankle_pitch_joint': 0.2,
            'right_ankle_roll_joint': 0.2,

            'right_shoulder_pitch_joint': 4.0,
            'right_shoulder_roll_joint': 4.0,
            'right_shoulder_yaw_joint': 4.0,
            'right_elbow_pitch_joint': 0.2,
            'right_elbow_roll_joint': 0.2,

            # 'right_zero_joint': 3.0,
            # 'right_one_joint': 3.0,
            # 'right_two_joint': 3.0,
            # 'right_three_joint': 3.0,
            # 'right_four_joint': 3.0,
            # 'right_five_joint': 3.0,
            # 'right_six_joint': 3.0

            }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        fix_base_link = True # fixe the base of the robot
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1.urdf'
        name = "g1"
        foot_name = 'ankle'
        terminate_after_contacts_on = ['torso','head','shoulder','elbow','six']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        joint_tracking_sigma = 0.04
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            uff_tracking= 1.


class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_g1'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  