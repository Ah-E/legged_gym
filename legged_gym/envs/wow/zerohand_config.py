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

class ZerohandRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 1
        num_observations = 293#197
        num_actions = 40#16

    
    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 2.0] # x,y,z [m]-1.
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 7
            'left_shoulder_pitch_joint': 0.00,   # [rad]-1.5,1.5
            'left_shoulder_roll_joint': 0.,   # [rad]-1.75,0.75
            'left_shoulder_yaw_joint': 0.,  # [rad]-1.5,1.5
            'left_elbow_joint': 0.,   # [rad]0,2.5
            'left_wrist_yaw_joint': 0.,    # [rad]-1.5,1.5
            'left_wrist_pitch_joint': 0.,    # [rad]-1.15,1.5
            'left_wrist_roll_joint': 0.,    # [rad]-1 1

            # 7
            'right_shoulder_pitch_joint': 0.,   # [rad] -1.5,1.5
            'right_shoulder_roll_joint':0.,   # [rad]-0.75,1.75
            'right_shoulder_yaw_joint': 0.,  # [rad]-1.5,1.5
            'right_elbow_joint': 0.,   # [rad] -2.5,0
            'right_wrist_yaw_joint': 0.,   # [rad]-1.5,1.5
            'right_wrist_pitch_joint': 0.,   # [rad] -1.15,1.15
            'right_wrist_roll_joint': 0.,   # [rad] -1,1

            # 12
            'L_index_proximal_joint':0.,
            'L_index_intermediate_joint':0.,
            'L_middle_proximal_joint':0.,
            'L_middle_intermediate_joint':0.,
            'L_pinky_proximal_joint':0.,
            'L_pinky_intermediate_joint':0.,
            'L_ring_proximal_joint':0.,
            'L_ring_intermediate_joint':0.,
            'L_thumb_proximal_yaw_joint':0.,
            'L_thumb_proximal_pitch_joint':0.,
            'L_thumb_intermediate_joint':0.,
            'L_thumb_distal_joint':0.,

            #2
            'neck_pitch_joint':0.,
            'neck_yaw_joint':0.,

            #12
            'R_index_proximal_joint':0.,
            'R_index_intermediate_joint':0.,
            'R_middle_proximal_joint':0.,
            'R_middle_intermediate_joint':0.,
            'R_pinky_proximal_joint':0.,
            'R_pinky_intermediate_joint':0.,
            'R_ring_proximal_joint':0.,
            'R_ring_intermediate_joint':0.,
            'R_thumb_proximal_yaw_joint':0.,
            'R_thumb_proximal_pitch_joint':0.,
            'R_thumb_intermediate_joint':0.,
            'R_thumb_distal_joint':0.,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {   
                        'left_shoulder_pitch_joint': 200.00,
                        'left_shoulder_roll_joint': 200., 
                        'left_shoulder_yaw_joint': 200., 
                        'left_elbow_joint': 200., 
                        'left_wrist_yaw_joint': 200.,
                        'left_wrist_pitch_joint': 200., 
                        'left_wrist_roll_joint': 200., 
                        'right_shoulder_pitch_joint': 200., 
                        'right_shoulder_roll_joint': 200., 
                        'right_shoulder_yaw_joint': 200.,
                        'right_elbow_joint': 200., 
                        'right_wrist_yaw_joint':200., 
                        'right_wrist_pitch_joint': 200., 
                        'right_wrist_roll_joint': 200.,

                        'L_index_proximal_joint':0,
                        'L_index_intermediate_joint':0,
                        'L_middle_proximal_joint':0,
                        'L_middle_intermediate_joint':0,
                        'L_pinky_proximal_joint':0,
                        'L_pinky_intermediate_joint':0,
                        'L_ring_proximal_joint':0,
                        'L_ring_intermediate_joint':0,
                        'L_thumb_proximal_yaw_joint':0,
                        'L_thumb_proximal_pitch_joint':0,
                        'L_thumb_intermediate_joint':0,
                        'L_thumb_distal_joint':0,

                        'neck_pitch_joint':200.,
                        'neck_yaw_joint':200.,

                        'R_index_proximal_joint':0,
                        'R_index_intermediate_joint':0,
                        'R_middle_proximal_joint':0,
                        'R_middle_intermediate_joint':0,
                        'R_pinky_proximal_joint':0,
                        'R_pinky_intermediate_joint':0,
                        'R_ring_proximal_joint':0,
                        'R_ring_intermediate_joint':0,
                        'R_thumb_proximal_yaw_joint':0,
                        'R_thumb_proximal_pitch_joint':0,
                        'R_thumb_intermediate_joint':0,
                        'R_thumb_distal_joint':0,

                        }  # [N*m/rad]
        damping = { 
            'left_shoulder_pitch_joint': 5,   # [rad]-55,55
            'left_shoulder_roll_joint': 5,   # [rad]-575,0.75
            'left_shoulder_yaw_joint': 5,  # [rad]-55,55
            'left_elbow_joint': 5,   # [rad]0,2.5
            'left_wrist_yaw_joint': 5,    # [rad]-55,55
            'left_wrist_pitch_joint': 5,    # [rad]-555,55
            'left_wrist_roll_joint': 5,    # [rad]-5 5

            'right_shoulder_pitch_joint': 5,   # [rad] -55,55
            'right_shoulder_roll_joint': 5,   # [rad]-0.75,575
            'right_shoulder_yaw_joint': 5,  # [rad]-55,55
            'right_elbow_joint': 5,   # [rad] -2.5,0
            'right_wrist_yaw_joint': 5,   # [rad]-55,55
            'right_wrist_pitch_joint': 5,   # [rad] -555,555
            'right_wrist_roll_joint': 5,   # [rad] -1,1

            'L_index_proximal_joint':0,
            'L_index_intermediate_joint':0,
            'L_middle_proximal_joint':0,
            'L_middle_intermediate_joint':0,
            'L_pinky_proximal_joint':0,
            'L_pinky_intermediate_joint':0,
            'L_ring_proximal_joint':0,
            'L_ring_intermediate_joint':0,
            'L_thumb_proximal_yaw_joint':0,
            'L_thumb_proximal_pitch_joint':0,
            'L_thumb_intermediate_joint':0,
            'L_thumb_distal_joint':0,

            'neck_pitch_joint':1,
            'neck_yaw_joint':1,

            'R_index_proximal_joint':0,
            'R_index_intermediate_joint':0,
            'R_middle_proximal_joint':0,
            'R_middle_intermediate_joint':0,
            'R_pinky_proximal_joint':0,
            'R_pinky_intermediate_joint':0,
            'R_ring_proximal_joint':0,
            'R_ring_intermediate_joint':0,
            'R_thumb_proximal_yaw_joint':0,
            'R_thumb_proximal_pitch_joint':0,
            'R_thumb_intermediate_joint':0,
            'R_thumb_distal_joint':0,

            }   # [N*m*s/rad]     # [N*m*s/rad]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        fix_base_link = True # fixe the base of the robot
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wow_body/urdf/wow_body.urdf'
        name = "wow"
        foot_name = 'ankle'
        terminate_after_contacts_on = ['pelvis','torso','hip','knee','arm','shoulder']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        default_dof_drive_mode = 1 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
  
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


class ZerohandRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_g1'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  