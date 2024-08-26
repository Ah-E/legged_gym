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

class FrankaRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 1
        num_observations = 169
        num_actions = 16

    
    class terrain( LeggedRobotCfg.terrain):
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_1_joint':.0,
            'left_2_joint':.0,
            'left_3_joint':.0,
            'left_4_joint':.0,
            'left_5_joint':.0,
            'left_6_joint':.0,
            'left_7_joint':.0,
            'left_8_joint':.0,
            
            'right_1_joint':.0,
            'right_2_joint':.0,
            'right_3_joint':.0,
            'right_4_joint':.0,
            'right_5_joint':.0,
            'right_6_joint':.0,
            'right_7_joint':.0,
            'right_8_joint':.0,          

        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {               
            'left_1_joint':100.,
            'left_2_joint':100.,
            'left_3_joint':100.,
            'left_4_joint':100.,
            'left_5_joint':100.,
            'left_6_joint':100.,
            'left_7_joint':100.,
            'left_8_joint':100.,
                        
            'right_1_joint':100.,
            'right_2_joint':100.,
            'right_3_joint':100.,
            'right_4_joint':100.,
            'right_5_joint':100.,
            'right_6_joint':100.,
            'right_7_joint':100.,
            'right_8_joint':100.,   
                        }  # [N*m/rad]
        damping = { 
            'left_1_joint':2.,
            'left_2_joint':2.,
            'left_3_joint':2.,
            'left_4_joint':2.,
            'left_5_joint':2.,
            'left_6_joint':2.,
            'left_7_joint':2.,
            'left_8_joint':2.,
                        
            'right_1_joint':2.,
            'right_2_joint':2.,
            'right_3_joint':2.,
            'right_4_joint':2.,
            'right_5_joint':2.,
            'right_6_joint':2.,
            'right_7_joint':2.,
            'right_8_joint':2., 
                    }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"
        foot_name = 'toe'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
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

class CassieRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_cassie'

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  