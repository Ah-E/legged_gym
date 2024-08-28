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
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from .little_hand_config import LittleHandRoughCfg
import sys


#ik
from .wow_little_ik import WowLittle_IK


class LittleHand(LeggedRobot):
    def __init__(self, cfg: LittleHandRoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._init_ik()

        # left_pose = np.eye(4)
        # right_pose = np.eye(4)
        # left_pose[0:3,3] = np.array([0.0,0.2677,-0.5507])
        # right_pose[0:3,3] = np.array([0.0,-0.2677,-0.5510])
        # left_pose = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.2677],[0.,0.,1.,-0.55],[0.,0.,0.,1.]])
        left_pose = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.3],[0.,0.,1.,-0.5],[0.,0.,0.,1.]])
        right_pose = np.array([[0.,0.,-1,0.4],[1.,0.,0.,-0.3],[0.,-1.,0.,-0.3],[ 0.,0.,0.,1.]])
        init_dof_pos = (torch.cat((self.dof_pos[0,0:7],self.dof_pos[0,8:15]),dim=0)).cpu().numpy()
        # self.q_control = self.ik(left_pose,right_pose,init_dof_pos)
        self.B[:] = self.ik(left_pose,right_pose,init_dof_pos)


        self._prepare_reward_function()
        self.init_done = True
    
    def _init_ik(self):
        urdf_file = '../../resources/robots/wow_little/urdf/up_body_wo_dummy.urdf'
        urdf_path = '../../resources/robots/wow_little/urdf'
        self.littlehand_ik = WowLittle_IK(urdf_file,urdf_path)
        
    def ik(self,left_pose,right_pose,init_dof_pos):
        q_ik,flag = self.littlehand_ik.ik_fun(left_pose,right_pose,init_dof_pos)
        print("q_ik: ",q_ik)
        if flag:
            # q_ik = np.concatenate((q_ik[0:7],np.zeros(2),q_ik[7:14],np.zeros(24)))
            q_ik = np.concatenate((q_ik[0:7],np.zeros(1),q_ik[7:14],np.zeros(1)))
            Q = torch.from_numpy(q_ik)
        else:
            print("ik fail")
        return Q




    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        ########-------------------------------- my scripts -------------------------------########

        
        self.time = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.uff = torch.zeros(self.num_envs,self.num_actions,dtype=torch.float, device=self.device) 
        self.ufb = torch.zeros(self.num_envs,self.num_actions,dtype=torch.float, device=self.device)
        # self.u = torch.zeros(self.num_envs,self.num_actions,dtype=torch.float, device=self.device)
        self.ufb_last = torch.zeros(self.num_envs,self.num_actions,dtype=torch.float, device=self.device)
        self.filter_alpha = 0.8

        self.A =  torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device)
        self.B =  torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device)#bias

        self.T = 4. * torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device) # time of a walking period

        ########-------------------------------- my scripts -------------------------------########

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        ########-------------------------------- my scripts -------------------------------########
        self.time += self.cfg.control.decimation * self.dt
        self.time *= 1.*(self.time<self.T)
        self.uff = uff_cos(self.A,self.B,self.time,self.T)
        self.ufb = (1-self.filter_alpha)*self.actions + self.filter_alpha*self.ufb_last
        # print(self.actions[0,:])
        self.ufb_last = 1.*self.ufb
        ########-------------------------------- my scripts -------------------------------########
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            #self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            ########-------------------------------- my scripts -------------------------------########
            self.torques = self._compute_torques(self.uff).view(self.torques.shape) #+ self.ufb
            # print(self.torques)
            # print(self.uff)
            # if self.time[0,0] == 0.0:
            #     print("self.uff:", self.uff)
            #     print("self.dof_pos:", self.dof_pos)
            #     print("self.torques:", self.torques)

            ########-------------------------------- my scripts -------------------------------########
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.uff,
                                    ),dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        #self.reset_buf |= self.time_out_buf
        ########-------------------------------- my scripts -------------------------------########
        #roll_cutoff = torch.abs(self.roll) > 1.1
        #pitch_cutoff = torch.abs(self.pitch) > 1.1
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        #self.reset_buf |= roll_cutoff
        #self.reset_buf |= pitch_cutoff
        self.time *= (1.-1.*self.reset_buf).unsqueeze(1)
        ########-------------------------------- my scripts -------------------------------########


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        
    #------------ reward functions----------------
    def _reward_uff_tracking(self):
        # Penalize uff tracking error
        joint_tracking_error=torch.sum(torch.square(self.uff[:,:]-self.dof_pos[:,:]))
        # print(joint_tracking_error)

        return torch.exp(-joint_tracking_error/self.cfg.rewards.joint_tracking_sigma)


def uff_cos(A,B,t,T):
    uff = A/2.*(1.-torch.cos(4.*np.pi*t/T))+B
    uff2 = A/2.*(1.-torch.cos(4.*np.pi*t/T+np.pi))+B
    uff[:,6:12] = uff2[:,6:12]
    uff[:,18:23] = uff2[:,18:23]
    return uff