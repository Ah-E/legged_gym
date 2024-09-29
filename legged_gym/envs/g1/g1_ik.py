# conda install conda-forge::casadi
#  conda install conda-forge::meshcat-python 
# conda install tsid -c conda-forge
# git clone https://github.com/stack-of-tasks/eigenpy
# https://github.com/stack-of-tasks/tsid/blob/master/demo/demo_quadruped.py
import casadi
import pinocchio as pin
import tsid    
import meshcat.geometry

import numpy as np
from numpy import nan
from numpy.linalg import norm as norm

from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper    
from pinocchio.visualize import MeshcatVisualizer
                         
import time   
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class G1_W_IK:
    def __init__(self,pos_ori_ratio=100,urdf_file='../../../resources/robots/g1/urdf/g1.urdf',urdf_path='../../../resources/robots/g1/urdf'):
        #urdf_file和urdf_path需要修改为实际运行程序时所在目录对应的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)

        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_file,
            urdf_path,
            verbose=True
            )
        self.pos_ori_ratio = pos_ori_ratio
        self.mixed_jointsToLockIDs = [
                                    # #在解算过程中需要忽略/锁定（lock）的关节列表
                                    #   "torso_joint",

                                    #   "left_shoulder_pitch_joint",
                                    #   "left_shoulder_roll_joint",
                                    #   "left_shoulder_yaw_joint",
                                    #   "left_elbow_pitch_joint",
                                    #   "left_elbow_roll_joint",

                                    #   "right_shoulder_pitch_joint",
                                    #   "right_shoulder_roll_joint",
                                    #   "right_shoulder_yaw_joint",
                                    #   "right_elbow_pitch_joint",
                                    #   "right_elbow_roll_joint",
                                      

                                      ]
        # self.mixed_jointsToLockIDs = [
        #                             #在解算过程中需要忽略/锁定（lock）的关节列表
        #                               "torso_joint",

        #                               "left_hip_pitch_link",
        #                               "left_hip_roll_joint",
        #                               "left_hip_yaw_joint",
        #                               "left_knee_joint",
        #                               "left_ankle_pitch_joint",
        #                               "left_ankle_roll_joint",

        #                               "right_hip_pitch_joint",
        #                               "right_hip_roll_joint",
        #                               "right_hip_yaw_joint",
        #                               "right_knee_joint",
        #                               "right_ankle_pitch_joint",
        #                               "right_ankle_roll_joint"
        #                               ]
        
        #生成忽略部分关节的简化模型，锁定关节的角度配置参考reference_configuration                              
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            #增加左脚末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('LF_ee',
                      self.reduced_robot.model.getJointId('left_ankle_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T), # end efector placements related to the joint above
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            #增加右脚末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('RF_ee',
                      self.reduced_robot.model.getJointId('right_ankle_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            #增加左手末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('LH_ee',
                      self.reduced_robot.model.getJointId('left_elbow_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            #增加右手末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('RH_ee',
                      self.reduced_robot.model.getJointId('right_elbow_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )

        #迭代初始值
        self.init_data = np.zeros(self.reduced_robot.model.nq)  
        # 基于casadi创建机器人模型 Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()    
        self.model = pin.Model(self.reduced_robot.model)
        self.data = self.model.createData()   
        # 创建符号变量 Creating symbolic variables    <<<<<<<<<<<<<<<<<<<<<<<
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) # 创建关节角度符号变量（变量名，元素，每个元素的维度）
        self.cTf_lf = casadi.SX.sym("tf_lf", 4, 4) # 创建左腿末端执行器符号变量
        self.cTf_rf = casadi.SX.sym("tf_rf", 4, 4) # 创建右腿末端执行器符号变量
        self.cTf_lh = casadi.SX.sym("tf_lh", 4, 4) # 创建左手末端执行器符号变量
        self.cTf_rh = casadi.SX.sym("tf_rh", 4, 4) # 创建右手末端执行器符号变量
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq) # 计算正运动学 fk

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("LH_ee") # 获得左末端执行器坐标系id
        self.R_hand_id = self.reduced_robot.model.getFrameId("RH_ee") # 获得右末端执行器坐标系id
        self.L_foot_id = self.reduced_robot.model.getFrameId("LF_ee") # 获得左末端执行器坐标系id
        self.R_foot_id = self.reduced_robot.model.getFrameId("RF_ee") # 获得右末端执行器坐标系id

        
        # 优化过程中用到的误差形式（位姿误差向量）
        self.position_error = casadi.Function(
            "position_error",
            [self.cq, self.cTf_lf, self.cTf_rf,self.cTf_lh,self.cTf_rh],
            [
                casadi.vertcat(
                    (self.cdata.oMf[self.L_foot_id].translation - self.cTf_lf[:3, 3]),
                    (self.cdata.oMf[self.R_foot_id].translation - self.cTf_rf[:3, 3]),
                    (self.cdata.oMf[self.L_hand_id].translation - self.cTf_lh[:3, 3]),
                    (self.cdata.oMf[self.R_hand_id].translation - self.cTf_rh[:3, 3])
                )
            ]
        )
        self.orientation_error = casadi.Function(
            "orientation_error",
            [self.cq, self.cTf_lf, self.cTf_rf,self.cTf_lh,self.cTf_rh],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_foot_id].rotation.T @ self.cTf_lf[:3, :3]),
                    cpin.log3(self.cdata.oMf[self.R_foot_id].rotation.T @ self.cTf_rf[:3, :3]),
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation.T @ self.cTf_lh[:3, :3]),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation.T @ self.cTf_rh[:3, :3]),                   
                )
            ]
        )

        # 设置位置误差和姿态误差权重
        orientation_weight = 1.0  # 姿态误差权重可以较小
        position_weight = pos_ori_ratio  # 优先保证位置精准度，设定较大权重


        # 定义优化问题Defining the optimization problem
        self.opti = casadi.Opti() # 
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)# 定义优化变量维度
        # 如果需要平滑处理，启用以下注释掉的代码
        # self.param_q_ik_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_lf = self.opti.parameter(4, 4) # 定义一个辅助优化参数（用以优化左侧末端执行器位姿误差）
        self.param_tf_rf = self.opti.parameter(4, 4) # 定义一个辅助优化参数（用以优化右侧末端执行器位姿误差）
        self.param_tf_lh = self.opti.parameter(4, 4) # 定义一个辅助优化参数（用以优化左侧末端执行器位姿误差）
        self.param_tf_rh = self.opti.parameter(4, 4) # 定义一个辅助优化参数（用以优化右侧末端执行器位姿误差）
        # 优化目标：权重 * 位置误差 + 姿态误差
        self.totalcost = position_weight * casadi.sumsqr(self.position_error(self.var_q, self.param_tf_lf, self.param_tf_rf,self.param_tf_lh, self.param_tf_rh)) + \
                        orientation_weight * casadi.sumsqr(self.orientation_error(self.var_q, self.param_tf_lf, self.param_tf_rf,self.param_tf_lh, self.param_tf_rh))

        # self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf_l, self.param_tf_r))# 先通过self.error求出左右末端执行器位姿误差，再求其平方和      

  
        self.regularization = casadi.sumsqr(self.var_q) # 正则化
        # 如果需要平滑处理，启用以下注释掉的代码
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.param_q_ik_last)

        # 设置优化约束和目标 Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # 设置优化目标形式
        self.opti.minimize(20 * self.totalcost + self.regularization) # self.regularization 防止过渡拟合
        
        # 如果需要平滑处理，启用以下注释掉的代码
        # self.opti.minimize(20 * self.totalcost + 0.001*self.regularization + 0.1*self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0, # 关闭输出 Turn off the solver’s output information.
                'max_iter':30, # 最大迭代次数
                'tol':5e-3 # 收敛容差 convergence tolerance
            },
            'print_time':False
        }
        self.opti.solver("ipopt", opts) 
        for i in range(self.model.nq+1):
            print(i,": ",self.model.names[i])



    # 尺寸匹配
    def adjust_pose(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=1.20):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose
    

    
    
    #正运动学求解
    def fk_fun(self,dof_pos):
        pin.forwardKinematics(self.model, self.data, dof_pos)
        # update forward kinematics of end effector frame
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId("LF_ee"))
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId("RF_ee"))
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId("LH_ee"))
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId("RH_ee"))
        T_LF_ = self.data.oMf[self.model.getFrameId("LF_ee")]
        T_RF_ = self.data.oMf[self.model.getFrameId("RF_ee")]
        T_LH_ = self.data.oMf[self.model.getFrameId("LH_ee")]
        T_RH_ = self.data.oMf[self.model.getFrameId("RH_ee")]
        T_LF = np.eye(4)
        T_RF = np.eye(4)
        T_LH = np.eye(4)
        T_RH = np.eye(4)
        
        T_LF[0:3,0:3] = T_LF_.rotation
        T_LF[0:3,3] = T_LF_.translation
        T_RF[0:3,0:3] = T_RF_.rotation
        T_RF[0:3,3] = T_RF_.translation
        
        T_LH[0:3,0:3] = T_LH_.rotation
        T_LH[0:3,3] = T_LH_.translation
        T_RH[0:3,0:3] = T_RH_.rotation
        T_RH[0:3,3] = T_RH_.translation
        # T_L = self.cdata.oMf[self.L_hand_id] 
        # T_R = self.cdata.oMf[self.R_hand_id]
        return T_LF,T_RF,T_LH,T_RH
    
    #逆运动学求解
    def ik_fun(self, left_foot_pose, right_foot_pose, left_hand_pose, right_hand_pose, motorstate=None, motorV=None):
        
        # 迭代初始值
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)


        # self.vis.viewer['L_ee_target'].set_transform(left_pose)
        # self.vis.viewer['R_ee_target'].set_transform(right_pose)
        # origin_pose = np.eye(4)
        # self.vis.viewer['Origin'].set_transform(origin_pose)

        # left_pose, right_pose = self.adjust_pose(left_pose, right_pose)
        #设置优化变量及目标
        self.opti.set_value(self.param_tf_lf, left_foot_pose)
        self.opti.set_value(self.param_tf_rf, right_foot_pose)
        self.opti.set_value(self.param_tf_lh, left_hand_pose)
        self.opti.set_value(self.param_tf_rh, right_hand_pose)
        try:
            # sol = self.opti.solve()
            # 设置约束
            sol = self.opti.solve_limited()
            #求解优化问题
            sol_q = self.opti.value(self.var_q)

            # self.vis.display(sol_q)
            self.init_data = sol_q

            # 如果需要计算力矩，启用以下代码
            # 求解力矩
            # if motorV is not None:
            #     v =motorV * 0.0
            # else:
            #     v = (sol_q-self.init_data ) * 0.0
            # tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q,v,np.zeros(self.reduced_robot.model.nv))
            
            
            return sol_q,True
            # 如果需要计算力矩，启用以下代码
            # return sol_q, tau_ff ,True
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            # return sol_q, '',False
            return sol_q,False
        
class G1_W_ID:
    def __init__(self,pos_ori_ratio=100,urdf_file='../../../resources/robots/g1/urdf/g1.urdf',urdf_path='../../../resources/robots/g1/urdf'):
        #urdf_file和urdf_path需要修改为实际运行程序时所在目录对应的路径
        # build model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_file,
            urdf_path,
            verbose=True
            )
        self.pos_ori_ratio = pos_ori_ratio
        self.mixed_jointsToLockIDs = [
                                    # #在解算过程中需要忽略/锁定（lock）的关节列表
                                    #   "torso_joint",

                                    #   "left_shoulder_pitch_joint",
                                    #   "left_shoulder_roll_joint",
                                    #   "left_shoulder_yaw_joint",
                                    #   "left_elbow_pitch_joint",
                                    #   "left_elbow_roll_joint",

                                    #   "right_shoulder_pitch_joint",
                                    #   "right_shoulder_roll_joint",
                                    #   "right_shoulder_yaw_joint",
                                    #   "right_elbow_pitch_joint",
                                    #   "right_elbow_roll_joint",
                                      
                                    #   "left_hip_pitch_link",
                                    #   "left_hip_roll_joint",
                                    #   "left_hip_yaw_joint",
                                    #   "left_knee_joint",
                                    #   "left_ankle_pitch_joint",
                                    #   "left_ankle_roll_joint",

                                    #   "right_hip_pitch_joint",
                                    #   "right_hip_roll_joint",
                                    #   "right_hip_yaw_joint",
                                    #   "right_knee_joint",
                                    #   "right_ankle_pitch_joint",
                                    #   "right_ankle_roll_joint"
                                      ]

        
        #生成忽略部分关节的简化模型，锁定关节的角度配置参考reference_configuration                              
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        self.reduced_robot.model.addFrame(
            #增加左脚末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('LF_ee',
                      self.reduced_robot.model.getJointId('left_ankle_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T), # end efector placements related to the joint above
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            #增加右脚末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('RF_ee',
                      self.reduced_robot.model.getJointId('right_ankle_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            #增加左手末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('LH_ee',
                      self.reduced_robot.model.getJointId('left_elbow_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            #增加右手末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('RH_ee',
                      self.reduced_robot.model.getJointId('right_elbow_roll_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addJoint(
            self.reduced_robot.getJointId('universe'),# 添加到根关节，父关节ID为‘universe’
            pin.JointModelFreeFlyer(),
            pin.SE3.identity,
            "free_flyer_joint")
        
    def id_fun(self):
        # 创建动力学优化问题
        q = np.zeros(self.reduced_robot.model.nq)
        q[6] = 1.0  # 设置自由飞行器的四元数部分
        q[2] += 0.9  # 提升机器人高度
        v = np.zeros(self.reduced_robot.nv)
        assert [self.reduced_robot.model().existFrame(name) for name in contact_frames]
        invdyn = tsid.InverseDynamicsFormulationAccForce("tsid",self.reduced_robot, False)
        invdyn.computeProblemData(0.0,q,v)
        data = invdyn.data()
        
        
        # 接触约束contact constraints
        mu = 0.3 #friction coefficient
        fMin = 1.0 # minumum normal force
        fMax = 100.0 # maximum normal force
        # contact_frames = ["LF_ee","RF_ee","LH_ee","RH_ee"]
        contact_frames = ["LF_ee","RF_ee"]
        contactNormal = np.array([0.0, 0.0, 1.0]) # contact normal direction
        w_contactForceRef = 1e-5  # weight of force regularization task       
        kp_contact = 10.0  # proportional gain of contact constraint
        contacts = 4*[None]
        for i,name in enumerate(contact_frames):
            contacts[i] = tsid.ContactPoint(name,
                                            self.reduced_robot,
                                            name,
                                            contactNormal,
                                            mu,
                                            fMin,
                                            fMax)
            contacts[i].setKp(kp_contact*np.ones(3))
            contacts[i].setKd(2.0*np.sqrt(kp_contact)*np.ones(3))
            invdyn.addRigidContact(contacts[i], True)
            H_rf_ref = self.reduced_robot.framePosition(self.data,self.reduced_robotmodel().getFrameId(name))
            contacts[i].setReference(H_rf_ref)
            contacts[i].useLocalFrame(False)
            invdyn.addRigidContact(contacts[i], w_contactForceRef, 1.0, 1)

        # simulation steps
        N_SIMULATION = 100
        
        # 设置重力加速度的大小
        g = 9.81
        gravity_vector = np.array([0.0, 0.0, -g])
        invdyn.setGravity(gravity_vector)

        #Place the robot onto the ground
        id_contact = self.reduced_robot.model.getFrameId(contact_frames[0])
        q[2] -= self.reduced_robot.framePosition(self.data,id_contact).translation[2]
        self.reduced_robot.computeAllTerms(data,q,v)
        
        # CoM task
        w_com = 1.0  # weight of center of mass task
        kp_com = 10.0  # proportional gain of center of mass task
        comTask = tsid.TaskComEquality("task-com", self.reduced_robot)
        comTask.setKp(kp_com * np.ones(3)) # 设置质心任务的比例增益
        comTask.setKd(2.0 * np.sqrt(kp_com) * np.ones(3)) #设置质心任务的微分增益
        invdyn.addMotionTask(comTask, w_com, 1, 0.0)#将质心任务添加到逆运动学问题中，并设置权重和优先级
        # CoM reference trajectory
        com_ref = self.reduced_robot.com(data)#获取当前质心位置
        trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)#创建一个常值轨迹，用于描述质心位置
        sampleCom = trajCom.computeNext()#计算下一个样本点
        
        com_pos = np.empty((3, N_SIMULATION)) * nan
        com_vel = np.empty((3, N_SIMULATION)) * nan
        com_acc = np.empty((3, N_SIMULATION)) * nan

        com_pos_ref = np.empty((3, N_SIMULATION)) * nan
        com_vel_ref = np.empty((3, N_SIMULATION)) * nan
        com_acc_ref = np.empty((3, N_SIMULATION)) * nan
        com_acc_des = (
            np.empty((3, N_SIMULATION)) * nan
        )  # acc_des = acc_ref - Kp*pos_err - Kd*vel_err
        sol_tau = np.empty((self.reduced_robot.model.nq, N_SIMULATION)) * nan
        sol_q = np.empty((self.reduced_robot.model.nq, N_SIMULATION)) * nan
        
        #posture task
        w_posture = 1e-2  # weight of joint posture task
        kp_posture = 10.0  # proportional gain of joint posture task
        postureTask = tsid.TaskJointPosture("task-posture", self.reduced_robot)
        postureTask.setKp(kp_posture * np.ones(self.reduced_robot.nv - 6))
        postureTask.setKd(2.0 * np.sqrt(kp_posture) * np.ones(self.reduced_robotc.nv - 6))
        invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)
        #1 是任务的优先级。在tsid中，任务可以被赋予不同的优先级，其中数值越小，优先级越高。这里设置为1意味着质心任务的优先级是中等，因为通常会有更重要的任务（比如保持平衡）可能被赋予优先级0。
        #0.0 是任务激活的标志。在tsid中，你可以通过设置一个阈值来决定任务是否激活。如果设置为0.0，任务始终是激活的；如果设置为大于0.0的值，那么只有当任务的误差超过这个阈值时，任务才会被激活。这里设置为0.0，意味着质心任务始终处于激活状态
        q_ref = q[7:]#获取除了自由飞行器关节之外的关节配置
        trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)#创建一个常值轨迹，用于描述关节配置

        print(
            "Create QP solver with ",
            invdyn.nVar,
            " variables, ",
            invdyn.nEq,
            " equality and ",
            invdyn.nIn,
            " inequality constraints",
        )
        
        solver = tsid.SolverHQuadProgFast("qp solver")
        solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)  # 设置QP求解器的大小
        t = 0.0
        dt = 0.001
        for i in range(0, N_SIMULATION):
            time_start = time.time()
            # 设置跟踪轨迹
            # sampleCom.pos(offset + np.multiply(amp, np.sin(two_pi_f * t)))
            # sampleCom.vel(np.multiply(two_pi_f_amp, np.cos(two_pi_f * t)))
            # sampleCom.acc(np.multiply(two_pi_f_squared_amp, -np.sin(two_pi_f * t)))

            comTask.setReference(sampleCom)
            samplePosture = trajPosture.computeNext()
            postureTask.setReference(samplePosture)

            HQPData = invdyn.computeProblemData(t, q, v)
            if i == 0:
                HQPData.print_all()

            sol = solver.solve(HQPData)
            if sol.status != 0:
                print("[%d] QP problem could not be solved! Error code:" % (i), sol.status)
                break

            tau = invdyn.getActuatorForces(sol)
            dv = invdyn.getAccelerations(sol)
            

            PRINT_N = 10
            if i % PRINT_N == 0:
                print(f"Time {t:.3f}")
                print("\tNormal forces: ", end=" ")
                for contact in contacts:
                    if invdyn.checkContact(contact.name, sol):
                        f = invdyn.getContactForce(contact.name, sol)
                        print(f"{contact.getNormalForce(f):4.1f}", end=" ")

                print(
                    "\n\ttracking err {}: {:.3f}".format(
                        comTask.name.ljust(20, "."), norm(comTask.position_error, 2)
                    )
                )
                print(f"\t||v||: {norm(v, 2):.3f}\t ||dv||: {norm(dv):.3f}")

            v_mean = v + 0.5 * dt * dv
            v += dt * dv
            q = pin.integrate(self.reduced_robot.model(), q, dt * v_mean)
            t += dt


            time_spent = time.time() - time_start
            if time_spent < dt:
                time.sleep(dt - time_spent)
                
            sol_tau[:, i] = tau
            sol_q[:,i] = q
            com_pos[:, i] = self.reduced_robot.com(invdyn.data())
            com_vel[:, i] = self.reduced_robot.com_vel(invdyn.data())
            com_acc[:, i] = comTask.getAcceleration(dv)
            com_pos_ref[:, i] = sampleCom.pos()
            com_vel_ref[:, i] = sampleCom.vel()
            com_acc_ref[:, i] = sampleCom.acc()
            com_acc_des[:, i] = comTask.getDesiredAcceleration
            
        return com_pos,sol_tau,sol_q      
        
        
# LF: [[ 1.00000000e+00  0.00000000e+00 -5.55111512e-17  8.06828946e-07]
#  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  1.17871300e-01]
#  [ 5.55111512e-17  0.00000000e+00  1.00000000e+00 -7.19675917e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# RF: [[ 1.00000000e+00  0.00000000e+00 -5.55111512e-17  8.06828946e-07]
#  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00 -1.17871300e-01]
#  [ 5.55111512e-17  0.00000000e+00  1.00000000e+00 -7.19675917e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# LH: [[1.         0.         0.         0.09396   ]
#  [0.         1.         0.         0.16207805]
#  [0.         0.         1.         0.08491301]
#  [0.         0.         0.         1.        ]]
# RH: [[ 1.          0.          0.          0.09396   ]
#  [ 0.          1.          0.         -0.16207805]
#  [ 0.          0.          1.          0.08491301]
#  [ 0.          0.          0.          1.        ]]
if __name__ == "__main__":
    

    #单独测试id时启用以下代码
    G1ID = G1_W_ID()
    com_pos,sol_tau,sol_q = G1ID.id_fun()
    
    # #单独测试ik时启用以下代码
    # G1IK = G1_W_IK()
    # dof_pos = np.zeros(23)
    # LF,RF,LH,RH = G1IK.fk_fun(dof_pos)
    # print("LF:",LF)
    # print("RF:",RF)    
    # print("LH:",LH)
    # print("RH:",RH)
    # left_foot_pose = np.eye(4)
    # right_foot_pose = np.eye(4)
    # left_hand_pose = np.eye(4)
    # right_hand_pose = np.eye(4)
    
    # left_foot_pose[0:3,3] = np.array(   [0.0,   1.17871300e-01, -5.19675917e-01])
    # right_foot_pose[0:3,3] = np.array(  [0.0,   -1.17871300e-01,-7.19675917e-01])
    # left_hand_pose[0:3,3] = np.array(   [0.09396,   0.16207805, 0.08491301])
    # right_hand_pose[0:3,3] = np.array(  [0.09396,   -0.16207805,0.08491301])

    # sol_q, flag = G1IK.ik_fun(left_foot_pose, right_foot_pose,left_hand_pose,right_hand_pose)
    # # print("sol_q: ",sol_q)
    # LF,RF,LH,RH = G1IK.fk_fun(sol_q)
    # print("sol:")
    # print("LF:",LF)
    # print("RF:",RF)    
    # print("LH:",LH)
    # print("RH:",RH)



        
        
        
        
        
     