# conda install conda-forge::casadi
#  conda install conda-forge::meshcat-python 

import casadi
import meshcat.geometry
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin                
from pinocchio.robot_wrapper import RobotWrapper    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class WowLittle_IK:
    def __init__(self,urdf_file='../../../resources/robots/wow_little/urdf/up_body_wo_dummy.urdf',urdf_path='../../../resources/robots/wow_little/urdf'):
        #urdf_file和urdf_path需要修改为实际运行程序时所在目录对应的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)

        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_file,
            urdf_path,
            verbose=True
            )
        self.mixed_jointsToLockIDs = [
                                    #在解算过程中需要忽略/锁定（lock）的关节列表
                                      "left_8_joint",
                                      "right_8_joint"
                                      ]
        #生成忽略部分关节的简化模型，锁定关节的角度配置参考reference_configuration                              
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            #增加左手末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_7_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T), # end efector placements related to the joint above
                      pin.FrameType.OP_FRAME)
        )
        self.reduced_robot.model.addFrame(
            #增加右手末端执行器end effector坐标系，并描述该坐标系的相对位置参考关节坐标系及偏移位姿
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_7_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.0,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        #迭代初始值
        self.init_data = np.zeros(self.reduced_robot.model.nq)

        # 基于casadi创建机器人模型 Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.model = pin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.data = self.model.createData()
   

        # 创建符号变量 Creating symbolic variables    <<<<<<<<<<<<<<<<<<<<<<<
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) # 创建关节角度符号变量（变量名，元素，每个元素的维度）
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4) # 创建左手末端执行器符号变量
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4) # 创建左手末端执行器符号变量
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq) # 计算正运动学 fk

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee") # 获得左末端执行器坐标系id
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee") # 获得右末端执行器坐标系id
        
        # 优化过程中用到的误差形式（位姿误差向量）
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf_l, self.cTf_r],#符号变量
            [
                #cpin.log6: 计算位资的对数映射 calculate the logarithmic map of the pose error
                # casadi.certcat: 合并左右手误差到一个向量共同计算 connect the two end-effector error vectors to optimize simultaneously
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.L_hand_id].inverse() * cpin.SE3(self.cTf_l)
                    ).vector,
                    cpin.log6(
                        self.cdata.oMf[self.R_hand_id].inverse() * cpin.SE3(self.cTf_r)
                    ).vector
                )
            ],#输出参数
        )

        # 定义优化问题Defining the optimization problem
        self.opti = casadi.Opti() # 
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)# 定义优化变量维度
        # 如果需要平滑处理，启用以下注释掉的代码
        # self.param_q_ik_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4) # 定义一个辅助优化参数（用以优化左侧末端执行器位姿误差）
        self.param_tf_r = self.opti.parameter(4, 4) # 定义一个辅助优化参数（用以优化右侧末端执行器位姿误差）
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf_l, self.param_tf_r))# 先通过self.error求出左右末端执行器位姿误差，再求其平方和      
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

    # 尺寸匹配
    def adjust_pose(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=1.20):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose
    
    def fk_fun(self,dof_pos):
        pin.forwardKinematics(self.model, self.data, dof_pos)
        # update forward kinematics of end effector frame
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId("L_ee"))
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId("R_ee"))
        T_L_ = self.data.oMf[self.model.getFrameId("L_ee")]
        T_R_ = self.data.oMf[self.model.getFrameId("R_ee")]
        T_L = np.eye(4)
        T_R = np.eye(4)

        T_L[0:3,0:3] = T_L_.rotation
        T_L[0:3,3] = T_L_.translation
        T_R[0:3,0:3] = T_R_.rotation
        T_R[0:3,3] = T_R_.translation
        # T_L = self.cdata.oMf[self.L_hand_id] 
        # T_R = self.cdata.oMf[self.R_hand_id]
        return T_L,T_R
        
        
    #逆运动学求解
    def ik_fun(self, left_pose, right_pose, motorstate=None, motorV=None):
        
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
        self.opti.set_value(self.param_tf_l, left_pose)
        self.opti.set_value(self.param_tf_r, right_pose)

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
            print("ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            # return sol_q, '',False
            return sol_q,False


def Ry(theta):
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    return R

if __name__ == "__main__":
    
    #单独测试ik时启用以下代码
    littlehand = WowLittle_IK()
    left_pose = np.eye(4)
    right_pose = np.eye(4)
    left_pose[0:3,3] = np.array([0.2,0.25,-0.3])
    left_pose[0:3,0:3] = Ry(1)
    print("Ry: ",Ry(1))
    right_pose[0:3,3] = np.array([0.0,-0.25,-0.5])
    # sol_q, tau_ff,flag = zerohandIK.ik_fun(left_pose, right_pose)


    # t1 = time.time()
    # num = 100
    # for i in range(num):
    #     sol_q, flag = littlehand.ik_fun(left_pose, right_pose)
    # print("use time: ", (time.time() - t1)/num)

    sol_q, flag = littlehand.ik_fun(left_pose, right_pose)
    T_L,T_R = littlehand.fk_fun(sol_q)
    print("sol_q: ",sol_q)
    print("T_L: ", T_L)
    print("T_R: ", T_R)

    # print("tau_ff: ",tau_ff)
    
