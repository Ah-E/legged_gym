from __future__ import print_function
import numpy as np
from numpy.linalg import norm, solve
# pinocchio
import pinocchio as pin
from sys import argv
# from os.path import dirname,join,abspath



class BipedalModel():
    def __init__(self,urdf_filename,LF_NAME,RF_NAME,q_ID_marker):
        self.urdf_filename = urdf_filename
        self.model = pin.buildModelFromUrdf(self.urdf_filename)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.LF_NAME = LF_NAME
        self.RF_NAME = RF_NAME
        self.q_ID_marker = q_ID_marker

        print("model name: " + self.model.name)
        self.ik_config()
        self.success_bool = np.zeros(2, dtype=bool)

        # set the end link of the robot (either joint or end effector)
        self.LF_ID = self.model.getFrameId(self.LF_NAME)
        self.RF_ID = self.model.getFrameId(self.RF_NAME)

        # get joint limits from urdf file
        self.lower_limit = np.zeros(self.model.njoints-1)
        self.upper_limit = np.zeros(self.model.njoints-1)
        for joint_id in range(self.model.njoints):
            self.lower_limit[joint_id-1] = self.model.lowerPositionLimit[joint_id-1]
            self.upper_limit[joint_id-1] = self.model.upperPositionLimit[joint_id-1]
        # a = self.model.getJointId('left_ankle_joint')#  l: 1,7   r: 11,
        # print(a)


    def ik_config(self,eps_foot = 1e-2,IT_MAX = 1000,DT = 5e-3,damp = 1e-12):
        # set different final error 
        self.eps_foot = eps_foot
        self.IT_MAX = IT_MAX
        self.DT = DT
        self.damp = damp


    def fk(self):
        # a = self.model.getJointId('right_1_joint')#  l: 1,7   r: 8,14
        # print(a)
        pin.forwardKinematics(self.model, self.data, self.q)
        # update forward kinematics of end effector frame
        pin.updateFramePlacement(self.model, self.data,self.LF_ID)
        pin.updateFramePlacement(self.model, self.data,self.RF_ID)
        print("\nJoint placements:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name==self.LF_NAME or name==self.RF_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f}".format(name, *oMi.translation.T.flat)))
        print("\nJoint rotation:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name==self.LF_NAME or name==self.RF_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f}".format(name, *oMi.rotation.flat))) 



    def ik(self,q,T_LF,T_RF,R_LF,R_RF):
        self.q = q
        # print joint angles (rad)
        # print("q: %s" % q.T)
        i = 1  
        while True:
            # update forward kinematics of all joints
            pin.forwardKinematics(self.model, self.data, self.q)
            # update forward kinematics of end effector frame
            pin.updateFramePlacement(self.model, self.data,self.LF_ID)
            pin.updateFramePlacement(self.model, self.data,self.RF_ID)

            # convert the target placements (position and rotation) into the end effector frame
            fMd_lf = self.data.oMf[self.LF_ID].actInv(pin.SE3(R_LF,T_LF))
            fMd_rf = self.data.oMf[self.RF_ID].actInv(pin.SE3(R_RF,T_RF))

            # get the error vetor to target placements
            err_lf = pin.log(fMd_lf).vector 
            err_rf = pin.log(fMd_rf).vector 

            if self.success_bool[0] and self.success_bool[1]:
                # print("q: %s" % self.q.T)
                # print("Convergence achieved!")
                break
            if i >= self.IT_MAX:
                # print(self.success_bool)
                # print("q: %s" % self.q.T)
                # print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")                 
                break

            if norm(err_lf) > self.eps_foot:
                J3 = pin.computeFrameJacobian(self.model, self.data, self.q, self.LF_ID) # in joint frame
                J3 = -np.dot(pin.Jlog6(fMd_lf.inverse()), J3)
                v3 = -J3.T.dot(solve(J3.dot(J3.T) + self.damp * np.eye(6), err_lf))
                q3 = pin.integrate(self.model, self.q, v3 * self.DT)
                self.q[self.q_ID_marker[0]:self.q_ID_marker[1]]=q3[self.q_ID_marker[0]:self.q_ID_marker[1]]
            else:
                self.success_bool[0] = True

            if norm(err_rf) > self.eps_foot:
                J4 = pin.computeFrameJacobian(self.model, self.data, self.q, self.RF_ID) # in joint frame
                J4 = -np.dot(pin.Jlog6(fMd_rf.inverse()), J4)
                v4 = -J4.T.dot(solve(J4.dot(J4.T) + self.damp * np.eye(6), err_rf))
                q4 = pin.integrate(self.model, self.q, v4 * self.DT)
                self.q[self.q_ID_marker[2]:self.q_ID_marker[3]]=q4[self.q_ID_marker[2]:self.q_ID_marker[3]]
            else:
                self.success_bool[1] = True

            # limit the joint angles 
            self.q = np.clip(self.q,self.lower_limit,self.upper_limit)
            i += 1

        return self.q

    def print_results(self):
        print(self.success_bool)
        print("q: %s" % self.q.T)
        pin.forwardKinematics(self.model, self.data, self.q)
        # update forward kinematics of end effector frame
        pin.updateFramePlacement(self.model, self.data,self.LF_ID)
        pin.updateFramePlacement(self.model, self.data,self.RF_ID)
        print("\nJoint placements:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name == name==self.LF_NAME or name==self.RF_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f}".format(name, *oMi.translation.T.flat)))
        print("\nJoint rotation:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name == self.LF_NAME or name == self.RF_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f}".format(name, *oMi.rotation.flat))) 


R_lf = np.eye(3)
R_rf = np.eye(3)
T_lf = np.array([0.0100, 0.0850 , -0.7063])
T_rf = np.array([0.0100, -0.0850, -0.7063])
q_ID_marker = np.array([0,5,5,10])
wow = BipedalModel("../../../resources/robots/wow/urdf/wow_narm.urdf","left_ankle_joint","right_ankle_joint",q_ID_marker)
wow.fk()
wow.ik(np.zeros(10),T_lf,T_rf,R_lf,R_rf)
wow.print_results()
# q = wow.fk()








# R_lh = np.eye(3)
# R_rh = np.eye(3)
# R_lf = np.eye(3)
# # R_lf = np.array([[0.9104 , -0.0566,  0.4099], [0.3550 , 0.6158 ,-0.7034],[0.2126,  0.7859 , 0.5807]])
# R_rf = np.eye(3)
# # R_lf = np.array([[1,  0,  0],[0,  0,  1],[0,  0,  1]])
# # R_rf = np.array([[1,  0,  0],[0,  0,  1],[0,  0,  1]])
# # T_lh = np.array([0.0751,  0.1645,  0.0765])
# # T_rh = np.array([0.0751, -0.1645,  0.0765])
# T_lh = np.array([0.094,  0.1621,  0.0849])
# T_rh = np.array([0.094, -0.1621,  0.0849])
# T_lf = np.array([0.00, 0.1179, -0.7197])

# T_lf = np.array([0.00,  0.1179, -0.5])
# T_rf = np.array([0.00, -0.1179, -0.7197])
# q_ID_marker = np.array([13,18,18,23,0,6,6,12])
# g1 = HumanoidModel("../../../resources/robots/g1/urdf/g1.urdf",'left_elbow_roll_joint','right_elbow_roll_joint','left_ankle_roll_joint','right_ankle_roll_joint',q_ID_marker)
# q = g1.ik(np.zeros(23),T_lh,T_rh,T_lf,T_rf,R_lh,R_rh,R_lf,R_rf)
# print(q)



