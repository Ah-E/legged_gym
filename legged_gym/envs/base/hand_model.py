from __future__ import print_function
import numpy as np
from numpy.linalg import norm, solve
# pinocchio
import pinocchio as pin
from sys import argv
# from os.path import dirname,join,abspath



class HandModel():
    def __init__(self,urdf_filename,LH_NAME,RH_NAME,q_ID_marker):
        self.urdf_filename = urdf_filename
        self.model = pin.buildModelFromUrdf(self.urdf_filename)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        # q = pin.randomConfiguration(self.model)
        self.LH_NAME = LH_NAME
        self.RH_NAME = RH_NAME
        self.q_ID_marker = q_ID_marker

        print("model name: " + self.model.name)
        self.ik_config()
        self.success_bool = np.zeros(2, dtype=bool)

        # set the end link of the robot (either joint or end effector)
        self.LH_ID = self.model.getFrameId(self.LH_NAME)# frame id in pinocchio
        self.RH_ID = self.model.getFrameId(self.RH_NAME)

        # get joint limits from urdf file
        self.lower_limit = np.zeros(self.model.njoints-1)
        self.upper_limit = np.zeros(self.model.njoints-1)
        for joint_id in range(self.model.njoints):
            self.lower_limit[joint_id-1] = self.model.lowerPositionLimit[joint_id-1]
            self.upper_limit[joint_id-1] = self.model.upperPositionLimit[joint_id-1]
        a = self.model.getJointId('right_wrist_roll_joint')#  l: 1,7   r: 8,14
        print(a)

    def ik_config(self,eps_hand = 1e-3,IT_MAX = 1000,DT = 5e-3,damp = 1e-12):
        # set different final error 
        self.eps_hand = eps_hand
        self.IT_MAX = IT_MAX
        self.DT = DT
        self.damp = damp

    def fk(self):
        self.q = pin.neutral(self.model)
        # self.q = pin.randomConfiguration(self.model)
        pin.forwardKinematics(self.model, self.data, self.q)
        # update forward kinematics of end effector frame
        pin.updateFramePlacement(self.model, self.data,self.LH_ID)
        pin.updateFramePlacement(self.model, self.data,self.RH_ID)
        print("\nJoint placements:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name==self.LH_NAME or name==self.RH_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f}".format(name, *oMi.translation.T.flat)))
        print("\nJoint rotation:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name==self.LH_NAME or name==self.RH_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f}".format(name, *oMi.rotation.flat))) 


    def ik(self,q,T_LH,T_RH,R_LH,R_RH):
        self.q = q
        # print joint angles (rad)
        # print("q: %s" % q.T)
        i = 1  
        while True:
            # update forward kinematics of all joints
            pin.forwardKinematics(self.model, self.data, self.q)
            # update forward kinematics of end effector frame
            pin.updateFramePlacement(self.model, self.data, self.LH_ID)
            pin.updateFramePlacement(self.model, self.data, self.RH_ID)

            # convert the target placements (position and rotation) into the end effector frame
            fMd_lh = self.data.oMf[self.LH_ID].actInv(pin.SE3(R_LH,T_LH))
            fMd_rh = self.data.oMf[self.RH_ID].actInv(pin.SE3(R_RH,T_RH))

            # get the error vetor to target placements
            err_lh = pin.log(fMd_lh).vector 
            err_rh = pin.log(fMd_rh).vector 
            # if (i % 100==1):
            #     print("err_lh")
            #     print(err_lh)
            #     print("err_rh")
            #     print(err_rh)


            if self.success_bool[0] and self.success_bool[1]:
                print(self.success_bool)
                # print("q: %s" % self.q.T)
                # print("Convergence achieved!")
                break
            if i >= self.IT_MAX:
                print(self.success_bool)
                # print("q: %s" % self.q.T)
                # print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")                 
                break
            if norm(err_lh) > self.eps_hand:
                J1 = pin.computeFrameJacobian(self.model, self.data, self.q, self.LH_ID) # in joint frame
                J1 = -np.dot(pin.Jlog6(fMd_lh.inverse()), J1)
                v1 = -J1.T.dot(solve(J1.dot(J1.T) + self.damp * np.eye(6), err_lh))
                # if (i%100==1):
                #     print("v1: ",v1)
                #     print("J1: ",J1)

                q1 = pin.integrate(self.model, self.q, v1 * self.DT)
                self.q[self.q_ID_marker[0]:self.q_ID_marker[1]]=q1[self.q_ID_marker[0]:self.q_ID_marker[1]]
            else:
                self.success_bool[0] = True

            if norm(err_rh) > self.eps_hand:
                J2 = pin.computeFrameJacobian(self.model, self.data, self.q, self.RH_ID)
                J2 = -np.dot(pin.Jlog6(fMd_rh.inverse()), J2)
                v2 = -J2.T.dot(solve(J2.dot(J2.T) + self.damp * np.eye(6), err_rh))
                q2 = pin.integrate(self.model, self.q, v2 * self.DT)
                self.q[self.q_ID_marker[2]:self.q_ID_marker[3]]=q2[self.q_ID_marker[2]:self.q_ID_marker[3]]
            else:
                self.success_bool[1] = True


                    # print(err_rh)
            # limit the joint angles 
            self.q = np.clip(self.q,self.lower_limit,self.upper_limit)
            i += 1
        # print('J1')
        # print(J1)


        return self.q

    def print_results(self):
        # update forward kinematics of all joints
        pin.forwardKinematics(self.model, self.data, self.q)
        # update forward kinematics of end effector frame
        pin.updateFramePlacement(self.model, self.data,self.LH_ID)
        pin.updateFramePlacement(self.model, self.data,self.RH_ID)
        print(self.success_bool)
        print("q: %s" % self.q.T)
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name==self.LH_NAME or name==self.RH_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f}".format(name, *oMi.translation.T.flat)))
        print("\nJoint rotation:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name==self.LH_NAME or name==self.RH_NAME:
                print(("{:<24} : {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f}".format(name, *oMi.rotation.flat))) 



R_lh = np.eye(3)
R_rh = np.eye(3)
T_lh = np.array([0.,0.1579, -0.2236]) + 1e-12 * np.ones(3)
print(T_lh)
T_rh = np.array([0.00, -0.1579, -0.2238]) + 1e-12 * np.ones(3)
q_ID_marker = np.array([0,7,7,14])
# franka = HandModel("../../../resources/robots/franka/urdf/up_body_wo_dummy.urdf","left_7_joint","right_7_joint",q_ID_marker)
franka = HandModel("../../../resources/robots/wow_body/urdf/wow_body.urdf","left_wrist_roll_joint","right_wrist_roll_joint",q_ID_marker)


franka.fk()
franka.ik(np.zeros(38),T_lh,T_rh,R_lh,R_rh)
# print(q)
franka.print_results()
# franka.print_results()
# print(q)



