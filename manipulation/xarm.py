import os
import numpy as np
import pybullet as p
from manipulation.robot import Robot
from collections import namedtuple

class Xarm(Robot):
    def __init__(self, controllable_joints='right', slider=False):
        right_arm_joint_indices = [1, 2, 3, 4, 5, 6, 7] # Controllable arm joints
        right_end_effector = 16 # Used to get the pose of the end effector
        right_gripper_indices = [11, 14] # left and right finger joints
        right_hand = 9 # TODO: check this
        self.finger_fully_open_joint_angle = 0.9 ### NOTE: this is reversed. in xarm, 0 is fully open, and 0.85 is fully closed
        self.finger_fully_close_joint_angle = 0.0
                
        super(Xarm, self).__init__(controllable_joints, right_arm_joint_indices, right_end_effector, right_gripper_indices)
        self.right_hand = right_hand

    def init(self, directory, id, np_random, fixed_base=False, use_suction=True, debug=False):
        self.body = p.loadURDF(os.path.join(directory, "xarm_description/urdf/xarm_7dof.urdf"), useFixedBase=fixed_base, basePosition=[0, 0, 0], physicsClientId=id)

        mimic_parent_name = 'drive_joint'
        mimic_children_names = [
            'right_outer_knuckle_joint',
            'left_inner_knuckle_joint',
            'right_inner_knuckle_joint',
            'left_finger_joint',
            'right_finger_joint'
        ]
        
        mimic_left_parent_name = 'drive_joint'
        mimic_left_children_names = [
            'left_inner_knuckle_joint',
            'left_finger_joint',
        ]
        
        mimic_right_parent_name = 'right_outer_knuckle_joint'
        mimic_right_children_names = [
            'right_inner_knuckle_joint',
            'right_finger_joint',
        ]

        numJoints = p.getNumJoints(self.body)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type', "link_name", 'damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.body, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            link_name = info[12].decode("utf-8")
            controllable = (jointType != p.JOINT_FIXED)
            info = jointInfo(jointID,jointName,jointType,link_name,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            # print(info)
            joints.append(info)
        
        self.joints = joints
        self.mimic_parent_id = [joint.id for joint in joints if joint.name == mimic_parent_name][0]
        self.mimic_children_ids = [joint.id for joint in joints if joint.name in mimic_children_names]
        
        self.mimic_left_parent_id = [joint.id for joint in joints if joint.name == mimic_left_parent_name][0]
        self.mimic_left_children_ids = [joint.id for joint in joints if joint.name in mimic_left_children_names]
        self.mimic_right_parent_id = [joint.id for joint in joints if joint.name == mimic_right_parent_name][0]
        self.mimic_right_children_ids = [joint.id for joint in joints if joint.name in mimic_right_children_names]

        if debug:
            for i in range(p.getNumJoints(self.body, physicsClientId=id)):
                print(p.getJointInfo(self.body, i, physicsClientId=id))
                link_name = p.getJointInfo(self.body, i, physicsClientId=id)[12].decode('utf-8')
                joint_limits = p.getJointInfo(self.body, i, physicsClientId=id)[8:10]
                print("link_name: ", link_name)
        
        super(Xarm, self).init(self.body, id, np_random)
        
    def set_gripper_open_position(self, indices, positions, set_instantly=False, force=500, debug=False):
        open_length = self.finger_fully_open_joint_angle - positions[0]
        
        if not set_instantly:

            # p.setJointMotorControl2(self.body, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_length,
            #                     force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity, physicsClientId=self.id)
            # gripper_joint_positions = p.getJointState(self.body, self.mimic_parent_id, physicsClientId=self.id)[0] 
            # if debug:
            #     import pdb; pdb.set_trace()
            # for joint_id in self.mimic_children_ids:
            #     p.setJointMotorControl2(self.body, joint_id, p.POSITION_CONTROL, targetPosition=gripper_joint_positions,
            #                             force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity, physicsClientId=self.id)
            
            p.setJointMotorControl2(self.body, self.mimic_left_parent_id, p.POSITION_CONTROL, targetPosition=open_length,
                                force=force, maxVelocity=self.joints[self.mimic_left_parent_id].maxVelocity, physicsClientId=self.id)
            p.setJointMotorControl2(self.body, self.mimic_right_parent_id, p.POSITION_CONTROL, targetPosition=open_length,
                                force=force, maxVelocity=self.joints[self.mimic_right_parent_id].maxVelocity, physicsClientId=self.id)
            
            gripper_left_joint_positions = p.getJointState(self.body, self.mimic_left_parent_id, physicsClientId=self.id)[0]
            gripper_right_joint_positions = p.getJointState(self.body, self.mimic_right_parent_id, physicsClientId=self.id)[0]
            
            for joint_id in self.mimic_left_children_ids:
                p.setJointMotorControl2(self.body, joint_id, p.POSITION_CONTROL, targetPosition=gripper_left_joint_positions,
                                    force=force, maxVelocity=self.joints[joint_id].maxVelocity, physicsClientId=self.id)
            for joint_id in self.mimic_right_children_ids:
                p.setJointMotorControl2(self.body, joint_id, p.POSITION_CONTROL, targetPosition=gripper_right_joint_positions,
                                    force=force, maxVelocity=self.joints[joint_id].maxVelocity, physicsClientId=self.id)
            
        if set_instantly:
            self.set_joint_angles([self.mimic_parent_id] + self.mimic_children_ids, [open_length for _ in range(6)], use_limits=True)
        
        
        # p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=positions, 
        #         physicsClientId=self.id)
        
        # if set_instantly:
        #     self.set_joint_angles(indices, positions, use_limits=True)
        
