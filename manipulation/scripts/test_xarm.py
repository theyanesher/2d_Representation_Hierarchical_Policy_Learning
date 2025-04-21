import pybullet as p
import pybullet_data as pd
import time
from collections import namedtuple
import math
import os

def move_gripper(open_length):
    # open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
    # Control the mimic gripper joint(s)
    p.setJointMotorControl2(xarm, mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_length,
                            force=joints[mimic_parent_id].maxForce, maxVelocity=joints[mimic_parent_id].maxVelocity)
    gripper_joint_positions = p.getJointState(xarm, mimic_parent_id)[0] 
    for joint_id in mimic_children_ids:
        p.setJointMotorControl2(xarm, joint_id, p.POSITION_CONTROL, targetPosition=gripper_joint_positions,
                                force=joints[joint_id].maxForce, maxVelocity=joints[joint_id].maxVelocity)

p.connect(p.GUI)
p.setGravity(0, 0, 0)
p.setTimeStep(1/240.)

asset_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/manipulation/assets"
# planeId = p.loadURDF(os.path.join(asset_dir, "plane", "plane.urdf"))
xarm = p.loadURDF(
    "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/manipulation/assets/xarm_description/urdf/xarm_7dof.urdf", 
    useFixedBase=True, 
)

mimic_parent_name = 'drive_joint'
mimic_children_names = [
    'right_outer_knuckle_joint',
    'left_inner_knuckle_joint',
    'right_inner_knuckle_joint',
    'left_finger_joint',
    'right_finger_joint'
]

numJoints = p.getNumJoints(xarm)
jointInfo = namedtuple('jointInfo', 
    ['id','name','type', "link_name", 'damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
joints = []
for i in range(numJoints):
    info = p.getJointInfo(xarm, i)
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
    


mimic_parent_id = [joint.id for joint in joints if joint.name == mimic_parent_name][0]
mimic_children_ids = [joint.id for joint in joints if joint.name in mimic_children_names]

for id in mimic_children_ids:
    dynamics_info = p.getDynamicsInfo(xarm, id)
    print("dynamics_info: ", dynamics_info)

# exit()
# joint_vals = (-1.216224516810089, 1.0560968395448116, 1.126902391390439,
#         1.9378128822622178, 0.19481367902305902, -0.42124895378023686, 1.188535735786957)

# for i in range(1, 1+7):
#     p.resetJointState(xarm, i, targetValue=joint_vals[i-1], targetVelocity=0)

paramId = p.addUserDebugParameter("drive_joint", 0.0425, 0.9, 0.0425)

skip_cam_frames = 10  


# timesteps = 40  
# for i in range(timesteps):
#     angle = 1.0
#     move_gripper(angle)
#     p.stepSimulation()
#     new_angle = p.getJointState(xarm, mimic_parent_id)[0]
#     print(f"angle: {angle}, new_angle: {new_angle}")

left_finger_pos = p.getLinkState(xarm, 11)[0]
right_finger_pos = p.getLinkState(xarm, 14)[0]
tcp_pos = p.getLinkState(xarm, 16)[0]
hand_pos = p.getLinkState(xarm, 9)[0]

# p.addUserDebugPoints([left_finger_pos, right_finger_pos, tcp_pos, hand_pos], [[0, 0, 1] for _ in range(4)], 60)



# tcp_pos = p.getLinkState(xarm, 14)[0]
while (1):
    targetPos = p.readUserDebugParameter(paramId)
    # print(targetPos)
    move_gripper(targetPos)
    for _ in range(10):
        p.stepSimulation()
    now_tcp_pos = p.getLinkState(xarm, 14)[0]
    print(f"original: {tcp_pos}, now: {now_tcp_pos}")

# print("tcp_pos: ", tcp_pos)
# move_gripper(0.85)
# for _ in range(10000):
#     p.stepSimulation()
# tcp_pos_open = p.getLinkState(xarm, 16)[0]
# print("tcp_pos_open: ", tcp_pos_open)
	
