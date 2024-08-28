import numpy as np
import pybullet as p
from manipulation.panda import Panda
import os.path as osp
import time


class FilterRobotPCDInterface():
    def __init__(self, gui=False):
        super().__init__()
        self.gui = gui
        if self.gui:
            try:
                self.id = p.connect(p.GUI)
            except:
                self.id = p.connect(p.DIRECT)
        else:
            self.id = p.connect(p.DIRECT)
        self.gravity = -9.81
        p.setTimeStep(1/240, physicsClientId=self.id)
        p.resetSimulation(physicsClientId=self.id)
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.asset_dir = osp.join("/media/ziyu/Elements/workspace/RoboGen-sim2real/manipulation", "assets/")
        
        self.robot_base_pos = self.load_robot()

        self.initial_robot_link_pcs = []
        for i in range(12):
            link_pc = np.load(f"/media/ziyu/Elements/workspace/RoboGen-sim2real/local_exps/filter_robot_pointcloud/robot_pointcloud/link_{i}.npy")
            self.initial_robot_link_pcs.append(link_pc)

    def load_robot(self, robot_initial_joint_angles=None):
      
        # Create robot
        self.robot = Panda(slider=False)
        self.robot.init(self.asset_dir, self.id, 0, fixed_base=True, use_suction=False)
        self.agents = [self.robot]
        self.suction_id = self.robot.right_gripper_indices[0]

        # Set robot base position & orientation, and joint angles
        robot_base_pos = [0, 0, 0]
        robot_base_orient = [0, 0, 0, 1]
        self.robot_base_orient = robot_base_orient
        self.robot.set_base_pos_orient(robot_base_pos, robot_base_orient)
        init_joint_angles = self.get_robot_init_joint_angles(robot_initial_joint_angles)

        self.robot.set_joint_angles(self.robot.right_arm_joint_indices, init_joint_angles)    

        self.robot.set_gravity(0, 0, 0)
        
        return robot_base_pos    

    def get_robot_init_joint_angles(self, robot_init_joint_angles=None):
        if robot_init_joint_angles is None:
            init_joint_angles = [0 for _ in range(len(self.robot.right_arm_joint_indices))]

            init_joint_angles[3] = -0.4
            init_joint_angles[5] = 0.4
            return init_joint_angles  
        return robot_init_joint_angles

    def get_robot_pc(self, robot_id, physicsClientId=0):
        robot_link_pcs = []
        for link in range(12):
            res = p.getLinkState(robot_id, link, physicsClientId=physicsClientId)
            pos = res[0]
            orient = res[1]

            T_body_to_world = np.eye(4)
            T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
            T_body_to_world[:3, 3] = pos
            point_cloud = self.initial_robot_link_pcs[link].reshape(-1, 3)
            point_cloud_homogeneous = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1))], axis=1)
            transformed_pc_homogeneous = (T_body_to_world @ point_cloud_homogeneous.T).T
            transformed_pc = transformed_pc_homogeneous[:, :3]
            robot_link_pcs.append(transformed_pc)

        robot_pc = np.concatenate(robot_link_pcs, axis=0)
        return robot_pc


    def filter_robot_pc(self, whole_pcd, input_joint_values, filter_radius=0.01):
        # set robot joint angles to input_joint_values
        self.robot.set_joint_angles(self.robot.right_arm_joint_indices, input_joint_values)

        # get robot point cloud
        robot_pc = self.get_robot_pc(self.robot.body, physicsClientId=self.id)

        # filter robot point cloud from whole_pcd
        # whole_pcd: N x 3
        # robot_pc: M x 3
        distance = np.linalg.norm(whole_pcd[:, None] - robot_pc, axis=-1)
        min_distance = np.min(distance, axis=1)
        mask = min_distance > filter_radius
        filtered_pcd = whole_pcd[mask]

        return filtered_pcd

if __name__ == "__main__":
    interface = FilterRobotPCDInterface(gui=True)
    time1 = time.time()
    inpuit_joint_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    interface.robot.set_joint_angles(interface.robot.right_arm_joint_indices, inpuit_joint_values)
    robot_pc = interface.get_robot_pc(interface.robot.body, physicsClientId=interface.id)
    time2 = time.time()
    print("==========time======", time2 - time1)
    import pdb; pdb.set_trace()
    p.addUserDebugPoints(robot_pc, [[1, 0, 0] for _ in range(len(robot_pc))], physicsClientId=interface.id)


    

