import pybullet as p
import numpy as np
from bullet_sim.utils import take_round_images_around_object, get_pc
import open3d as o3d
from scipy.spatial.transform import Rotation as R

### NOTE: panda finger upper joint limit: 0.04
### NOTE: panda y axis is parallel to the gripper, see slack image.

def voxelize_pc(pc, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    try:
        voxelized_pcd = pcd.voxel_down_sample(voxel_size)
    except RuntimeError:
        return None
    voxelized_pc = np.asarray(voxelized_pcd.points)
    return voxelized_pc

def rotation_matrix_x(theta):
    """Return a 3x3 rotation matrix for a rotation around the x-axis by angle theta."""
    return R.from_matrix(np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ]))

def set_eef_to_pose(simulator, target_pos, target_quat, instant=True):
    robot = simulator.robot
    joint = robot.right_end_effector 
    ik_indices = robot.right_arm_ik_indices 
    cur_pos, cur_orient = robot.get_pos_orient(robot.right_end_effector)

    if instant:
        iter = 0
        pos_error = np.linalg.norm(cur_pos - target_pos) 
        orient_error = np.linalg.norm(cur_orient - target_quat)
        while (pos_error > 0.01 or orient_error > 0.01) and iter < 5:
            robot_joint_angles = simulator.robot.ik(joint, target_pos, target_quat, ik_indices, max_iterations=200, use_current_as_rest=True)
            robot.set_joint_angles(ik_indices, robot_joint_angles, use_limits=True, velocities=0)
            p.stepSimulation(physicsClientId=simulator.id)
            cur_pos, cur_orient = robot.get_pos_orient(robot.right_end_effector)
            pos_error = np.linalg.norm(cur_pos - target_pos) 
            orient_error = np.linalg.norm(cur_orient - target_quat)
            print(f"iter {iter} pos error: {pos_error}, orient error: {orient_error}")
            iter += 1
    else:
        rgbs = []
        iter = 0
        while (np.linalg.norm(cur_pos - target_pos) > 0.01 or np.linalg.norm(cur_orient - target_quat) > 0.01) and iter < 10:
            for _ in range(20):
                robot_joint_angles = simulator.robot.ik(joint, target_pos, target_quat, ik_indices, max_iterations=200, use_current_as_rest=True)
                robot.control(robot.controllable_joint_indices, robot_joint_angles, robot.motor_gains, robot.motor_forces)
                p.stepSimulation(physicsClientId=simulator.id)
                rgb, _ = simulator.render()
                rgbs.append(rgb)
                cur_pos, cur_orient = robot.get_pos_orient(robot.right_end_effector)
            iter += 1

        return rgbs
    
def align_gripper_z_with_normal(normal):
    n_WS = normal
    Gz = n_WS  # gripper z axis aligns with normal # TODO: check the object axis of the franka gripper
    # make orthonormal y axis, aligned with world down
    # y = np.array([0.0, 0.0, -1.0])
    # or, make it horizontal
    y = np.array([0.0, -1, 0])
    
    # if np.abs(np.dot(z, Gy)) < 1e-6: # NOTE: I am not sure if this is correct, if normal is pointing straight down, then the dot should be almost 1 right?
    #     # normal was pointing straight down.  reject this sample.
    #     return np.inf, None, None

    Gy = y - np.dot(y, Gz) * Gz
    Gx = np.cross(Gy, Gz)
    # R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    return R_WG

def align_gripper_x_with_normal(normal):
    n_WS = normal
    Gx = n_WS  # gripper z axis aligns with normal # TODO: check the object axis of the franka gripper
    # make orthonormal y axis, aligned with world down
    # y = np.array([0.0, 0.0, -1.0])
    # or, make it horizontal
    y = np.array([0.0, -1, 0])
    
    # if np.abs(np.dot(z, Gy)) < 1e-6: # NOTE: I am not sure if this is correct, if normal is pointing straight down, then the dot should be almost 1 right?
    #     # normal was pointing straight down.  reject this sample.
    #     return np.inf, None, None

    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    # R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    return R_WG

def GenerateAntipodalGraspCandidate(
    simulator,
    pc_points,
    pc_normals,
    object_id,
):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """

    index = np.random.randint(0, pc_points.shape[0])

    # Use S for sample point/frame.
    p_WS = pc_points[index]
    n_WS = pc_normals[index]

    print("sampled point position: ", p_WS)
    print()

    assert np.isclose(
        np.linalg.norm(n_WS), 1.0
    ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gy = n_WS  #align franka y axis with the normal
    # make orthonormal z axis, aligned with world down
    z = np.array([0.0, 0.0, -1.0])
    # if np.abs(np.dot(z, Gy)) < 1e-6: # NOTE: I am not sure if this is correct, if normal is pointing straight down, then the dot should be almost 1 right?
    #     # normal was pointing straight down.  reject this sample.
    #     return np.inf, None, None

    Gz = z - np.dot(z, Gy) * Gy
    Gx = np.cross(Gy, Gz)
    # R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0, 0, 0] # TODO: check the length and coordinate of the franka gripper

    # Try orientations from the center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # Rotate the object in the hand by a random rotation (around the normal).
        # R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))
        R_WG2 = R_WG * rotation_matrix_x(theta)
        
        # Use G for gripper frame.
        # import pdb; pdb.set_trace()
        p_SG_W = - (R_WG2.as_matrix() @ np.array(p_GS_G).reshape(3, 1)).flatten()
        p_WG = p_WS.flatten()  + p_SG_W # in our case p_SG_W is always 0

        # X_G = RigidTransform(R_WG2, p_WG)
        # plant.SetFreeBodyPose(plant_context, wsg, X_G)

        # use ik to set the robot eef to the target pose
        quat_WG2 = R_WG2.as_quat() # scipy quaternion is [x y z w], e.g., [0 0 0 1] is identity, which aligns with what pybullet is using. 
        print("set eef to sampled pose: ")
        # import pdb; pdb.set_trace()
        old_joint_angles = simulator.robot.get_joint_angles()
        object_pos, object_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=simulator.id)
        set_eef_to_pose(simulator, p_WG, quat_WG2, instant=True)


        print("compute cost")
        cost = GraspCandidateCost(simulator, object_id, pc_points, pc_normals, adjust_X_G=True)
        final_image = simulator.render()[0]

        # recover to old joint angles & old object pose
        simulator.robot.set_joint_angles(simulator.robot.all_joint_indices, old_joint_angles, use_limits=True)
        p.resetBasePositionAndOrientation(object_id, object_pos, object_orient, physicsClientId=simulator.id)
        p.stepSimulation(physicsClientId=simulator.id)
        
        if np.isfinite(cost):
            return cost, p_WG, quat_WG2, final_image

    return np.inf, None, None, final_image


def GraspCandidateCost(
    simulator,
    object_id,
    pc_points,
    pc_normals,
    adjust_X_G=False,
):
    """
    Returns:
        cost: The grasp cost

    If adjust_X_G is True, then it also updates the gripper pose in the plant
    context.
    """
    robot = simulator.robot

    # Transform cloud into gripper frame
    # X_GW = X_G.inverse()
    # p_GC = X_GW @ cloud.xyzs()

    ### my attempt with GPT's help
    ### Transform each point in the point cloud to the gripper frame
    cur_pos, cur_orient = robot.get_pos_orient(robot.right_end_effector)
    T_body_to_world = np.eye(4)
    T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(cur_orient)).reshape(3, 3)
    T_body_to_world[:3, 3] = cur_pos

    X_GW = p.invertTransform(cur_pos, cur_orient)
    translation = np.array(X_GW[0])
    rotation = np.array(p.getMatrixFromQuaternion(X_GW[1])).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation ### this is the transformation from world frame to gripper frame

    pc_homogeneous = np.hstack((pc_points, np.ones((pc_points.shape[0], 1))))  # Convert to homogeneous coordinates Nx4
    pc_transformed_homogeneous = T @ pc_homogeneous.T # 4x4 @ 4xN = 4xN
    p_GC = pc_transformed_homogeneous[:3, :] # 3xN

    ### Crop to a region inside of the finger box.
    crop_min = [-0.01, -0.04, -0.01] # TODO: check the length and coordinate of the franka gripper
    crop_max = [0.01, 0.04, 0.01]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )

    if adjust_X_G and np.sum(indices) > 0:
        print("adjusting pose to center at the point cloud!")
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        gripper_new_pos = T_body_to_world @ np.array([p_Gcenter_x, 0, 0, 1]).reshape(4, 1)
        gripper_new_pos = gripper_new_pos[:3, 0]
        set_eef_to_pose(simulator, gripper_new_pos, cur_orient)

    ### Check collisions between the gripper and the point cloud
    # margin = 0.0  # must be smaller than the margin used in the point cloud preprocessing.
    # for i in range(cloud.size()):
    #     distances = query_object.ComputeSignedDistanceToPoint(
    #         cloud.xyz(i), threshold=margin
    #     )
    #     if distances:
    #         cost = np.inf
    #         if verbose:
    #             print("Gripper is colliding with the point cloud!\n")
    #             print(f"cost: {cost}")
    #         return cost

    contact_points = p.getClosestPoints(simulator.robot.body, object_id, 0, physicsClientId=simulator.id)
    if len(contact_points) > 0:
        print("there is already contact! cost is inf")
        cost = np.inf
        return cost

    ### cost is how well the normal aligns with the gripper x axis
    # n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])
    cost = 0
    n_GC = rotation @ (pc_normals.T)[:, indices] # 3x3 @ 3xN = 3xN
    cost -= np.sum(n_GC[1, :] ** 2)

    ### Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    # cost += 20.0 * X_G.rotation().matrix()[2, 1]
    cost += 20.0 * rotation[2, 1]

    print(f"cost: {cost}")
    # print(f"normal terms: {n_GC[0,:]**2}")
    return cost

def get_pc_and_normal(simulator, object_name):
    camera_width=640
    camera_height=480
    rgbs, depths, view_camera_matrices, project_camera_matrices = \
        take_round_images_around_object(simulator, object_name, 
                                        return_camera_matrices=True, camera_height=camera_height, camera_width=camera_width, 
                                        only_object=True)
    pcs = []
    for depth, view_matrix, project_matrix in zip(depths, view_camera_matrices, project_camera_matrices):
        pc = get_pc(project_matrix, view_matrix, depth, camera_width, camera_height, mask_infinite=True)
        pcs.append(pc)


    pc = np.concatenate(pcs, axis=0)
    pc = voxelize_pc(pc, voxel_size=0.0005) 

    ### get normals of the point cloud
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)

    return pc, normals

def grasp(simulator, object_name, sample_num=100):
    ### set gripper to be open
    robot = simulator.robot
    robot.set_gripper_open_position(robot.right_gripper_indices, [0.1] * len(robot.right_gripper_indices), set_instantly=True)

    ### remove all other objects in the scene
    prev_rgbas = []
    object_id = simulator.urdf_ids[object_name]
    print("object position: ", p.getBasePositionAndOrientation(object_id, physicsClientId=simulator.id)[0])
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=simulator.id)[0][14:18]
                prev_rgbas.append(prev_rgba)
                p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### record old camera matrices
    old_view_matrix, old_project_matrix = env.view_matrix, env.projection_matrix

    ### take round view images of the objects, convert to point clouds, merge all point clouds, voxelized point cloud
    pc, normals = get_pc_and_normal(simulator, object_name)

    from matplotlib import pyplot as plt
    ax = plt.axes(projection='3d')
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2])
    plt.savefig("bullet_sim/data/grasp_pc.png")

    ### reset to old camera pose
    env.view_matrix = old_view_matrix
    env.projection_matrix = old_project_matrix

    ### set color back
    cnt = 0
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=simulator.id)
                cnt += 1

    ### sample grasping pose (from Russ Tedrake's note)
    costs = []
    p_WGs = []
    quat_WGs = []
    sampled_rgbs = []
    for idx in range(sample_num):
        print("sampling grasping pose: ", idx)
        cost, p_WG, quat_WG, final_image = GenerateAntipodalGraspCandidate(simulator, pc, normals, object_id)
        costs.append(cost)
        p_WGs.append(p_WG)
        quat_WGs.append(quat_WG)
        sampled_rgbs.append(final_image)

    best_idx = np.argmin(costs)
    best_cost = costs[best_idx]
    best_p_WG = p_WGs[best_idx]
    best_quat_WG = quat_WGs[best_idx]

    ### execute grasping pose with the highest score
    rgbs = set_eef_to_pose(simulator, best_p_WG, best_quat_WG, instant=False)

    ### close gripper 
    robot = simulator.robot
    grasp_rgbs = []
    for _ in range(20):
        robot.set_gripper_open_position(robot.right_gripper_indices, [0] * len(robot.right_gripper_indices), set_instantly=False)
        p.stepSimulation(physicsClientId=simulator.id)
        rgb, _ = simulator.render()
        grasp_rgbs.append(rgb)
    
    lift_rgbs = set_eef_to_pose(simulator, best_p_WG + np.array([0, 0, 0.1]), best_quat_WG, instant=False)

    return sampled_rgbs + rgbs + grasp_rgbs + lift_rgbs


def grasp_handle(simulator, handle_pcd, sample_num=100):
    ### set gripper to be open
    robot = simulator.robot
    robot.set_gripper_open_position(robot.right_gripper_indices, [0.1] * len(robot.right_gripper_indices), set_instantly=True)

    pc = handle_pcd
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)

    ### sample grasping pose (from Russ Tedrake's note)
    costs = []
    p_WGs = []
    quat_WGs = []
    sampled_rgbs = []
    for idx in range(sample_num):
        print("sampling grasping pose: ", idx)
        cost, p_WG, quat_WG, final_image = GenerateAntipodalGraspCandidate(simulator, pc, normals, object_id)
        costs.append(cost)
        p_WGs.append(p_WG)
        quat_WGs.append(quat_WG)
        sampled_rgbs.append(final_image)

    best_idx = np.argmin(costs)
    best_cost = costs[best_idx]
    best_p_WG = p_WGs[best_idx]
    best_quat_WG = quat_WGs[best_idx]

    ### execute grasping pose with the highest score
    rgbs = set_eef_to_pose(simulator, best_p_WG, best_quat_WG, instant=False)

    ### close gripper 
    robot = simulator.robot
    grasp_rgbs = []
    for _ in range(20):
        robot.set_gripper_open_position(robot.right_gripper_indices, [0] * len(robot.right_gripper_indices), set_instantly=False)
        p.stepSimulation(physicsClientId=simulator.id)
        rgb, _ = simulator.render()
        grasp_rgbs.append(rgb)
    
    lift_rgbs = set_eef_to_pose(simulator, best_p_WG + np.array([0, 0, 0.1]), best_quat_WG, instant=False)

    return sampled_rgbs + rgbs + grasp_rgbs + lift_rgbs

if __name__ == "__main__":
    from bullet_sim.sim import SimpleEnv
    from bullet_sim.utils import save_numpy_as_gif
    import cv2

    task_config = "bullet_sim/data/grasp.yaml"
    gui = True
    use_table = False
    env = SimpleEnv(config_path=task_config, 
                    gui=gui, 
                    panda_slider=True,
                    use_suction=True,
                    mobile=False,
                    translation_mode='direct-translation', 
                    rotation_mode='euler-angle',
                    gpt_adjust_position=False,
                )
    env.reset()
    p.addUserDebugLine([0, 1, 0], [0, 0, 0], [1, 0, 0], lineWidth=5, lifeTime=0, physicsClientId=env.id)
    
    rgb, depth = env.render()
    cv2.imwrite("bullet_sim/data/grasp.png", rgb)
    import pdb; pdb.set_trace()

    images = grasp(env, "mustard_bottle", 100)
    save_numpy_as_gif(np.array(images), "bullet_sim/data/grasp-3.gif")
    