import numpy as np
import pybullet as p
import cv2
from manipulation.grasping_utils import get_pc_and_normal, rotation_matrix_y
from manipulation.lift_utils import build_up_env, set_eef_to_pose, save_numpy_as_gif, mp_to_target, grasp_the_object, mp_to_target_with_object, lift_up_the_object
from manipulation.gpt_reward_api import set_joint_value, get_link_id_from_name
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
import time
from manipulation.utils import save_env, load_env

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
    np.random.seed(time.time_ns() % 2**32)
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

    gripper_new_pos = cur_pos
    if adjust_X_G and np.sum(indices) > 0:
        # print("adjusting pose to center at the point cloud!")
        set_eef_to_pose(simulator, gripper_new_pos, cur_orient)
        
    for name, obj_id in simulator.urdf_ids.items():
        if name == 'robot' or name == 'plane': continue
        contact_points = p.getClosestPoints(simulator.robot.body, obj_id, 0, physicsClientId=simulator.id)
        if len(contact_points) > 0:
            # print("there is already contact with the object! cost is inf")
            cost = np.inf
            return cost, gripper_new_pos

    ### cost is how well the normal aligns with the gripper x axis (in the gripper frame). Russ's robot is x axis. 
    cost = 0
    n_GC = rotation @ (pc_normals.T)[:, indices] # 3x3 @ 3xN = 3xN
    cost -= np.sum(n_GC[1, :] ** 2) # in our case, it is the franka y axis.

    ### Original Russ: Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    # cost += 20.0 * X_G.rotation().matrix()[2, 1]
    
    ### ours: we should probably encourage the gripper to have horizontal grasp. This is implemented later

    # print(f"cost: {cost}")
    print("found a finite cost: ", cost)
    return cost, gripper_new_pos

def GenerateAntipodalGraspCandidate(
    args
):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """
    env_kwargs, pc_points, pc_normals, object_id, a_new_pos, a_init_pos, a_init_orn, index = args
    simulator, _ = build_up_env(**env_kwargs)
    simulator.reset()
    set_joint_value(simulator, 'microwave', 'joint_0')
    p.resetBasePositionAndOrientation(object_id, a_new_pos, a_init_orn, physicsClientId=simulator.id)
    
    # print("generate antipodal grasp candidate!")
    # time.sleep(10)
    ### set gripper to be open
    robot = simulator.robot
    robot.set_gripper_open_position(robot.right_gripper_indices, [0.04] * len(robot.right_gripper_indices), set_instantly=True)
    
    if index is None:
        index = np.random.randint(0, pc_points.shape[0])

    # Use S for sample point/frame.
    p_WS = pc_points[index]
    n_WS = pc_normals[index]

    assert np.isclose(
        np.linalg.norm(n_WS), 1.0
    ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gy = n_WS  #align franka y axis with the normal # NOTE: this is the kuka robot's x axis in Russ Tedrake's note.
    # make orthonormal z axis, aligned with world down
    z = np.array([0.0, 0.0, -1.0])

    Gz = z - np.dot(z, Gy) * Gy
    Gx = np.cross(Gy, Gz)
    # R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0, 0, 0] # TODO: check the length and coordinate of the franka gripper

    # Try orientations from the center out
    min_roll = -np.pi / 2.0
    max_roll = np.pi / 2.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    costs = []
    gripper_positions = []
    gripper_orientations = []
    final_images = []
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # simulator.reset(old_state)
        p.addUserDebugLine(p_WS, p_WS + n_WS, [0, 1, 0], 10, 0, physicsClientId=simulator.id)
        # Rotate the object in the hand by a random rotation (around the normal).
        # R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta)) # NOTE: for Russ's robot, this is rotating around the x axis.
        # R_WG2 = R_WG * rotation_matrix_x(theta)
        R_WG2 = R_WG * rotation_matrix_y(theta) # NOTE: for franka, it should be rotating around the y axis.
        
        # Use G for gripper frame.
        p_SG_W = - (R_WG2.as_matrix() @ np.array(p_GS_G).reshape(3, 1)).flatten()
        p_WG = p_WS.flatten()  + p_SG_W # in our case p_SG_W is always 0, beacuse we are directly using the link of panda grasp center

        # X_G = RigidTransform(R_WG2, p_WG)
        # plant.SetFreeBodyPose(plant_context, wsg, X_G)

        # use ik to set the robot eef to the target pose
        quat_WG2 = R_WG2.as_quat() # scipy quaternion is [x y z w], e.g., [0 0 0 1] is identity, which aligns with what pybullet is using. 
        # print("set eef to sampled pose: ")
        set_eef_to_pose(simulator, p_WG, quat_WG2, instant=True)

        # print("theta: ", theta)
        # time.sleep(5)


        # print("compute cost")
        cost, gripper_new_pos = GraspCandidateCost(simulator, object_id, pc_points, pc_normals, adjust_X_G=True)
        final_image = simulator.render()

        if np.isfinite(cost):
            init_p_WG = p_WG - a_new_pos + a_init_pos
            init_quat_WG = quat_WG2
            set_eef_to_pose(simulator, init_p_WG, init_quat_WG, instant=True)
            init_cost, _ = GraspCandidateCost(simulator, object_id, pc_points, pc_normals, adjust_X_G=True)
            if not np.isfinite(init_cost):
                cost = np.inf

        costs.append(cost)
        gripper_positions.append(gripper_new_pos)
        gripper_orientations.append(quat_WG2)
        final_images.append(final_image)

    simulator.close()
    lowest_cost_index = np.argmin(costs)
    lowest_cost = costs[lowest_cost_index]
    if not np.isfinite(lowest_cost):
        return np.inf, None, None, final_image
    else:
        return lowest_cost, gripper_positions[lowest_cost_index], gripper_orientations[lowest_cost_index], final_images[lowest_cost_index]


def pick_and_place_ab(simulator, env_kwargs, obj_a, obj_b, target_bbox=None, parallel_worker_num=64):
    a_id, b_id = simulator.urdf_ids[obj_a], simulator.urdf_ids[obj_b]

    bbox_a_min, bbox_a_max = simulator.get_aabb(a_id)

    if target_bbox is None:
        bbox_b_min, bbox_b_max = simulator.get_aabb(b_id)
    else:
        bbox_b_min, bbox_b_max = target_bbox

    bbox_a_range = bbox_a_max - bbox_a_min

    sample_min = bbox_b_min + 0.5 * bbox_a_range
    sample_max = bbox_b_max - 0.5 * bbox_a_range

    a_init_pos, a_init_orn = p.getBasePositionAndOrientation(a_id)
    b_init_pos, b_init_orn = p.getBasePositionAndOrientation(b_id)
    sample_num = 500
    rewards = []
    a_new_poses = []

    for i in range(sample_num):
        collision = True
        while collision:
            a_new_pos = np.random.uniform(sample_min, sample_max)
            p.resetBasePositionAndOrientation(a_id, a_new_pos, a_init_orn)
            
            # check collision
            p.performCollisionDetection()
            contact_points = p.getContactPoints(a_id)
            if len(contact_points) == 0:
                collision = False
                break

        a_new_poses.append(a_new_pos)
        # rewards.append(-(np.linalg.norm(a_new_pos - np.array([-0.1,0.02,0.45])))**2)
        rewards.append(simulator._compute_reward()[0])

        p.resetBasePositionAndOrientation(a_id, a_init_pos, a_init_orn)
        p.resetBasePositionAndOrientation(b_id, b_init_pos, b_init_orn)
    

    # trying to sample the grasp pose
    rewards = np.array(rewards)
    sorted_idx = np.argsort(rewards)[::-1]

    try_object_pose_num = 10
    sample_grasp_pose_num = 256

    for obj_pose_i in range(try_object_pose_num):
        print("Trying to place the object with the ", obj_pose_i, "th pose with reward: ", rewards[sorted_idx[obj_pose_i]])
        
        a_new_pos = a_new_poses[sorted_idx[obj_pose_i]]
        p.resetBasePositionAndOrientation(a_id, a_new_pos, a_init_orn)

        robot = simulator.robot
        robot.set_gripper_open_position(robot.right_gripper_indices, [0.1] * len(robot.right_gripper_indices), set_instantly=True)

        pc, normals = get_pc_and_normal(simulator, obj_a)
        costs = []
        p_WGs = []
        quat_WGs = []
        sampled_rgbs = []
        pool = Pool(processes=parallel_worker_num)
        sample_iters = sample_grasp_pose_num // parallel_worker_num
        for it in range(sample_iters):
            return_list = pool.map(GenerateAntipodalGraspCandidate, [(env_kwargs, pc, normals, a_id, a_new_pos, a_init_pos, a_init_orn, None) for i in range(parallel_worker_num)])
            costs.extend([x[0] for x in return_list])
            p_WGs.extend([x[1] for x in return_list])
            quat_WGs.extend([x[2] for x in return_list])
            sampled_rgbs.extend([x[3] for x in return_list])
        pool.close()
        

        sorted_grasp_idx = np.argsort(costs)
        mp_1_success, mp_1_rgb, mp_1_states = False, [], []
        grasp_success, grasp_rgb, grasp_states = False, [], []
        lift_success, lift_rgb, lift_states = False, [], []
        mp_2_success, mp_2_rgb, mp_2_states = False, [], []

        ori_state = save_env(simulator, None)
        for grasp_idx in sorted_grasp_idx:
            p_WG = p_WGs[grasp_idx]
            quat_WG = quat_WGs[grasp_idx]
            if not np.isfinite(costs[grasp_idx]) and costs[grasp_idx] < 0:
                mp_1_success, mp_1_rgb, mp_1_states = False, [], []
                grasp_success, grasp_rgb, grasp_states = False, [], []
                lift_success, lift_rgb, lift_states = False, [], []
                mp_2_success, mp_2_rgb, mp_2_states = False, [], []
                break 
            
            load_env(simulator, state=ori_state)
            print("p_WG: ", p_WG + a_init_pos - a_new_pos)
            print("quat_WG: ", quat_WG)
            input("here comes a finite cost, try to move the gripper to the grasp the original object")
            
            p.resetBasePositionAndOrientation(a_id, a_init_pos, a_init_orn)
            # first, try to move the gripper to the grasp the original object
            p_WG_init = p_WG + a_init_pos - a_new_pos
            quat_WG_init = quat_WG
            mp_1_success, mp_1_rgb, mp_1_states = mp_to_target(simulator, p_WG_init, quat_WG_init, max_sampling_it=50)
            if mp_1_success:
                print("First stage motion planning success! Moved to the init grasp pose")
                save_numpy_as_gif(np.array(mp_1_rgb), "local_gifs/pick_place_mp_1.gif")
                # then, try to grasp the original object
                grasp_success, grasp_rgb, grasp_states = grasp_the_object(simulator, obj_a, index=len(mp_1_rgb))
                if grasp_success:
                    print("Grasp success! Grasped the original object")
                    save_numpy_as_gif(np.array(grasp_rgb), "local_gifs/pick_place_grasp.gif")
                    lift_success, lift_rgb, lift_states = lift_up_the_object(simulator, obj_a, index=len(mp_1_rgb) + len(grasp_rgb), height=0.05)
                    if lift_success:
                        print("Lift success! Lifted the original object")
                        save_numpy_as_gif(np.array(lift_rgb), "local_gifs/pick_place_lift.gif")
                        save_env(simulator, 'pick_place_grasp.pkl')
                        print("saved the state after lift")
                        print("p_WG: ", p_WG)
                        print("quat_WG: ", quat_WG)
                        input("try to move the gripper to the target pose without any collision of the robot body and the object")
                        # next, try to move the gripper to the target pose without any collision of the robot body and the object
                        mp_2_success, mp_2_rgb, mp_2_states = mp_to_target_with_object(simulator, p_WG, quat_WG, object_name=obj_a, max_sampling_it=50, index=len(grasp_rgb) + len(mp_1_rgb) + len(lift_rgb))
                        if mp_2_success:
                            print("Second stage motion planning success! Moved to the target pose")
                            save_numpy_as_gif(np.array(mp_2_rgb), "local_gifs/pick_place_mp_2.gif")
                            break
                        else:
                            print("Second stage motion planning failed!")
                            continue
                    else:
                        print("Lift failed!")
                        continue
                else:
                    print("Grasp failed!")
                    continue
            else:
                print("First stage motion planning failed!")
                continue

        if mp_1_success and grasp_success and lift_success and mp_2_success:
            print("Pick and place success!")
            total_rgb = mp_1_rgb + grasp_rgb + lift_rgb + mp_2_rgb
            total_states = mp_1_states + grasp_states + lift_states + mp_2_states
            save_numpy_as_gif(np.array(total_rgb), "local_gifs/pick_and_place_success.gif")
            return total_rgb
        else:
            print("Pick and place failed for this target position!")

    return mp_1_rgb + grasp_rgb + mp_2_rgb
    

if __name__ == '__main__':

    # env_kwargs = {
    #     "task_config": "data/generated_task_from_description/robogen_put_an_object_into_microwave_stand/bottle/put_an_object_into_microwave.yaml",
    #     "solution_path": "data/generated_task_from_description/robogen_put_an_object_into_microwave_stand/put_an_object_into_microwave",
    #     "task_name": "put_an_object_into_microwave",
    #     # "restore_state_file": "/home/ziyu/Desktop/workspace/RoboGen-sim2real/pick_place_grasp.pkl",
    #     "restore_state_file": None, # "/home/ziyu/Desktop/workspace/RoboGen-sim2real/pick_place_grasp.pkl",
    #     "render": False,
    #     "randomize": False,
    #     "obj_id": 0,
    # }
    # render_env_kwargs = env_kwargs.copy()
    # render_env_kwargs['render'] = True
    # simulator, _ = build_up_env(**render_env_kwargs)
    # # input("1, press enter to continue")
    # simulator.primitive_save_path = 'put_a_hamburger_into_microwave_stand'
    # object_name = 'target_object'
    # set_joint_value(simulator, 'microwave', 'joint_0')
    # obj_b_link_id = get_link_id_from_name(simulator, "microwave", 'link_3')
    # target_bbox = simulator.get_aabb_link(simulator.urdf_ids['microwave'], obj_b_link_id)

    # rgbs = pick_and_place_ab(simulator, env_kwargs, object_name, 'microwave', target_bbox, parallel_worker_num=6)


    env_kwargs = {
        "task_config": "data/generated_task_from_description/robogen_put_an_object_into_box_stand/coffee/put_an_object_into_microwave.yaml",
        "solution_path": "data/generated_task_from_description/robogen_put_an_object_into_box_stand/put_an_object_into_microwave",
        "task_name": "put_an_object_into_microwave",
        # "restore_state_file": "/home/ziyu/Desktop/workspace/RoboGen-sim2real/pick_place_grasp.pkl",
        "restore_state_file": None, # "/home/ziyu/Desktop/workspace/RoboGen-sim2real/pick_place_grasp.pkl",
        "render": False,
        "randomize": False,
        "obj_id": 0,
    }
    render_env_kwargs = env_kwargs.copy()
    render_env_kwargs['render'] = True
    simulator, _ = build_up_env(**render_env_kwargs)
    # input("1, press enter to continue")
    simulator.primitive_save_path = 'put_a_hamburger_into_box_stand'
    object_name = 'target_object'
    set_joint_value(simulator, 'microwave', 'joint_0')
    obj_b_link_id = get_link_id_from_name(simulator, "microwave", 'link_1')
    target_bbox = simulator.get_aabb_link(simulator.urdf_ids['microwave'], obj_b_link_id)

    import pdb; pdb.set_trace()

    rgbs = pick_and_place_ab(simulator, env_kwargs, object_name, 'microwave', target_bbox, parallel_worker_num=8)
    # print("len of rgbs: ", len(rgbs))