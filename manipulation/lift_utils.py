import pybullet as p
import os
import numpy as np
import open3d as o3d
from manipulation.motion_planning_utils import motion_planning
from manipulation.grasping_utils import get_pc_and_normal, align_gripper_z_with_normal, align_gripper_x_with_normal
from manipulation.gpt_reward_api import get_link_pc, get_bounding_box, get_link_id_from_name
from manipulation.utils import save_env, load_env
from manipulation.gpt_primitive_api import *
from manipulation.utils import save_numpy_as_gif, save_env, take_round_images, build_up_env, load_gif
import open3d
from manipulation.gpt_primitive_api import *
from manipulation.grasping_utils import *


def mp_to_target(env, target_pos, target_orientation, target_link=None, max_sampling_it=100):
    # return:
    #   success: True/False
    #   rgb
    #   states
    save_path = get_save_path(env)
    release_rgbs = []
    release_states = []
    release_steps = 20
    for t in range(release_steps):
        env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [0.04, 0.04], set_instantly=False)
        p.stepSimulation()
        rgb = env.render()
        release_rgbs.append(rgb)
        state_save_path = os.path.join(save_path, "states", "state_{}.pkl".format(t))
        save_env(env, state_save_path)
        release_states.append(state_save_path)
    ori_state = save_env(env, None)
    it = 0
    while True:
        all_objects = list(env.urdf_ids.keys())
        all_objects.remove('robot')
        obstacles = [env.urdf_ids[obj] for obj in all_objects]
        # obstacles = []
        allow_collision_links = []
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res, path = motion_planning(env, target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links, target_link=target_link, max_sampling_it=max_sampling_it, smooth_path=True)

        if res:
            rgbs = release_rgbs
            intermediate_states = release_states
            for idx, q in enumerate(path):
                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
                    
                rgb = env.render()
                rgbs.append(rgb)
                save_state_path = os.path.join(save_path, "states", "state_{}.pkl".format(idx + release_steps))
                save_env(env, save_state_path)
                intermediate_states.append(save_state_path)
            input("found a valid path, press enter to continue...")
            return True, rgbs, intermediate_states
        
        it += 1
        # input("here: " + str(it) + " times, press enter to continue...")
        if it % 10 == 0:
            print("===================== Failed to find a valid IK solution : " + str(it) + " times ==========================")
        if it > 10:
            print("Failed to find a valid IK solution")
            return False, None, None
        load_env(env, state=ori_state)

def mp_to_target_with_object(env, target_pos, target_orientation, object_name, target_link=None, max_sampling_it=100, index=0):
    save_path = get_save_path(env)
    release_rgbs = []
    release_states = []
    release_steps = 20
    object_id = env.urdf_ids[object_name]

    # check grasp the object already
    for t in range(5):
        cur_joint_angle = p.getJointState(env.robot.body, env.robot.right_gripper_indices[0], physicsClientId=env.id)[0]
        new_joint_angle = cur_joint_angle -  0.01
        new_joint_angle = np.clip(new_joint_angle, 0, 0.04)
        env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [new_joint_angle, new_joint_angle], set_instantly=False)
        p.stepSimulation(physicsClientId=env.id)
        if not p.getContactPoints(env.robot.body, object_id, env.robot.right_gripper_indices[0], -1, physicsClientId=env.id) \
            or not p.getContactPoints(env.robot.body, object_id, env.robot.right_gripper_indices[1], -1, physicsClientId=env.id):
            return False, release_rgbs, release_states
    
    print("========= grasp the object successfully ==========")
        
    # motion planning to the target pose without any collision of the robot body and the object
    ori_state = save_env(env, None)
    it = 0
    while True:
        all_objects = list(env.urdf_ids.keys())
        all_objects.remove('robot')
        all_objects.remove(object_name)
        obstacles = [env.urdf_ids[obj] for obj in all_objects]
        allow_collision_links = [] 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        res, path = motion_planning(env, target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links, object_id=object_id, target_link=target_link, max_sampling_it=max_sampling_it, smooth_path=True)
        
        if res:
            if object_name is not None:
                object_init_pos, object_init_orn = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
                robot_init_pos, robot_init_orn = env.robot.get_pos_orient(env.robot.right_end_effector)
                world_to_robot = p.invertTransform(robot_init_pos, robot_init_orn)
                object_in_robot = p.multiplyTransforms(world_to_robot[0], world_to_robot[1], object_init_pos, object_init_orn)
                object_in_robot_pos, object_in_robot_orn = object_in_robot[0], object_in_robot[1]
            rgbs = release_rgbs
            intermediate_states = release_states
            for idx, q in enumerate(path):
                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
                if object_name is not None:
                    robot_target_pos, robot_target_orn = env.robot.get_pos_orient(env.robot.right_end_effector)
                    object_target_pos, objectr_target_orientation = p.multiplyTransforms(robot_target_pos, robot_target_orn, object_in_robot_pos, object_in_robot_orn)
                    p.resetBasePositionAndOrientation(object_id, object_target_pos, objectr_target_orientation, physicsClientId=env.id)

                # env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [0, 0], set_instantly=False)
                    
                rgb = env.render()
                rgbs.append(rgb)
                save_state_path = os.path.join(save_path, "states", "state_{}.pkl".format(idx + release_steps + index))
                save_env(env, save_state_path)
                intermediate_states.append(save_state_path)
            input("found a valid path, press enter to continue...")
            return True, rgbs, intermediate_states
        it += 1
        # input("here: " + str(it) + " times, press enter to continue...")
        if it % 10 == 0:
            print("===================== Failed to find a valid IK solution : " + str(it) + " times ==========================")
        if it > 10:
            print("Failed to find a valid IK solution")
            return False, None, None
        load_env(env, state=ori_state)
    

def grasp_the_object(env, obj_name, index=0):
    # return:
    #   success: True/False
    #   rgb
    #   states
    save_path = get_save_path(env)
    ori_state = save_env(env, None)
    it = 0
    rgbs = []
    states = []
    while True:
        
        for t in range(20):
            rgb = env.render()
            save_state_path = os.path.join(save_path, "states", "state_{}.pkl".format(t+index))
            save_env(env, save_state_path)

            rgbs.append(rgb)
            states.append(save_state_path)

            cur_joint_angle = p.getJointState(env.robot.body, env.robot.right_gripper_indices[0], physicsClientId=env.id)
            # print("cur_joint_angle: ", cur_joint_angle)
            # input("press enter to continue...")
            new_joint_angle = cur_joint_angle[0] -  0.02
            new_joint_angle = np.clip(new_joint_angle, 0, 0.04)
            env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [new_joint_angle, new_joint_angle], set_instantly=False)
            p.stepSimulation(physicsClientId=env.id)
            
            if p.getContactPoints(env.robot.body, env.urdf_ids[obj_name], env.robot.right_gripper_indices[0], -1, physicsClientId=env.id) \
                and p.getContactPoints(env.robot.body, env.urdf_ids[obj_name], env.robot.right_gripper_indices[1], -1, physicsClientId=env.id):
                return True, rgbs, states
            # if p.getContactPoints(env.robot.body, env.urdf_ids[obj_name], env.robot.right_gripper_indices[0], -1, physicsClientId=env.id)
            # cur_joint_angle = p.getJointState(env.robot.body, env.robot.right_gripper_indices[0], physicsClientId=env.id)
            
        it += 1
        if it > 20:
            print("Failed to grasp the object")
            return False, None, None
        load_env(env, state=ori_state)
        rgbs = []
        states = []


def lift_up_the_object(env, obj_name, index=0, height=0.5):
    # return:
    #   success: True/False
    #   rgb
    #   states
    pass
    save_path = get_save_path(env)
    ori_state = save_env(env, None)
    # use ik to lift up the object 
    it = 0
    rgbs = []
    states = []
    while True:
        agent = env.robot
        joint = agent.right_end_effector
        success = True

        times = height / 0.05
        times = np.ceil(times).astype(int)
        for t in range(times):
            rgb = env.render()
            save_state_path = os.path.join(save_path, "states", "state_{}.pkl".format(t+index))
            save_env(env, save_state_path)

            rgbs.append(rgb)
            states.append(save_state_path)

            ik_indices = [_ for _ in range(len(env.robot.right_arm_joint_indices))]
            cur_pos, cur_orient = agent.get_pos_orient(joint)
            # print("cur_pos: ", cur_pos)
            target_pos = cur_pos + np.array([0, 0, 0.05])
            agent_joint_angles = agent.ik(joint, target_pos, cur_orient, ik_indices)
            for _ in range(2):
                agent.control(agent.controllable_joint_indices, agent_joint_angles)
            cur_joint_angle = p.getJointState(env.robot.body, env.robot.right_gripper_indices[0], physicsClientId=env.id)[0]
            new_joint_angle = cur_joint_angle -  0.005
            new_joint_angle = np.clip(new_joint_angle, 0, 0.04)
            env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [new_joint_angle, new_joint_angle], set_instantly=False)
            
            p.stepSimulation(physicsClientId=env.id)

            if not p.getContactPoints(env.robot.body, env.urdf_ids[obj_name], env.robot.right_gripper_indices[0], -1, physicsClientId=env.id) \
                or not p.getContactPoints(env.robot.body, env.urdf_ids[obj_name], env.robot.right_gripper_indices[1], -1, physicsClientId=env.id):
                success = False
                break

        if success:
            return True, rgbs, states
        
        it += 1
        if it > 20:
            print("Failed to lift up the object")
            return False, None, None
        load_env(env, state=ori_state)
        rgbs = []
        states = []

def set_eef_to_pose(simulator, target_pos, target_quat, instant=True):
    robot = simulator.robot
    joint = robot.right_end_effector 
    ik_indices = robot.right_arm_ik_indices 
    cur_pos, cur_orient = robot.get_pos_orient(robot.right_end_effector)

    if instant:
        iter = 0
        pos_error = np.linalg.norm(cur_pos - target_pos) 
        orient_error = np.linalg.norm(cur_orient - target_quat)
        while iter < 20:
            ik_lower_limits = simulator.robot.ik_lower_limits 
            ik_upper_limits = simulator.robot.ik_upper_limits 
            ik_start_pose = np.random.uniform(ik_lower_limits, ik_upper_limits)
            robot.set_joint_angles(ik_indices, ik_start_pose, use_limits=True, velocities=0)
            robot_joint_angles = simulator.robot.ik(joint, target_pos, target_quat, ik_indices, max_iterations=10000)
            robot.set_joint_angles(ik_indices, robot_joint_angles, use_limits=True, velocities=0)
            # p.stepSimulation(physicsClientId=simulator.id)
            cur_pos, cur_orient = robot.get_pos_orient(robot.right_end_effector)
            pos_error = np.linalg.norm(cur_pos - target_pos) 
            # orient_error = np.linalg.norm(cur_orient - target_quat)
            # print(f"iter {iter} pos error: {pos_error}, orient error: {orient_error}")
            # print(f"iter {iter} pos error: {pos_error}")
            if pos_error < 0.05:
                break
            iter += 1
    else:
        rgbs = []
        iter = 0
        while (np.linalg.norm(cur_pos - target_pos) > 0.01 or np.linalg.norm(cur_orient - target_quat) > 0.01) and iter < 10000:
            for _ in range(20):
                robot_joint_angles = simulator.robot.ik(joint, target_pos, target_quat, ik_indices, max_iterations=200)
                robot.control(robot.controllable_joint_indices, robot_joint_angles, robot.motor_gains, robot.motor_forces)
                p.stepSimulation(physicsClientId=simulator.id)
                rgb = simulator.render()
                rgbs.append(rgb)
                cur_pos, cur_orient = robot.get_pos_orient(robot.right_end_effector)
            iter += 1

        return rgbs

    
def GenerateAntipodalGraspCandidate(
    simulator,
    pc_points,
    pc_normals,
    object_id,
    index=None,
):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """
    if index is None:
        index = np.random.randint(0, pc_points.shape[0])

    # Use S for sample point/frame.
    p_WS = pc_points[index]
    n_WS = pc_normals[index]

    # print("sampled point position: ", p_WS)
    # print()

    assert np.isclose(
        np.linalg.norm(n_WS), 1.0
    ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gy = n_WS  #align franka y axis with the normal # NOTE: this is the kuka robot's x axis in Russ Tedrake's note.
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
    min_roll = -np.pi / 2.0
    max_roll = np.pi / 2.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    old_state = save_env(simulator)
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # simulator.reset(old_state)
        # p.addUserDebugLine(p_WS, p_WS + n_WS, [0, 1, 0], 10, 0, physicsClientId=simulator.id)
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
        print("set eef to sampled pose: ")
        old_joint_angles = simulator.robot.get_joint_angles()
        object_pos, object_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=simulator.id)
        set_eef_to_pose(simulator, p_WG, quat_WG2, instant=True)


        print("compute cost")
        cost = GraspCandidateCost(simulator, object_id, pc_points, pc_normals, adjust_X_G=True)
        final_image = simulator.render()

        # recover to old joint angles & old object pose
        # simulator.robot.set_joint_angles(simulator.robot.all_joint_indices, old_joint_angles, use_limits=True)
        # p.resetBasePositionAndOrientation(object_id, object_pos, object_orient, physicsClientId=simulator.id)
        # p.stepSimulation(physicsClientId=simulator.id)
        
        if np.isfinite(cost):
            simulator.reset(old_state)
            return cost, p_WG, quat_WG2, final_image

    simulator.reset(old_state)
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
        # p_GC_x = p_GC[0, indices]
        # p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        # gripper_new_pos = T_body_to_world @ np.array([p_Gcenter_x, 0, 0, 1]).reshape(4, 1)
        # gripper_new_pos = gripper_new_pos[:3, 0]
        gripper_new_pos = np.mean(pc_points[indices, :], axis=0)
        set_eef_to_pose(simulator, gripper_new_pos, cur_orient)
        
    for obj_id in simulator.urdf_ids.values():
        contact_points = p.getClosestPoints(simulator.robot.body, obj_id, 0, physicsClientId=simulator.id)
        if len(contact_points) > 0:
            print("there is already contact with the object! cost is inf")
            cost = np.inf
            return cost

    ### cost is how well the normal aligns with the gripper x axis (in the gripper frame). Russ's robot is x axis. 
    # n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])
    cost = 0
    n_GC = rotation @ (pc_normals.T)[:, indices] # 3x3 @ 3xN = 3xN
    cost -= np.sum(n_GC[1, :] ** 2) # in our case, it is the franka y axis.

    ### Original Russ: Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    # cost += 20.0 * X_G.rotation().matrix()[2, 1]
    
    ### ours: we should probably encourage the gripper to have horizontal grasp. This is implemented later

    print(f"cost: {cost}")
    return cost


def sampling_grasp_and_lift(simulator, object_name, sample_num=100, sort_for_future_putting=False, target_pos=None):
    ### set gripper to be open
    robot = simulator.robot
    robot.set_gripper_open_position(robot.right_gripper_indices, [0.1] * len(robot.right_gripper_indices), set_instantly=True)

    ### take round view images of the objects, convert to point clouds, merge all point clouds, voxelized point cloud
    pc, normals = get_pc_and_normal(simulator, object_name)

    object_id = simulator.urdf_ids[object_name]
    ### sample grasping pose (from Russ Tedrake's note)
    costs = []
    p_WGs = []
    quat_WGs = []
    sampled_rgbs = []
    for idx in range(sample_num):
        print("sampling grasping pose: ", idx)
        # if idx == 0:
        #     cost, p_WG, quat_WG, final_image = GenerateAntipodalGraspCandidate(simulator, pc, normals, object_id, index=np.argmax(pc[:, 2]))
        # else:
        cost, p_WG, quat_WG, final_image = GenerateAntipodalGraspCandidate(simulator, pc, normals, object_id)
        costs.append(cost)
        p_WGs.append(p_WG)
        quat_WGs.append(quat_WG)
        sampled_rgbs.append(final_image)

    # for image in sampled_rgbs[:10]:
    #     plt.imshow(image)
    #     plt.show()
        
    if sort_for_future_putting and target_pos is not None:
        # get the orientation score for each grasp
        horizon_scores = []
        align_scores = []
        for i in range(len(p_WGs)):
            h_s, a_s = get_orientation_score_for_future_putting(simulator, object_name, target_pos, quat_WGs[i])
            horizon_scores.append(h_s)
            align_scores.append(a_s)
        costs = np.array(costs) - np.array(horizon_scores) * 10 - np.array(align_scores) * 10


    input("press enter to execute the best grasping pose")
    index = np.argsort(costs, axis=0)
    mp_rgb, mp_states, mp_success = [], [], False
    grasp_rgb, grasp_states, grasp_success = [], [], False
    lift_rgb, lift_states, lift_success = [], [], False
    max_it = sample_num // 20
    p_WG, quat_WG = None, None
    ori_state = save_env(simulator, None)
    for i in range(max_it):
        simulator.reset(ori_state)
        print("try to execute the best grasping pose: ", i, " with cost: ", costs[index[i]])
        # input("press enter to continue...")
        p_WG = p_WGs[index[i]]
        quat_WG = quat_WGs[index[i]]
        mp_success, mp_rgb, mp_states = mp_to_target(simulator, p_WG, quat_WG, max_sampling_it=50)
        if mp_success:
            print("motion planning success!")
            save_numpy_as_gif(np.array(mp_rgb), 'sampling_based_lift/mp.gif')
            grasp_success, grasp_rgb, grasp_states = grasp_the_object(simulator, object_name, index=len(mp_rgb))
            if grasp_success:
                print("Grasp success")
                save_numpy_as_gif(np.array(grasp_rgb), 'sampling_based_lift/grasp.gif')
                input("press enter to continue...")
                lift_success, lift_rgb, lift_states = lift_up_the_object(simulator, object_name, index=len(mp_rgb)+len(grasp_rgb))
                if lift_success:
                    print("Lift success")
                    save_numpy_as_gif(np.array(lift_rgb), 'sampling_based_lift/lift.gif')
                    # break
                else:
                    save_numpy_as_gif(np.array(lift_rgb), 'sampling_based_lift/lift_failed.gif')
                    print("Lift failed")
            else:
                print("Grasp failed")
        else:
            print("Motion planing failed")

        mp_rgb, mp_states, mp_success = [], [], False
        grasp_rgb, grasp_states, grasp_success = [], [], False
        lift_rgb, lift_states, lift_success = [], [], False

        # load_env(simulator, state=ori_state)

    if mp_success and grasp_success and lift_success:
        print("Success")
        total_rgb = mp_rgb + grasp_rgb + lift_rgb
        total_states = mp_states + grasp_states + lift_states
        save_numpy_as_gif(np.array(total_rgb), 'lift_total.gif')
    else:
        print("Failed to lift the object with contact GraspNet")

    return mp_rgb + grasp_rgb + lift_rgb


def generate_pc_for_contact_graspnet(env, object_name):
    # env.reset()
    full_pc = get_full_pc_aroung_obj(env, object_name, elevation=30)
    pc, nomals = get_pc_and_normal(env, object_name, elevation=30)
    save_f = {}
    save_f['xyz'] = full_pc
    save_f['pc_seg'] = {1: pc}
    np.savez(f'contact_graspnet_predict/{object_name}.npz', **save_f)

def get_orientation_score_for_future_putting(env, object_name, target_pos, orientation):
    # make sure it is a horizontal grasp and the orientation is the same as the line from object to target
    # import pdb; pdb.set_trace()
    if np.array(orientation).shape[0] == 3:
        orientation = p.getQuaternionFromEuler(orientation)

    # orientation is a quaternion
    ori_matrix = p.getMatrixFromQuaternion(orientation)
    z_axis = np.array(np.array(ori_matrix).reshape([3, 3])[:, 2])

    horizontal_score = -np.abs(np.dot(z_axis, [0, 0, 1]))
    
    # make sure the target orientation is the same as the object orientation
    object_id = env.urdf_ids[object_name]
    object_pos, object_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)

    object2target = np.array(target_pos) - np.array(object_pos)
    object2target[2] = 0
    object2target = object2target / np.linalg.norm(object2target)

    z_axis_h = z_axis
    z_axis_h[2] = 0
    z_axis_h = z_axis_h / np.linalg.norm(z_axis_h)

    align_score = np.dot(object2target, z_axis_h)

    return horizontal_score, align_score
    

if __name__ == '__main__':

    simulator, save_config = build_up_env(
        # "data/generated_task_from_description/lift_a_hamburger/lift_a_hamburger.yaml",
        # "data/generated_task_from_description/lift_a_hamburger/task_lift_a_hamburger",
        # "lift_a_hamburger",
        # "initial_states/smaller_hamburger_lift_initialization_not_close.pkl",
        "/home/ziyu/Desktop/workspace/RoboGen-sim2real/data/generated_task_from_description/Open_a_microwave_Microwave_100426_2024-04-06-10-44-48/Open_a_microwave_The_robotic_arm_will_open_the_microwave_door.yaml",
        "data/generated_task_from_description/Open_a_microwave_Microwave_100426_2024-04-06-10-44-48/task_Open_a_microwave",
        "open_the_microwave_door",
        None,
        # "initial_states/bottle_lift_initialization_not_close.pkl",
        # "data/generated_task_from_description/lift_a_box/lift_a_box.yaml",
        # "data/generated_task_from_description/lift_a_box/task_lift_a_box",
        # "lift_a_box",
        # "initial_states/gold_bar_initialization_not_close.pkl",
        render=True, 
        randomize=False, 
        obj_id=0
    )
    simulator.primitive_save_path = 'sampling_based_lift'
    object_name = "target_object"
    import pdb; pdb.set_trace()

    # save the point cloud for contact graspnet
    # generate_pc_for_contact_graspnet(simulator, object_name)

    # execute sampling based grasping and lifting
    rgbs = sampling_grasp_and_lift(simulator, object_name, sample_num=20)
    save_numpy_as_gif(np.array(rgbs), "sampling_based_lift/grasp_and_lift.gif")