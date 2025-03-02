import numpy as np
import pybullet as p
from bullet_sim.grasping_primitive import get_pc_and_normal, align_gripper_z_with_normal, align_gripper_x_with_normal
from bullet_sim.gpt_reward_api import get_bounding_box
from bullet_sim.motion_planning_utils import motion_planning
import cv2

def sample_grasp_pose(simulator, object_pc, object_normal, object_com):
    
    random_point = object_pc[np.random.randint(0, object_pc.shape[0])]
    random_normal = object_normal[np.random.randint(0, object_normal.shape[0])]

    ### adjust the normal such that it points outwards the object.
    line = object_com - random_point
    if np.dot(line, random_normal) > 0:
        random_normal = -random_normal

    mp_target_poses = []
    target_orientations = []
    for normal in [random_normal, -random_normal]:
        target_pos = random_point
        # TODO: we can further randomize the grasping pose inside align_gripper_z_with_normal/align_gripper_x_with_normal
        if simulator.robot_name in ["panda", "sawyer"]:
            target_orientation = align_gripper_z_with_normal(-normal, random=True).as_quat()
            mp_target_pos = target_pos #+ normal * 0.03
        elif simulator.robot_name in ['ur5', 'fetch']:
            target_orientation = align_gripper_x_with_normal(-normal, random=True).as_quat()
            if simulator.robot_name == 'ur5':
                mp_target_pos = target_pos #+ normal * 0.07
            elif simulator.robot_name == 'fetch':
                mp_target_pos = target_pos #+ normal * 0.07
        mp_target_poses.append(mp_target_pos)
        target_orientations.append(target_orientation)

    return mp_target_poses, target_orientations

def pick_and_place_ab(simulator, obj_a, obj_b):
    obj_a_id, obj_b_id = simulator.urdf_ids[obj_a], simulator.urdf_ids[obj_b]

    bbox_b_min, bbox_b_max = simulator.get_aabb(obj_b_id)
    bbox_b_range = bbox_b_max - bbox_b_min
    sample_min = bbox_b_min - 0.1 * bbox_b_range
    sample_max = bbox_b_max + 0.1 * bbox_b_range
    
    obj_a_init_pos, obj_a_init_orient = p.getBasePositionAndOrientation(obj_a_id)
    obj_b_init_pos, obj_b_init_orient = p.getBasePositionAndOrientation(obj_b_id)
    init_joint_angles = simulator.robot.get_joint_angles(indices=simulator.robot.right_arm_joint_indices)
    
    sample_num = 500
    rewards = []
    obj_a_new_poses, obj_a_new_orients = [], []
    for sample_idx in range(sample_num):
    
        collision = True
        while collision:
            obj_a_new_pos = np.random.uniform(sample_min, sample_max)
            obj_a_new_orient = obj_a_init_orient # TODO: sample a new orientation
            p.resetBasePositionAndOrientation(obj_a_id, obj_a_new_pos, obj_a_new_orient)
            
            # check collision -- if there is collision, discard the sample
            p.performCollisionDetection(physicsClientId=simulator.id)
            contact_points = p.getContactPoints(obj_a_id, physicsClientId=simulator.id)
            if len(contact_points) == 0:
                collision = False
                break
        
        # should record the pose as before the object settles    
        obj_a_settled_pos, obj_a_settled_orient = p.getBasePositionAndOrientation(obj_a_id)
        obj_a_new_poses.append(obj_a_settled_pos)
        obj_a_new_orients.append(obj_a_settled_orient)
            
        # if no collision, wait for the object to settle
        wait_steps = 50
        for _ in range(wait_steps):
            p.stepSimulation()
        
        # use the reward function to score the sample
        reward, _ = simulator._compute_reward()
        rewards.append(reward)
        
        print("sampling obj a pose {} with reward {}".format(sample_idx, reward))
        
        # reset object a and b to its original position to prepare for the next sample
        p.resetBasePositionAndOrientation(obj_a_id, obj_a_init_pos, obj_a_init_orient)
        p.resetBasePositionAndOrientation(obj_b_id, obj_b_init_pos, obj_b_init_orient)


    # sample according to the reward
    rewards = np.array(rewards)
    sort_idx = np.argsort(rewards)[::-1] # from large to small

    try_object_pose_num = 10
    sample_grasp_pose_num = 100
    for object_pose_try_idx in sort_idx[:try_object_pose_num]:
        best_obj_place_pos = obj_a_new_poses[object_pose_try_idx]
        best_obj_place_orient = obj_a_new_orients[object_pose_try_idx]
        
        print("trying object target pose {} with reward {}".format(object_pose_try_idx, rewards[object_pose_try_idx]))
        p.resetBasePositionAndOrientation(obj_a_id, best_obj_place_pos, best_obj_place_orient)
        rgb, _ = simulator.render()
        cv2.imwrite("data/debug/{}_{}.png".format(object_pose_try_idx, round(reward, 3)), rgb)
        # import pdb; pdb.set_trace()
        
              
        # sample grasp pose for object a in place pose
        for grasp_try_idx in range(sample_grasp_pose_num):
            print("\tsample grasp pose {}".format(grasp_try_idx))
            p.resetBasePositionAndOrientation(obj_a_id, best_obj_place_pos, best_obj_place_orient)
            object_pc, object_normal = get_pc_and_normal(simulator, obj_a)
            bbox_min, bbox_max = get_bounding_box(simulator, obj_a)
            object_com = (bbox_min + bbox_max) / 2
            grasp_place_positions, grasp_place_orients = sample_grasp_pose(simulator, object_pc, object_normal, object_com)
            
            for grasp_place_pos, grasp_place_orient in zip(grasp_place_positions, grasp_place_orients):
                # compute the relative pose between robot and object a 
                # relative_pose_object_to_gripper = p.multiplyTransforms(grasp_place_pos, grasp_place_orient, *p.invertTransform(best_obj_place_pos, best_obj_place_orient))
                world_to_obj = p.invertTransform(best_obj_place_pos, best_obj_place_orient)
                relative_pose_gripper_to_object = p.multiplyTransforms(world_to_obj[0], world_to_obj[1], grasp_place_pos, grasp_place_orient)
            
                # compute the grasp pose for pick up object a in the current pose, keeping the relative pose unchanged
                # grasp_pick_pos, grasp_pick_orient = p.multiplyTransforms(relative_pose_gripper_to_object[0], relative_pose_gripper_to_object[1], obj_a_init_pos, obj_a_init_orient)
                grasp_pick_pos, grasp_pick_orient = p.multiplyTransforms(obj_a_init_pos, obj_a_init_orient, relative_pose_gripper_to_object[0], relative_pose_gripper_to_object[1])
            
                # try motiong planning from default pose to initial pose
                p.resetBasePositionAndOrientation(obj_a_id, obj_a_init_pos, obj_a_init_orient)
                simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, init_joint_angles)
                all_objects = list(simulator.urdf_ids.keys())
                all_objects.remove("robot")
                obstacles = [simulator.urdf_ids[x] for x in all_objects]
                # TODO: ideally should motion plan the pre-contact pose, and then do the contact. However that way the code logic will be much more complicated.
                obstacles.remove(obj_a_id) # allow collision with object a, to simplify things for now 
                allow_collision_links = []
                
                res_to_pick_grasp, path_to_pick_grasp = motion_planning(simulator, grasp_pick_pos, grasp_pick_orient, obstacles=obstacles, allow_collision_links=allow_collision_links)
                if not res_to_pick_grasp:
                    print("failed to find a path from default pose to pick pose")
                    continue
            
                ## try motion planning from pick pose to place pose
                # set robot to pick pose
                # TODO: should take account into the collision of the obejct in this step during motion planning
                p.resetBasePositionAndOrientation(obj_a_id, best_obj_place_pos, best_obj_place_orient)
                simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, path_to_pick_grasp[-1])
                res_pick_to_place, path_pick_to_place = motion_planning(simulator, grasp_place_pos, grasp_place_orient, obstacles=obstacles, allow_collision_links=allow_collision_links)
                if not res_pick_to_place:
                    print("failed to find a path from pick pose to place pose")
                    continue
                
                ## we find a path from default pose to pick pose to place pose
                p.resetBasePositionAndOrientation(obj_a_id, obj_a_init_pos, obj_a_init_orient)
                p.resetBasePositionAndOrientation(obj_b_id, obj_b_init_pos, obj_b_init_orient)
                simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, init_joint_angles)
                return path_to_pick_grasp, path_pick_to_place
            

if __name__ == "__main__":
    import importlib
    from bullet_sim.utils import default_config
    import copy
    import pickle
    
    task_config = "example_tasks/store_an_item_into_the_storagefurniture/store_an_item_into_the_storagefurniture_The_robot_arm_picks_up_an_item_and_places_it_inside_the_storage_furniture_then_closes_the_door_of_the_storage_furniture.yaml"
    solution_path = "example_tasks/store_an_item_into_the_storagefurniture/task_store_an_item_into_the_storagefurniture"
    task_name = "put_the_item_into_the_storage_furniture"
    module = importlib.import_module("{}.{}".format(solution_path.replace("/", "."), task_name))
    env_class = getattr(module, task_name)

    save_config = copy.deepcopy(default_config)
    save_config['config_path'] = task_config
    save_config['task_name'] = task_name
    save_config['restore_state_file'] = None
    save_config['translation_mode'] = "delta-translation"
    save_config['gui'] = True
    save_config['randomize'] = False
    save_config['use_bard'] = False
    save_config['obj_id'] = 0
    save_config['use_gpt_size'] = True
    save_config['use_gpt_joint_angle'] = True
    save_config['use_gpt_spatial_relationship'] = True
    save_config['use_distractor'] = False
    
    env = env_class(**save_config)
    
    env.reset()
    
    pick_path, place_path = pick_and_place_ab(env, "fruit", "fruit bowl")
    with open("data/debug/pick_and_place_path.pkl", "wb") as f:
        pickle.dump((pick_path, place_path), f)
    
    imgs = []
    for q in pick_path:
        for _ in range(3):
            # env.robot.control(env.robot.right_arm_joint_indices, q, env.robot.motor_gains, forces=5 * 240.)
            env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
            p.stepSimulation()
            img, _ = env.render()
            imgs.append(img)
    for _ in range(20):
        # env.robot.control(env.robot.right_arm_joint_indices, pick_path[-1], env.robot.motor_gains, forces=5 * 240.)
        env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
        p.stepSimulation()
        img, _ = env.render()
        imgs.append(img)
    
    env.activate_suction()
    
    for q in place_path:
        for _ in range(3):
            env.robot.control(env.robot.right_arm_joint_indices, q, env.robot.motor_gains, forces=5 * 240.)
            p.stepSimulation()
            img, _ = env.render()
            imgs.append(img)
    for _ in range(20):
        env.robot.control(env.robot.right_arm_joint_indices, place_path[-1], env.robot.motor_gains, forces=5 * 240.)
        # env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
        p.stepSimulation()
        img, _ = env.render()
        imgs.append(img)
    
    from bullet_sim.utils import save_numpy_as_gif
    save_numpy_as_gif(np.array(imgs), "data/debug/pick_and_place.gif", fps=10)