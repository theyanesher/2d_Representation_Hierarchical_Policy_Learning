from manipulation.utils import build_up_env
from manipulation.motion_planning_utils import motion_planning, motion_planning_joint_angle
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import pickle
import time

env, safe_config = build_up_env(
    "example_tasks/Change_Lamp_Direction/Change_Lamp_Direction_The_robotic_arm_will_alter_the_lamps_light_direction_by_manipulating_the_lamps_head.yaml",
    "example_tasks/Change_Lamp_Direction/task_Change_Lamp_Direction",
    "grasp_the_lamps_head", 
    "example_tasks/Change_Lamp_Direction/task_Change_Lamp_Direction/primitive_states/2024-03-03-22-04-48/grasp_the_lamps_head/state_0.pkl",

    # "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/Open_Microwave_Door_The_robotic_arm_will_open_the_microwave_door_to_insert_or_remove_items.yaml",
    # "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door",
    # "open_the_microwave_door", 
    # "data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door/experiment/2024-03-04-21-44-32/grasp_the_microwave_door_primitive/states/state_140.pkl", 
    
    render=True, 
    randomize=False, 
    obj_id=0
)

# motion plan to the target position and orientation
# with open("example_tasks/Change_Lamp_Direction/task_Change_Lamp_Direction/primitive_states/2024-03-03-16-03-07/grasp_the_lamps_head/target_joint_angle.pkl", "rb") as f:
with open("example_tasks/Change_Lamp_Direction/task_Change_Lamp_Direction/primitive_states/2024-03-03-22-04-48/grasp_the_lamps_head/target_joint_angle.pkl", "rb") as f:
# with open("data/generated_tasks_release/Microwave_7310_2024-03-04-21-20-19/task_Open_Microwave_Door/experiment/2024-03-04-21-44-32/grasp_the_microwave_door_primitive/target_joint_angle.pkl", "rb") as f:
    target_joint_angle = pickle.load(f)
with open("example_tasks/Change_Lamp_Direction/task_Change_Lamp_Direction/primitive_states/2024-03-03-22-04-48/grasp_the_lamps_head/current_joint_angle.pkl", "rb") as f:
    current_joint_angle = pickle.load(f)

all_objects = list(env.urdf_ids.keys())
all_objects.remove("robot")
obstacles = [env.urdf_ids[x] for x in all_objects]
allow_collision_links = []

planners = ["RRT", "RRTConnect", "RRTStar", "TRRT", "pRRT", "STRRTstar", "BITstar", "ABITstar", "AITstar"]
# planners = ["BITstar"]
plan_times = 10
ori_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)

planner_translations = defaultdict(list)
planner_rotations = defaultdict(list) 
planner_times = defaultdict(list) 
for planner in planners:
    for idx in range(plan_times):
        env.reset()
        env.robot.set_joint_angles(env.robot.right_arm_joint_indices, current_joint_angle)
        
        beg = time.time()
        res, path = motion_planning(env, None, None, target_joint_angle=target_joint_angle, planner=planner, obstacles=obstacles, allow_collision_links=allow_collision_links)
        end = time.time()
        planner_times[planner].append(end - beg)
        if res:
            cur_pos, cur_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
            length_pos = 0
            length_orient = 0
            for idx, q in enumerate(path):
                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
                pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
                length_pos += np.linalg.norm(pos - cur_pos)
                length_orient += np.arccos(2 * np.dot(orient, cur_orient)**2 - 1)
                cur_pos, cur_orient = pos, orient
            planner_translations[planner].append(length_pos)
            planner_rotations[planner].append(length_orient)
        else:
            planner_translations[planner].append(-1)
            planner_rotations[planner].append(-1)
        
max_translation_length = np.max([np.max(planner_translations[planner]) for planner in planners])
max_rotation_length = np.max([np.max(planner_rotations[planner]) for planner in planners])
min_translation_length = np.min([np.min(planner_translations[planner]) for planner in planners])
min_rotation_length = np.min([np.min(planner_rotations[planner]) for planner in planners])

for planner in planners:
    print("Planner: ", planner)
    print("Translation Lengths: ", np.mean(planner_translations[planner]))
    print("Rotation Lengths: ", np.mean(planner_rotations[planner]))
    print("Times: ", np.mean(planner_times[planner]))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()
# plot the bin histogram of translation and rotation lengths
for planner_idx, planner in enumerate(planners):
    # plot the error bar
    axes[0].bar(planner_idx, np.mean(planner_translations[planner]), yerr=np.std(planner_translations[planner]), label=planner)
    axes[1].bar(planner_idx, np.mean(planner_rotations[planner]), yerr=np.std(planner_rotations[planner]), label=planner)
    axes[2].bar(planner_idx, np.mean(planner_times[planner]), yerr=np.std(planner_times[planner]), label=planner)
    axes[0].set_title("Translation Lengths")
    axes[1].set_title("Rotation Lengths")
    axes[2].set_title("Times")
    axes[0].set_ylabel("Mean Translation Lengths")
    axes[1].set_ylabel("Mean Rotation Lengths")
    axes[2].set_ylabel("Mean Times")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    
plt.savefig("data/planner_comparison_2.png")
plt.show()
    
        
            
        
