import json
import os

eval_result_path = '/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/1211_grasp_and_lift'
# eval_result_path = '/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_1202_0_new_2_closed_goal'
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_1202_0_no_one_hot_2_closed_goal"
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_eval_0_grasping_low_level"
# # eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_eval_0"
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_1218_grasp_0_grasping_low_level"
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_eval_0_grasping_low_level_2"
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_1218_grasp_0_grasping_low_level_2"
# # eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_1218_grasp_0_grasping_low_level_3"
# # eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_eval_0_grasping_low_level_3"
# # eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_1218_grasp_0_grasping_low_level_pap_high_level"
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/debug_pap_1218_grasp_0_grasping_low_level_pap_high_level_increased_friction"
# all_scenes = os.listdir(os.path.join(eval_result_path, "top_left"))
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/1227_grasp_and_ik_gripper_width"
eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/1227_grasp_and_ik_gripper_width_2"
# eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/1227_grasp_and_ik"
# eval_result_path = '/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data_eval/1229_grasp_attn'
eval_result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/contact_graspnet_pytorch/data/cgn_eval_results/contact_graspnet200_precontact"
# all_scenes = os.listdir(os.path.join(eval_result_path, "grasp"))
all_scenes = os.listdir(os.path.join(eval_result_path))


grasp_successes = []
lifted_successes = []
for scene in all_scenes:
    # json_path = os.path.join(eval_result_path, "top_left", scene, "opened_joint_angles.json")
    # json_path = os.path.join(eval_result_path, "grasp", scene, "opened_joint_angles.json")
    json_path = os.path.join(eval_result_path, scene, "opened_joint_angles.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    for entry in data.values():
        # grasp_success = entry["grasped_and_lifted"]
        grasp_success = entry["grasp_success"]
        grasp_successes.append(grasp_success)
        lift_success = entry["grasped_and_lifted"]
        # lift_success = entry["place_success"]
        lifted_successes.append(lift_success)
        
print("Average grasp success for all scenes: " + str(sum(grasp_successes) / len(grasp_successes)))
print("Average lift success for all scenes: " + str(sum(lifted_successes) / len(lifted_successes)))
print("Total number of trials: " + str(len(grasp_successes)))
print("Total number of successful grasps: " + str(sum(grasp_successes)))
print("Total number of successful lifts: " + str(sum(lifted_successes)))