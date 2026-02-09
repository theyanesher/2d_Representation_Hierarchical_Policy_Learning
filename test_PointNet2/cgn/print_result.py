import json
import os
from collections import defaultdict

## no pre contact
cgn_path = "data/eval_results/contact_graspnet"
gmm_path = "data/eval_results/gmm-no-sigmoid"
gmm_path_2 = "data/eval_results/0706_articubot_gmm"
gmm_path_2 = "data/eval_results/0706_articubot_gmm_2"
gmm_path_2 = "data/eval_results/0706_articubot_gmm_3"
gmm_open = "data/eval_results/0706_articubot_gmm_always_open"
gmm_open = "data/eval_results/0706_articubot_gmm_always_open_3"
gmm_open_gradshimt = "data/eval_results/0706_articubot_gmm_always_open_gradschimit"
cgn_4_point = "data/eval_results/test_4_point_training"

## with pre contact
cgn_path_precontact = "data/eval_results/contact_graspnet_precontact"
gmm_path_precontact = "data/eval_results/0706_articubot_gmm_3_precontact"
result_path = "data/cgn_eval_results/model_100000.pth"

result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/cgn_eval_results/GMM__2025-07-23cgn_batch_norm_full_fp_model_235000.pthinput_9000_precontact"
# result_path = "data/cgn_eval_results/GMM__2025-07-24cgn_alone_instancenorm_gmm_model_125000.pth_precontact"
# result_path = "data/cgn_eval_results/GMM__2025-07-25articubot_cgn_in_not_fp_to_full_model_50000.pth_precontact"
result_path = "data/cgn_eval_results/GMM__2025-07-25both_articubot_wdp_cgn_gmm_model_30000.pth_precontact"
# result_path = "data/cgn_eval_results/GMM__2025-07-24cgn_alone_instancenorm_gmm_model_125000.pth_model.train_precontact"
# result_path = "data/cgn_eval_results/GMM__2025-07-17articubot_cgn_first_try-2_model_100000.pth_precontact"
result_path = "data/cgn_eval_results/GMM__2025-07-27articubot_cgn_layernorm_model_57500.pth_precontact"
result_path = "data/cgn_eval_results/GMM__2025-07-28articubot_cgn_both_bn_gmm_just_lang_embed_model_55000.pth_precontact"
result_path = "data/cgn_eval_results/GMM__2025-07-25articubot_cgn_in_not_fp_to_full_model_80000.pth_precontact"
result_path = "data/cgn_eval_results/GMM__2025-07-27cgn_alone_layernorm_gmm_model_432500.pth_precontact"
# result_path = "data/cgn_eval_results/GMM__2025-07-27articubot_cgn_layernorm_model_65000.pth_precontact"
# result_path = "data/cgn_eval_results/GMM__2025-07-29articubot_cgn_both_ln_wdp_50_model_30000.pth_precontact"
result_path = "data/cgn_eval_results/GMM__2025-07-29articubot_cgn_both_ln_wdp_50_model_50000.pth_precontact"
result_path = "data/cgn_eval_results/2025-11-05full-pick-place-and-grasping-lift_1017-no-cgn_model_495001.pthreal_precontact"
result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet_pytorch/data/eval_results/contact_graspnettest_precontact"
result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/contact_graspnet_pytorch/data/cgn_eval_results/contact_graspnet_precontact"
result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/M2T2/data/cgn_eval_results/_precontact"
result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/contact_graspnet_pytorch/data/cgn_eval_results/contact_graspnet_precontact"

import argparse
parser = argparse.ArgumentParser(description="Print results from CGN and GMM evaluation directories.")
parser.add_argument('--path', type=str, default=None, help='Path to CGN evaluation results')
args = parser.parse_args()
result_path = args.path if args.path else result_path


def get_results(dir):
    results = defaultdict(int)
    for file in os.listdir(dir):
        if file.endswith('.json') and "meta_results" not in file:
            with open(os.path.join(dir, file), 'r') as f:
                data = json.load(f)
                for key in data:
                    results[key] += data[key]
                    
    total_num = sum(results.values())
    for key in results:
        results[key] = results[key] / total_num if total_num > 0 else 0
    return results

# res_cgn = get_results(cgn_path)
# res_gmm = get_results(gmm_path)
# res_gmm_2 = get_results(gmm_path_2)
# res_gmm_open = get_results(gmm_open)
# cgn_4_point_results = get_results(cgn_4_point)
# gmm_open_gradschimt = get_results(gmm_open_gradshimt)
# cgn_precontact = get_results(cgn_path_precontact)
# res_gmm_precontact = get_results(gmm_path_precontact)
res_result_path = get_results(result_path)

all_keys = set(res_result_path.keys()).union(set(res_result_path.keys()))
for key in all_keys:
    # if key in res_gmm:
    #     print(f"{key}: CGN: {res_cgn[key]:.4f}, GMM: {res_gmm[key]:.4f} GMM2: {res_gmm_2[key]:.4f} GMM_open: {res_gmm_open[key]:.4f} cgn-4-point: {cgn_4_point_results[key]:.4f}")
    # else:
    #     print(f"{key}: CGN: {res_cgn[key]:.4f}, GMM: N/A")
    print("================================= {} ==================================".format(key))
    # print(f"CGN: {res_cgn[key]:.4f}")
    # print(f"GMM: {res_gmm[key]:.4f}")
    # print(f"GMM2: {res_gmm_2[key]:.4f}")
    # print(f"GMM_open: {res_gmm_open[key]:.4f}")
    # print(f"GMM_open_gradschimt: {gmm_open_gradschimt[key]:.4f}")
    # print(f"CGN precontact: {cgn_precontact[key]:.4f}")
    # print(f"GMM precontact: {res_gmm_precontact[key]:.4f}")
    print(f"GMM_open precontact: {res_result_path[key]:.4f}")
