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
gmm_open_precontact = "data/eval_results/0706_articubot_gmm_always_open_3_precontact"
gmm_open_precontact = "data/eval_results/0706_articubot_gmm_always_open_3_precontact"


def get_results(dir):
    results = defaultdict(int)
    for file in os.listdir(dir):
        if file.endswith('.json'):
            with open(os.path.join(dir, file), 'r') as f:
                data = json.load(f)
                for key in data:
                    results[key] += data[key]
                    
    total_num = sum(results.values())
    for key in results:
        results[key] = results[key] / total_num if total_num > 0 else 0
    return results

res_cgn = get_results(cgn_path)
res_gmm = get_results(gmm_path)
res_gmm_2 = get_results(gmm_path_2)
res_gmm_open = get_results(gmm_open)
cgn_4_point_results = get_results(cgn_4_point)
gmm_open_gradschimt = get_results(gmm_open_gradshimt)
cgn_precontact = get_results(cgn_path_precontact)
res_gmm_precontact = get_results(gmm_path_precontact)
res_gmm_open_precontact = get_results(gmm_open_precontact)

all_keys = set(res_cgn.keys()).union(set(res_gmm_open_precontact.keys()))
for key in all_keys:
    # if key in res_gmm:
    #     print(f"{key}: CGN: {res_cgn[key]:.4f}, GMM: {res_gmm[key]:.4f} GMM2: {res_gmm_2[key]:.4f} GMM_open: {res_gmm_open[key]:.4f} cgn-4-point: {cgn_4_point_results[key]:.4f}")
    # else:
    #     print(f"{key}: CGN: {res_cgn[key]:.4f}, GMM: N/A")
    print("================================= {} ==================================".format(key))
    print(f"CGN: {res_cgn[key]:.4f}")
    print(f"GMM: {res_gmm[key]:.4f}")
    print(f"GMM2: {res_gmm_2[key]:.4f}")
    print(f"GMM_open: {res_gmm_open[key]:.4f}")
    print(f"GMM_open_gradschimt: {gmm_open_gradschimt[key]:.4f}")
    print(f"CGN precontact: {cgn_precontact[key]:.4f}")
    print(f"GMM precontact: {res_gmm_precontact[key]:.4f}")
    print(f"GMM_open precontact: {res_gmm_open_precontact[key]:.4f}")
