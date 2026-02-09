import json
import os
from collections import defaultdict

## no pre contact
result_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/M2T2/data/cgn_eval_results_200/_precontact/"

def get_results(dir):
    results = defaultdict(int)
    for file in os.listdir(dir):
        if file.endswith('.json') and not 'meta' in file:
            with open(os.path.join(dir, file), 'r') as f:
                data = json.load(f)
                for key in data:
                    results[key] += data[key]
                    
    total_num = sum(results.values())
    for key in results:
        results[key] = results[key] / total_num if total_num > 0 else 0
    return results

results = get_results(result_path)
for key in results:
    print(f"{key}: {results[key]:.4f}")