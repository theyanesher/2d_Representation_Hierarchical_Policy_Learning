import os
# os.system("export PYTHONPATH=${PWD}:PYTHONPATH")

import time
import json
from chester.run_exp import VariantGenerator

def vv_to_params_seuss(vv):
    params = "{} {}".format(
        vv['folder_name'], vv['exp_folder'],
    )
    print("running params: ", params)
    return params

def run_task(vv):
    print("vv: ", vv)
        
    params = vv_to_params_seuss(vv)

    log_dir = os.path.join("/data/chenyuah/RoboGen-sim2real/sbatch_logs", vv['exp_folder'])
    out_log = os.path.join(log_dir, 'stdout.log')
    err_out = os.path.join(log_dir, 'stdout.err')

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=4, sort_keys=True)

    command = "sbatch -o {} -e {} -J {} scripts/sbatch_eval.sh {}".format(
        out_log, err_out, vv['exp_folder'],  params)

    print(command)
    os.system(command)
    time.sleep(5)

bucket_tasks = [100444]#, 100452, 100454, 100460, 100461, 100462, 100469, 100472, 102352, 102365]
faucet_tasks = [148, 149, 152, 153, 154, 168, 811, 857, 960, 991]
foldingchair_tasks = [100520, 100521, 100526, 100562, 100586, 100590, 100599, 102263, 102269, 102314]
laptop_tasks = [9748, 9912, 9960, 9968, 9992, 9996, 10040, 10098, 10101, 10238]
stapler_tasks = [103095, 103099, 103100, 103104, 103111, 103292, 103293, 103297, 103299, 103301]
toilet_tasks = [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102658]
bucket_tasks = [str(i) for i in bucket_tasks]
faucet_tasks = [str(i) for i in faucet_tasks]
foldingchair_tasks = [str(i) for i in foldingchair_tasks]
laptop_tasks = [str(i) for i in laptop_tasks]
stapler_tasks = [str(i) for i in stapler_tasks]
toilet_tasks = [str(i) for i in toilet_tasks]



# generate all parameter combinations you want to test
vg = VariantGenerator()

vg.add("folder_name", ["data/bucket/"])
vg.add("exp_folder", bucket_tasks)

# for each parameter combination, run the epxeriment
all_vvs = vg.variants()
for vv in all_vvs:
    run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/faucet/"])
# vg.add("exp_folder", faucet_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/foldingchair/"])
# vg.add("exp_folder", foldingchair_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/laptop/"])
# vg.add("exp_folder", laptop_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/stapler/"])
# vg.add("exp_folder", stapler_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/toilet/"])
# vg.add("exp_folder", toilet_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)