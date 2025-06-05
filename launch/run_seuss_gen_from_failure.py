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

    command = "sbatch -o {} -e {} -J {} scripts/sbatch_gen_from_failure.sh {}".format(
        out_log, err_out, vv['exp_folder'],  params)

    print(command)
    os.system(command)
    time.sleep(5)

bucket_tasks = [100443, 100444, 100452, 100454, 100460, 100461, 100462, 100469, 100472, 102352, 102358, 102365]
faucet_tasks = [149, 152, 153, 154, 168, 811, 822, 857, 908, 929, 1028, 1052, 1053, 1288, 1343, 1370, 1466, 1492, 1528, 1626, 1633, 1646, 1668, 1741, 1794, 1795, 1802, 1885, 1901, 1903, 1925,  1961, 1986, 2054]
foldingchair_tasks = [100531, 100532, 100557, 100561, 100562, 100568, 100579, 100586, 100590, 100599, 100600, 100608, 100609, 100611, 100616, 102255, 102263, 102269, 102314]
laptop_tasks = [9968, 9992, 9996, 10040, 10098, 10101, 10238, 10243, 10248,10269, 10270, 10280, 10289, 10305, 10306, 10383, 10626, 10697, 10885, 10915, 11075, 11156, 11242, 11248, 11395, 11405, 11406, 11429, 11477, 11581, 11586, 11691, 11778, 11876, 11888, 11945, 12073]
stapler_tasks = [103099, 103100, 103104, 103111, 103113, 103271, 103275, 103276, 103280, 103292, 103293, 103297, 103299, 103301, 103303, 103305, 103789, 103792]
toilet_tasks = [102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654, 102658, 102663, 102666, 102667, 102668, 102669, 102670, 102675, 102676, 102677, 102687, 102689, 102692, 102694, 102697, 102699, 102701, 102703, 102707, 102708, 103234]
bucket_tasks = [str(i) for i in bucket_tasks]
faucet_tasks = [str(i) for i in faucet_tasks]
foldingchair_tasks = [str(i) for i in foldingchair_tasks]
laptop_tasks = [str(i) for i in laptop_tasks]
stapler_tasks = [str(i) for i in stapler_tasks]
toilet_tasks = [str(i) for i in toilet_tasks]
# storagefurniture_tasks = [str(i) for i in storagefurniture_tasks]

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

# # vg = VariantGenerator()
# # vg.add("folder_name", ["data/storagefurniture/"])
# # vg.add("exp_folder", storagefurniture_tasks)

# # all_vvs = vg.variants()
# # for vv in all_vvs:
# #     run_task(vv)