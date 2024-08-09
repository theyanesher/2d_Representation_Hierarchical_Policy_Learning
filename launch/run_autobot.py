import os
# os.system("export PYTHONPATH=${PWD}:PYTHONPATH")

import time
import datetime
import json
from launch.utils import check_available_nodes_cpu_and_mem, AUTOBOT_NODELIST
from chester.run_exp import VariantGenerator

autobot_user = 'yufeiw2'
autobot_project_folder = 'RoboGen_sim2real'

def vv_to_params_autobot(vv):
    params = "{} {}".format(
        vv['exp_folder'], vv['save_data_name']
    )
        
    print("running params: ", params)
    return params

def run_task(vv, available_nodes):
    log_file = '/project_data/held/yufeiw2/RoboGen_sim2real/data/local/gen_data.log'
    exp_name = vv['exp_name']
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    
    log_dir = os.path.join("/project_data/held/{}/{}/".format(autobot_user, autobot_project_folder), "data/local", exp_name + "_" + time_string)
    out_log = os.path.join(log_dir, 'stdout.log')
    err_out = os.path.join(log_dir, 'stdout.err')

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=4, sort_keys=True)

    print("available nodes: ", available_nodes)
    print("vv: ", vv)
        
    for node in available_nodes:
        if node in AUTOBOT_NODELIST:
            params = vv_to_params_autobot(vv)
            real_node = "autobot-" + node
            command = "ssh -q {} \'nohup singularity exec --bind /project_data/held/{}/{}:/mnt/{}/ --nv /project_data/held/yufeiw2/robogen-dp3-act3d.sif /mnt/{}/scripts/gen_data_parallel.sh {} > {} 2> {} &\'".format(
                real_node,
                autobot_user, autobot_project_folder, autobot_project_folder, autobot_project_folder,
                params, out_log, err_out)
            print(command)
            os.system(command)
            with open(log_file, 'a') as f:
                f.write("running on node: {} command {}\n".format(node, params))
            time.sleep(30)
            break

# generate all parameter combinations you want to test
vg = VariantGenerator()
vg.add("exp_name", ["test-gen-data"])

vg.add("exp_folder", [
    "open_the_door_45413"
])
vg.add("save_data_name",[
    "debug"
])
# for each parameter combination, run the epxeriment
all_vvs = vg.variants()

success = False
while not success:
    available_nodes = check_available_nodes_cpu_and_mem()
    success = True
    
for vv in all_vvs:
    run_task(vv, available_nodes)