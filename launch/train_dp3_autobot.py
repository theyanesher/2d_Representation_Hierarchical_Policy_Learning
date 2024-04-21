import os
# os.system("export PYTHONPATH=${PWD}:PYTHONPATH")

import time
import datetime
import json
from launch.utils import check_available_nodes, AUTOBOT_NODELIST
from chester.run_exp import VariantGenerator

autobot_user = 'yufeiw2'
autobot_project_folder = 'RoboGen_sim2real'

def vv_to_params_autobot(vv):
    params = "{} {} {} {}".format(
        vv['exp_name'], 
        vv['dataset_name'], 
        vv['in_gripper_frame'], 
        vv['cuda_id']
    )

        
    print("running params: ", params)
    return params

def run_task(vv, available_nodes):
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
    # import pdb; pdb.set_trace()
    
    
    for node in available_nodes:
        if node in AUTOBOT_NODELIST:
            gpu_ids = available_nodes[node]
            if len(gpu_ids) > 0:
                vv['cuda_id'] = gpu_ids[0]
                params = vv_to_params_autobot(vv)
                real_node = "autobot-" + node
                command = "ssh -q {} \'{} & nohup singularity exec --bind /project_data/held/{}/{}:/mnt/{}/ --nv /project_data/held/yufeiw2/robogen-dp3.sif /mnt/{}/launch/train_dp3.sh {} > {} 2> {} &\'".format(
                    real_node, 
                    "export CUDA_VISIBLE_DEVICES={}".format(gpu_ids[0]),
                    autobot_user, autobot_project_folder, autobot_project_folder, autobot_project_folder,
                    params, out_log, err_out)
                gpu_ids.pop(0)
                print(command)
                import pdb; pdb.set_trace()
                os.system(command)
                time.sleep(30)
                break

# generate all parameter combinations you want to test
vg = VariantGenerator()
vg.add("exp_name", ["test-autobot-train-dp3"])

vg.add("dataset_name", ["test_different_init_joint_angle_gripper"])
vg.add("in_gripper_frame", [1])
vg.add("cuda_id", [0]) 

# for each parameter combination, run the epxeriment
all_vvs = vg.variants()

success = False
while not success:
    available_nodes = check_available_nodes()
    success = True
    
for vv in all_vvs:
    run_task(vv, available_nodes)