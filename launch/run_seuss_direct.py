import os
# os.system("export PYTHONPATH=${PWD}:PYTHONPATH")

import time
import datetime
import json
from chester.run_exp import VariantGenerator

seuss_user = 'yufeiw2'
seuss_project_folder = 'RoboGen_sim2real'

def vv_to_params_seuss(vv):
    params = "{} {}".format(
        vv['exp_folder'], vv['save_data_name']
    )
        
    print("running params: ", params)
    return params

def run_task(vv):
    log_file = '/data/yufeiw2/RoboGen_sim2real/data/local/gen_data.log'
    exp_name = vv['exp_name']
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    
    log_dir = os.path.join("/data/{}/{}/".format(seuss_user, seuss_project_folder), "data/local", exp_name + "_" + time_string)
    out_log = os.path.join(log_dir, 'stdout.log')
    err_out = os.path.join(log_dir, 'stdout.err')

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=4, sort_keys=True)

    print("vv: ", vv)
        
    params = vv_to_params_seuss(vv)
    real_node = 'compute-1-9'
    command = "ssh -q {} \'nohup singularity exec --bind /data/{}/{}:/mnt/{}/ --nv /data/yufeiw2/robogen-dp3-act3d.sif /mnt/{}/scripts/gen_data_parallel.sh {} > {} 2> {} &\'".format(
        real_node,
        seuss_user, seuss_project_folder, seuss_project_folder, seuss_project_folder,
        params, out_log, err_out)
    # command = "sbatch scripts/sbatch_gen_data.sh {}".format(
        # params)
    print(command)
    os.system(command)
    time.sleep(5)

def get_save_data_name(exp_folder):
    return "0725-obj-" + str(exp_folder.split("_")[-1])

# generate all parameter combinations you want to test
vg = VariantGenerator()
vg.add("exp_name", ["test-gen-data"])

vg.add("exp_folder", [
    # "open_the_door_45413", 
    # "open_the_door_45420", 
    # "open_the_door_45427", 
    # "open_the_door_45443", 
    # "open_the_door_45504", 
    # "open_the_door_45594", 
    # "open_the_door_45620", 
    # "open_the_door_45623", 
    # "open_the_door_45633", 
    # "open_the_door_45636", 
    # "open_the_door_45667",  ### using slurm command for above
    
    # "open_the_door_45670",
    # "open_the_door_45689", ### 1-5
    
    # "open_the_door_45690", # killed
    # "open_the_door_45696", ### 1-9
    # "open_the_door_45717", ### 1-9
    
    # "open_the_door_45725", ### 0-23 failed
    # "open_the_door_45747", ### 0-23 # failed
    
    # "open_the_door_45749" ## 0-21
    # "open_the_door_45759" ## 0-23
    
    ### only extract data, 0-21
    # "open_the_door_45413",
    # "open_the_door_45420",
    
    ### only extract data, 0-23
    # "open_the_door_45427",
    # "open_the_door_45594",
    
    ### only extract data, 1-5
    # "open_the_door_45620",
    # "open_the_door_45623",
    
    ### only extract data, 1-9
    # "open_the_door_45670",
    # "open_the_door_45689",
    
    ### compute-0-21
    # "open_the_door_46480",
    # "open_the_door_46481",
    
    ### compute-0-23
    # "open_the_door_46544",
    # "open_the_door_46563",
    
    ### compute-1-5
    # "open_the_door_46641",
    # "open_the_door_46655",
    
    ### compute-1-9
    "open_the_door_46896",
    "open_the_door_47133",
])
vg.add("save_data_name", lambda exp_folder: [get_save_data_name(exp_folder)])
# for each parameter combination, run the epxeriment
all_vvs = vg.variants()
for vv in all_vvs:
    run_task(vv)