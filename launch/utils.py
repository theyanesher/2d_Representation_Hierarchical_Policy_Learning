import inspect
import numpy as np
import glob
import pickle
from pynvml import *

GPU_STATE_DIR = '/project_data/mtcmon/nodestats/'
AUTOBOT_NODELIST = [
    # '0-9', 
    # '0-11', 
    # '0-13', 
    # '0-15', 
    # '0-17', 
    # '0-19', 
    # '0-21', # 1
    # '0-23',    
    #   # 4x2080 + 32 cpus
    '0-25', 
    # '0-29', 
    # '0-33', # 8x3090 + 96 cpus
    # '0-37', # 1 # 8x3090 + 96 cpus
    # '1-14', # 1/2 # 10 x 3080 ti
    # '1-1', # 
    # "1-10" # 1/2
    # "1-18" # 1/2
]

def check_available_nodes():
    available_nodes = {}
    for node in AUTOBOT_NODELIST:
        print("checking node: ", node)
        gpu_ids = check_node_gpu("autobot-" + node)
        gpu_ids = gpu_ids.strip().lstrip()
        print("gpu_ids: ", gpu_ids)
        if gpu_ids == "":
            available_nodes[node] = []
            continue
        if gpu_ids is not None:
            available_nodes[node] = gpu_ids.strip().split(" ")
            
    return available_nodes
            
import subprocess

def check_node_gpu(node, script_path="/project_data/held/yufeiw2/BPref/launch/check_gpu.py"):
    try:
        # Construct the ssh command
        cmd = ['ssh', node, 'python', script_path]
        
        # Execute the command and fetch its output
        output = subprocess.check_output(cmd)
        
        return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error {e.returncode}, output: {e.output.decode('utf-8')}")
        return None

# Use the function
if __name__ == "__main__":
    node_name = "autobot-0-25"
    script_path = "/project_data/held/yufeiw2/pytorch_sac/launch/check_gpu.py"
    result = check_node_gpu(node_name, script_path)
    print(result)