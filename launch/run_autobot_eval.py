import os
import subprocess
import time
import json
from chester.run_exp import VariantGenerator

# GPU 控制参数
MAX_PROC_PER_GPU = 1
NUM_GPUS = 8

def get_gpu_process_count():
    """通过分析 nvidia-smi 输出，统计每张 GPU 的活跃进程数"""
    result = subprocess.run(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output = result.stdout
    gpu_counts = [0] * NUM_GPUS
    # 处理每一行输出数据，找出每个 GPU 上的进程
    for line in output.splitlines():
        parts = line.split()  # 基于空格来分隔
        
        if len(parts) >= 3 and "python" in line: 
            print(parts)
            gpu_id = int(parts[1].strip().replace("GPU-", ""))  # 提取 GPU 编号
            process_name = parts[6].strip()  # 提取进程名称
            if process_name == "python":  # 只统计 python 进程
                gpu_counts[gpu_id] += 1
    print(gpu_counts)
    return gpu_counts

def wait_for_free_gpu():
    """等待直到有 GPU 空闲（少于 MAX_PROC_PER_GPU），返回 GPU 编号"""
    while True:
        counts = get_gpu_process_count()
        for i, cnt in enumerate(counts):
            if cnt < MAX_PROC_PER_GPU:
                return i
        print("所有 GPU 都在运行中，等待 10 秒...")
        time.sleep(10)

def vv_to_params_seuss(vv):
    return f"{vv['folder_name']} {vv['exp_folder']}"

def run_task(vv):
    params = vv_to_params_seuss(vv)

    log_dir = os.path.join("/project_data/held/chenyuah/RoboGen-sim2real/sbatch_logs", vv['exp_folder'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=4, sort_keys=True)

    gpu_id = wait_for_free_gpu()
    out_log = os.path.join(log_dir, f'stdout_gpu{gpu_id}.log')
    err_out = os.path.join(log_dir, f'stderr_gpu{gpu_id}.err')

    command = f"CUDA_VISIBLE_DEVICES={gpu_id} bash scripts/sbatch_eval.sh {params} > {out_log} 2> {err_out} &"
    print(f"[GPU {gpu_id}] Running: {command}")
    os.system(command)
    time.sleep(60)

# ----------------------------------------
# 各任务列表
# ----------------------------------------
bucket_tasks = [100435, 100441]
faucet_tasks = [149, 960, 991]
foldingchair_tasks = [100520, 100521, 100526]
laptop_tasks = [9748, 9912, 9960]
stapler_tasks = [102990, 103095]
toilet_tasks = [101320, 102620, 102621]
storagefurniture_tasks = [41510, 45448, 46462, 46732, 46801]

# 转为字符串
bucket_tasks = [str(i) for i in bucket_tasks]
faucet_tasks = [str(i) for i in faucet_tasks]
foldingchair_tasks = [str(i) for i in foldingchair_tasks]
laptop_tasks = [str(i) for i in laptop_tasks]
stapler_tasks = [str(i) for i in stapler_tasks]
toilet_tasks = [str(i) for i in toilet_tasks]
storagefurniture_tasks = [str(i) for i in storagefurniture_tasks]

# ----------------------------------------
# 批量运行任务
# ----------------------------------------

task_sets = [
    ("data/bucket/", bucket_tasks),
    ("data/faucet/", faucet_tasks),
    ("data/foldingchair/", foldingchair_tasks),
    ("data/laptop/", laptop_tasks),
    ("data/stapler/", stapler_tasks),
    ("data/toilet/", toilet_tasks),
    ("data/storagefurniture/", storagefurniture_tasks),
]

for folder_name, task_list in task_sets:
    vg = VariantGenerator()
    vg.add("folder_name", [folder_name])
    vg.add("exp_folder", task_list)
    for vv in vg.variants():
        run_task(vv)
