import subprocess

gpu_ids = subprocess.check_output(['nvidia-smi', '--list-gpus']).decode().split('\n')
gpu_ids = [line[4] for line in gpu_ids if line]
# print("all_ids: ",     gpu_ids)

gpu_processes = {}

usable_gpu_ids = []
for gpu_id in gpu_ids:
    # print("checking gpu_id: ", gpu_id)
    # print("command: ", ['nvidia-smi', '-i', str(gpu_id), '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
    processes = subprocess.check_output(['nvidia-smi', '-i', str(gpu_id), '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits']).decode().strip().split('\n')
    if processes and processes[0]:
        gpu_processes[gpu_id] = [(p.split(',')[0].strip(), p.split(',')[1].strip()) for p in processes]
    else:
        usable_gpu_ids.append(gpu_id)

for id in usable_gpu_ids:
    print(id, end=" ")
