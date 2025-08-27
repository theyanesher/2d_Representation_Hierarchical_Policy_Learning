import os
import shutil
from multiprocessing import Pool, cpu_count
import time

path = "/project/flame/yufeiw2/RoboGen-sim2real/data/dp3_demo"

def delete_folder(folder_name):
    full_path = os.path.join(path, folder_name)
    if os.path.isdir(full_path): #and "1121" in folder_name:
        beg = time.time()
        shutil.rmtree(full_path)
        print(f"Deleting: {full_path} using time {time.time() - beg}")

if __name__ == "__main__":
    all_contexts = sorted(os.listdir(path))
    folders_to_delete = [
        context for context in all_contexts
        if os.path.isdir(os.path.join(path, context)) #and "1121" in context
    ]

    with Pool(processes=10) as pool:
        pool.map(delete_folder, folders_to_delete)
