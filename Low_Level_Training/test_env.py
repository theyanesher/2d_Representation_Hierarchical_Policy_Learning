# ddptest.py
import os, torch
local_rank = os.environ.get("LOCAL_RANK")
rank = os.environ.get("RANK")
world_size = os.environ.get("WORLD_SIZE")
print(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    if local_rank is not None:
        lr = int(local_rank)
        try:
            print("device name:", torch.cuda.get_device_name(lr))
        except Exception as e:
            print("device name error:", e)
except Exception as e:
    print("cuda error:", e)
