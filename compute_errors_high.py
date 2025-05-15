import json
import os
from statistics import mean

# 所有任务列表
categories = {
    "bucket": [100444, 100452, 100454, 100460, 100461, 100462, 100469, 100472, 102352, 102365],
    "faucet": [148, 149, 152, 153, 154, 168, 811, 857, 960],
    "foldingchair": [100520, 100521, 100526, 100562, 100586, 100590, 100599, 102263, 102269, 102314],
    "laptop": [9748, 9912, 9960, 9968, 9992, 9996, 10040, 10098, 10101, 10238],
    "stapler": [103095, 103099, 103100, 103104, 103111, 103292, 103293, 103297, 103299, 103301],
    "toilet": [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102658],
    "storagefurniture": [41510, 45448, 46462, 46732, 46801, 46874, 46922, 46966, 47570, 47578],
}

base_path = "data/0513_siglip"

category_avgs = {}
all_avgs = []

# 遍历每个类别和任务
for category, tasks in categories.items():
    avg_list = []
    for task_id in tasks:
        json_path = os.path.join(base_path, category, str(task_id), "error.json")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                avg_val = data["avg"]
                avg_list.append(avg_val)
                all_avgs.append(avg_val)
        except FileNotFoundError:
            print(f"[WARN] 文件不存在: {json_path}")
        except Exception as e:
            print(f"[ERROR] 读取失败 {json_path}: {e}")
    category_avgs[category] = mean(avg_list) if avg_list else None

# 打印每个类别的平均值
for cat, avg in category_avgs.items():
    print(f"{cat}: {avg}")

# 计算整体平均值
if all_avgs:
    overall_avg = mean(all_avgs)
    print(f"\nOverall avg: {overall_avg}")
else:
    print("\n没有可用的 avg 数据来计算 overall 平均值。")
