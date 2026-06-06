import numpy as np
import os
from tqdm import tqdm

ds1 = '/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/Articubot_Data_For_RVT/SMITH_High_Level_FineTune/LOW_LEVEL_NO_GMM_DATASET_GROOT_STYLE_DATASET/D2/Mug_Cleanup_D1'
ds2 = '/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data_yufei/mimicgen/mug_cleanup_d1_larger_ws'

all_s_diffs, all_a_diffs, all_pc_diffs, all_gpc_diffs, all_goal_gpc_diffs, missing, total = [], [], [], [], [], 0, 0

for demo_idx in tqdm(range(100), desc='Demos'):
    files = sorted(os.listdir(f'{ds2}/demo_{demo_idx}'), key=lambda x: int(x.split('.')[0]))
    demo_s, demo_a, demo_pc, demo_gpc, demo_goal_gpc = [], [], [], [], []
    for fname in tqdm(files, desc=f'  demo_{demo_idx}', leave=False):
        p1, p2 = f'{ds1}/demo_{demo_idx}/{fname}', f'{ds2}/demo_{demo_idx}/{fname}'
        if not os.path.exists(p1):
            missing += 1
            continue
        d1, d2 = np.load(p1, allow_pickle=True), np.load(p2, allow_pickle=True)
        demo_s.append(np.abs(d1['state'][0] - d2['state'][0]))
        demo_a.append(np.abs(d1['action'][0] - d2['action'][0]))
        demo_pc.append(np.abs(d1['point_cloud'][0] - d2['point_cloud'][0]))
        demo_gpc.append(np.abs(d1['gripper_pcd'][0] - d2['gripper_pcd'][0]))
        demo_goal_gpc.append(np.abs(d1['goal_gripper_pcd'][0] - d2['goal_gripper_pcd'][0]))
        total += 1

    ds, da = np.array(demo_s), np.array(demo_a)
    dpc, dgpc, dgoal = np.array(demo_pc), np.array(demo_gpc), np.array(demo_goal_gpc)
    tqdm.write(
        f'demo_{demo_idx:03d} | '
        f'STATE       max={ds.max():.3e} min={ds.min():.3e} mean={ds.mean():.3e} | '
        f'ACTION      max={da.max():.3e} min={da.min():.3e} mean={da.mean():.3e} | '
        f'POINT_CLOUD max={dpc.max():.3e} min={dpc.min():.3e} mean={dpc.mean():.3e} | '
        f'GRIPPER_PCD max={dgpc.max():.3e} min={dgpc.min():.3e} mean={dgpc.mean():.3e} | '
        f'GOAL_GRP_PC max={dgoal.max():.3e} min={dgoal.min():.3e} mean={dgoal.mean():.3e}'
    )
    all_s_diffs.extend(demo_s)
    all_a_diffs.extend(demo_a)
    all_pc_diffs.extend(demo_pc)
    all_gpc_diffs.extend(demo_gpc)
    all_goal_gpc_diffs.extend(demo_goal_gpc)

s, a = np.array(all_s_diffs), np.array(all_a_diffs)
pc, gpc, goal_gpc = np.array(all_pc_diffs), np.array(all_gpc_diffs), np.array(all_goal_gpc_diffs)
print(f'\n=== OVERALL ({total} timesteps, {missing} missing) ===')
print(f'STATE        -> max: {s.max():.3e}, min: {s.min():.3e}, mean: {s.mean():.3e}')
print(f'ACTION       -> max: {a.max():.3e}, min: {a.min():.3e}, mean: {a.mean():.3e}')
print(f'POINT_CLOUD  -> max: {pc.max():.3e}, min: {pc.min():.3e}, mean: {pc.mean():.3e}')
print(f'GRIPPER_PCD  -> max: {gpc.max():.3e}, min: {gpc.min():.3e}, mean: {gpc.mean():.3e}')
print(f'GOAL_GRP_PCD -> max: {goal_gpc.max():.3e}, min: {goal_gpc.min():.3e}, mean: {goal_gpc.mean():.3e}')
