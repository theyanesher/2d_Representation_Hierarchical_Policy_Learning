import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from io import BytesIO
from tqdm import tqdm

def visualize_exp(exp_path, save_path='./visualize.gif'):
    image_frames = []
    # 自动找到所有以数字命名的pkl文件，并排序
    file_names = sorted([f for f in os.listdir(exp_path) if f.endswith('.pkl') and f[:-4].isdigit()],
                        key=lambda x: int(x[:-4]))

    # 可视化每一帧
    with tqdm(total=len(file_names), desc="Processing frames") as pbar:
        for file_idx, file_name in enumerate(file_names):
            with open(os.path.join(exp_path, file_name), 'rb') as f:
                data = pickle.load(f)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # 读取数据
            pc = data['point_cloud'][0]
            gripper = data['gripper_pcd'][0]
            goal_gripper = data['goal_gripper_pcd'][0]

            # 可视化点云
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.5, c='blue', label='point_cloud')
            ax.scatter(gripper[:, 0], gripper[:, 1], gripper[:, 2], s=20, c='red', label='gripper_pcd')
            ax.scatter(goal_gripper[:, 0], goal_gripper[:, 1], goal_gripper[:, 2], s=20, c='green', label='goal_gripper_pcd')

            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(0, 1)
            ax.set_title(f'Frame: {file_idx + 1}/{len(file_names)}')
            ax.legend()

            # 保存当前帧为图像并加入 GIF
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_frames.append(imageio.imread(buf))
            buf.close()
            plt.close()
            pbar.update(1)

    # 保存为GIF
    imageio.mimsave(save_path, image_frames, duration=0.05)
    print(f"GIF saved to {save_path}")
    
dataset_prefix = './data/dp3_demo/seuss_gen/'
objects = [
    # bucket_tasks
    'bucket_100444',
    'bucket_100452',
    'bucket_100454',
    'bucket_100460',
    'bucket_100461',
    'bucket_100462',
    'bucket_100469',
    'bucket_100472',
    'bucket_102352',
    'bucket_102365',

    # faucet_tasks
    'faucet_148',
    'faucet_149',
    'faucet_152',
    'faucet_153',
    'faucet_154',
    'faucet_168',
    'faucet_811',
    'faucet_857',
    'faucet_960',
    'faucet_991',

    # foldingchair_tasks
    'foldingchair_100520',
    'foldingchair_100521',
    'foldingchair_100526',
    'foldingchair_100562',
    'foldingchair_100586',
    'foldingchair_100590',
    'foldingchair_100599',
    'foldingchair_102263',
    'foldingchair_102269',
    'foldingchair_102314',

    # laptop_tasks
    'laptop_9748',
    'laptop_9912',
    'laptop_9960',
    'laptop_9968',
    'laptop_9992',
    'laptop_9996',
    'laptop_10040',
    'laptop_10098',
    'laptop_10101',
    'laptop_10238',

    # stapler_tasks
    'stapler_103095',
    'stapler_103099',
    'stapler_103100',
    'stapler_103104',
    'stapler_103111',
    'stapler_103292',
    'stapler_103293',
    'stapler_103297',
    'stapler_103299',
    'stapler_103301',

    # toilet_tasks
    'toilet_101320',
    'toilet_102621',
    'toilet_102622',
    'toilet_102630',
    'toilet_102634',
    'toilet_102645',
    'toilet_102648',
    'toilet_102651',
    'toilet_102652',
    'toilet_102658',
]

for obj in objects:
    exp_folders = sorted(os.listdir(os.path.join(dataset_prefix, obj)))
    count = 0
    for folder in exp_folders:
        if folder.startswith('2025'):
            exp_path = os.path.join(dataset_prefix, obj, folder)
            save_path = os.path.join(exp_path, 'visualize.gif')
            print(f"Visualizing {exp_path}...")
            visualize_exp(exp_path, save_path)
            count += 1
            if count >= 2:
                break


