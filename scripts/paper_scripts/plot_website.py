import open3d as o3d
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import torch
import os
from cem_policy.utils import save_numpy_as_gif
from diffuser_actor_3d.robogen_utils import get_gripper_pos_orient_from_4_points, quaternion_to_rotation_matrix
from manipulation.utils import rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from scipy.spatial.transform import Rotation as R

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])
    
    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05
 
    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
 
    # goal_R = R.from_quat(gripper_orn)
    # import pdb; pdb.set_trace()
    goal_R = R.from_matrix(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd

def get_new_gripper_pcd(cur_gripper_pcd, low_level_action):
    gripper_pos, gripper_orient = get_gripper_pos_orient_from_4_points(cur_gripper_pcd)
    current_rotate_matrix = quaternion_to_rotation_matrix(gripper_orient)

    delta_pos = low_level_action[:3]
    delta_orient = low_level_action[3:9]
    delta_rotate_matrix = rotation_transfer_6D_to_matrix(delta_orient)
    
    after_rotate_matrix = current_rotate_matrix @ delta_rotate_matrix
    new_pos = gripper_pos + delta_pos
    
    # import pdb; pdb.set_trace()
    cur_finger_distance = np.clip(np.linalg.norm(cur_gripper_pcd[1] - cur_gripper_pcd[2]) / 2, 0, 0.04)
    new_finger_distance = cur_finger_distance + low_level_action[9]
    
    new_pcd = get_4_points_from_gripper_pos_orient(new_pos, after_rotate_matrix, new_finger_distance)
    return new_pcd


def adjust_axis_range(ax, pcd, xrange=None, yrange=None, zrange=None):
    
    if xrange is None:
        # Extract min and max for each axis
        x_min, x_max = np.min(pcd[:, 0]), np.max(pcd[:, 0])
        y_min, y_max = np.min(pcd[:, 1]), np.max(pcd[:, 1])
        z_min, z_max = np.min(pcd[:, 2]), np.max(pcd[:, 2])

        # Compute the center and max range
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        max_range = max(x_range, y_range, z_range) / 2.5 # Half of the largest range
        mid_x = (x_max + x_min) / 2.0
        mid_y = (y_max + y_min) / 2.0
        mid_z = (z_max + z_min) / 2.0
        
        ax.set_xlim([mid_x - max_range, mid_x + max_range])
        ax.set_ylim([mid_y - max_range, mid_y + max_range])
        ax.set_zlim([mid_z - max_range, mid_z + max_range])
        return ax, [mid_x - max_range, mid_x + max_range], [mid_y - max_range, mid_y + max_range], [mid_z - max_range, mid_z + max_range]
    else:
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.set_zlim(zrange)
    
        return ax


### load data
### Chialiang's data
data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat/2025-01-13-20-32-49-formal4"
save_name = 'chialiang-green'

### mobile base data
# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-19-21-38-14-good-20250122T024000Z-001/2025-01-19-21-38-14-good"
# save_anme = 'mobile_base_fridge'

### our lab
# knob drawer
# data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0112-paper-knob-cabinet/2025-01-12-18-17-51-trial-2"
# save_name = 'knob_cabinet'

data_paths = [
    # "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat/2025-01-13-20-32-49-formal4",
    # "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-19-21-38-14-good-20250122T024000Z-001/2025-01-19-21-38-14-good",
    # "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0112-paper-knob-cabinet/2025-01-12-18-17-51-trial-2",
    # "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/mobile-x-arm-raw-data/0129-test-nsh-4-cabinet-good-part-2/2025-01-29-22-03-16-good",
    # "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/mobile-x-arm-raw-data/0127-test-nsh-robo-drawer-good/2025-01-28-00-17-34",
    # "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0112-paper-white-cabinet/2025-01-12-16-53-05-trial-1",
    # "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/chialiang_results/formal-high-8_low-96-grey_drawer/2025-01-15-21-58-52-formal2",
    "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0110-paper-toy-oven/2025-01-10-18-15-07-oven-tiral-1",
]

save_names = [
    # "chialiang-green",
    # "mobile_base_fridge",
    # "knob_cabinet",
    # "nsh4_cabinet",
    # "robo_lounge_drawer",
    # "white_cabinet",
    # "chialiang-drawer",
    "oven",
]

for data_path, save_name in zip(data_paths, save_names):
    print("data_path", data_path)
    print("save_name", save_name)
    all_step_files = os.listdir(data_path)
    all_step_files = [x for x in all_step_files if ".pkl" in x]
    num_step = len(all_step_files) 

    images = []
    for step in range(1, num_step + 1):
        
        pkl_path = os.path.join(data_path, f"step_{step}.pkl")
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)
            
        high_level_input_dict = data["high_level_input_dict"]

        high_level_pcd = high_level_input_dict["high_level_point_cloud"]    
        high_level_gripper_pcd = high_level_input_dict["high_level_gripper_pcd"]
        low_level_actions_pred = data["low_level_outputs"]['action_pred'].squeeze(0).cpu().detach().numpy()

        elev = 8
        azim = 131
        obj_pcd_np = high_level_pcd.squeeze(0)[-1].cpu().detach().numpy()
        if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat/2025-01-13-20-32-49-formal4":
            elev = 24
            azim = -112
        if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-19-21-38-14-good-20250122T024000Z-001/2025-01-19-21-38-14-good":
            obj_pcd_np = obj_pcd_np[obj_pcd_np[:, 0] < 1]
            elev = 14
            azim = 151
        if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0112-paper-knob-cabinet/2025-01-12-18-17-51-trial-2":
            elev = 8
            azim = 131
        if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/mobile-x-arm-raw-data/0127-test-nsh-robo-drawer-good/2025-01-28-00-17-34":
            elev = 40
            azim = -169
        if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0112-paper-white-cabinet/2025-01-12-16-53-05-trial-1":
            elev = 15
            azim = 145
            
        gripper_pcd_np = high_level_gripper_pcd.squeeze(0)[-1].cpu().detach().numpy()

        current_gripper_pos = gripper_pcd_np[-1]
        future_gripper_points = [gripper_pcd_np]
        for t in range(8):
            new_gripper_pcd = get_new_gripper_pcd(future_gripper_points[-1], low_level_actions_pred[t])
            future_gripper_points.append(new_gripper_pcd)

        for t in range(4):
            low_level_action = low_level_actions_pred[t]
            fig  = plt.figure(figsize=(24, 8))
            ax = fig.add_subplot(141)
            rgb_img = data["rgbs"][t]
            if data_path in ["/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0112-paper-knob-cabinet/2025-01-12-18-17-51-trial-2",
                            "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0112-paper-white-cabinet/2025-01-12-16-53-05-trial-1",
                            "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/franka-arm-real-world-raw-data/0110-paper-toy-oven/2025-01-10-18-15-07-oven-tiral-1",
                            ]:

                h, w, _ = rgb_img.shape
                rgb_img = rgb_img[:, 150:w//2]
                
            ax.imshow(rgb_img)
            ax.axis('off')
            ax.set_title(f"RGB camera step {step * 4 + t}", fontsize=18)

            ### plot the input
            ax = fig.add_subplot(142, projection='3d')
            # x_value = obj_pcd_np[:, 0]
            # ## normalize the x_value
            # x_value = (x_value - x_value.min()) / (x_value.max() - x_value.min())
            # colors = plt.cm.viridis(x_value)
            down_sample_idx = np.arange(0, len(obj_pcd_np), 1)
            # ax = plt.axes(projection='3d')
            # ax.scatter(obj_pcd_np[down_sample_idx,0], obj_pcd_np[down_sample_idx,1], obj_pcd_np[down_sample_idx,2], c=colors, s=4)
            ax.scatter(obj_pcd_np[down_sample_idx,0], obj_pcd_np[down_sample_idx,1], obj_pcd_np[down_sample_idx,2], c="grey", s=4)
            ax.scatter(gripper_pcd_np[:,0], gripper_pcd_np[:,1], gripper_pcd_np[:,2], c='green', s=50)
            if step == 1:
                ax, xrange, yrange, zrange = adjust_axis_range(ax, np.concatenate([obj_pcd_np, gripper_pcd_np], axis=0))
            else:
                ax = adjust_axis_range(ax, np.concatenate([obj_pcd_np, gripper_pcd_np], axis=0), xrange=xrange, yrange=yrange, zrange=zrange)
            ax.view_init(elev, azim)

            # ax.axis('equal')
            ax.axis('off')
            ax.set_title(f"Policy observation (point cloud)\nGrey: obj pcd\nGreen: current eef points", fontsize=18)

            ### load model
            load_model_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/model_8.pth"
            # load_model_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/debug/model_36.pth"
            from test_PointNet2.model_invariant import PointNet2_super
            pointnet2_model = PointNet2_super(num_classes=13).to("cuda")
            pointnet2_model.load_state_dict(torch.load(load_model_path))
            pointnet2_model.eval()
            goal_policy = pointnet2_model

            ### get model prediction
            pointcloud = high_level_pcd[0, -1].to("cuda").unsqueeze(0)
            # if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat/2025-01-13-20-32-49-formal4" and step == 8:
            #     pointcloud = pointcloud[:, pointcloud[0, :, 0] > 0.59, :]
            if data_path == "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e52/yufei/projects/RoboGen-sim2real/data/real_world/2025-01-19-21-38-14-good-20250122T024000Z-001/2025-01-19-21-38-14-good":
                pointcloud = pointcloud[:, pointcloud[0, :, 0] < 1, :]
            gripper_pcd = high_level_gripper_pcd[0, -1].to("cuda").unsqueeze(0)
            inputs = torch.cat([pointcloud, gripper_pcd], dim=1) # B, N+4, 3, B=1

            inputs_ = inputs.permute(0, 2, 1)
            outputs = goal_policy(inputs_) # B, N, 13
            weights = outputs[:, :, -1] # B, N
            outputs = outputs[:, :-4, :-1] # B, N, 12
            weights = weights[:, :-4]
            weights = torch.nn.functional.softmax(weights, dim=1)

            B, N, _ = outputs.shape
            outputs = outputs.view(B, N, 4, 3)
            prediction = outputs + inputs[:, :-4, :3].unsqueeze(2)
            prediction = prediction * weights.unsqueeze(-1).unsqueeze(-1)
            prediction = prediction.sum(dim=1)
            prediction = prediction.unsqueeze(1)

            weights_numpy = weights.squeeze(0).cpu().detach().numpy()
            print("weights_numpy", weights_numpy.shape)
            outputs_numpy = outputs.squeeze(0).cpu().detach().numpy()
            print("outputs_numpy", outputs_numpy.shape)
            prediction_numpy = prediction.squeeze(0).squeeze(0).cpu().detach().numpy()
            print("prediction_numpy", prediction_numpy.shape)




            ### plot the weight
            ax = fig.add_subplot(143, projection='3d')
            # x_color = colors
            # weight_color = plt.cm.seismic(weights_numpy)
            # final_color = x_color * 0.5 + weight_color * 0.5
            ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c=weights_numpy, s=10,  cmap='seismic')
            ax = adjust_axis_range(ax, obj_pcd_np, xrange=xrange, yrange=yrange, zrange=zrange)
            ax.view_init(elev, azim)
            # ax.axis('equal')
            ax.axis('off')
            ax.set_title(f"High-level policy predicted weights", fontsize=18)

            ### plot the final preidction
            # ax = fig.add_subplot(154, projection='3d')
            # # ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c=colors, s=1, zorder=1, depthshade=True)
            # ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c="grey", s=1, zorder=1, depthshade=False)
            # ax.scatter(gripper_pcd_np[:,0], gripper_pcd_np[:,1], gripper_pcd_np[:,2], c='green', s=50)
            # ax.scatter(prediction_numpy[:, 0], prediction_numpy[:, 1], prediction_numpy[:, 2], c='red', s=50, zorder=10, depthshade=False)
            # ax = adjust_axis_range(ax, np.concatenate([obj_pcd_np, gripper_pcd_np, prediction_numpy], axis=0), xrange=xrange, yrange=yrange, zrange=zrange)
            # ax.view_init(elev, azim)
            # ax.set_title(f"high-level policy predicted goal eef points")
            # # ax.axis('equal')
            # ax.axis('off')
            
            ### plot the low-level prediction
            ax = fig.add_subplot(144, projection='3d')
            # ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c=colors, s=1, zorder=1, depthshade=True)
            new_pcd = future_gripper_points[t]
            ax.scatter(obj_pcd_np[:,0], obj_pcd_np[:,1], obj_pcd_np[:,2], c="grey", s=1, zorder=1, depthshade=False)
            ax.scatter(new_pcd[:,0], new_pcd[:,1], new_pcd[:,2], c='green', s=50)
            # TODO: compute the new gripper pcd
            ax.scatter(prediction_numpy[:, 0], prediction_numpy[:, 1], prediction_numpy[:, 2], c='red', s=50, zorder=10, depthshade=False)
            ax.plot([x[-1, 0] for x in future_gripper_points], [x[-1, 1] for x in future_gripper_points], [x[-1, 2] for x in future_gripper_points], c='blue', linewidth=2)
            ax = adjust_axis_range(ax, np.concatenate([obj_pcd_np, gripper_pcd_np, prediction_numpy], axis=0), xrange=xrange, yrange=yrange, zrange=zrange)
            ax.view_init(elev, azim)
            # ax.axis('equal')
            ax.axis('off')
            ax.set_title(f"High-level policy predicted goal eef points (red)\nLow-level policy predicted actions (blue line)", fontsize=18)
            
        
            
            fig.tight_layout()
            # plt.show()
            
            fig.canvas.draw()

            # Convert canvas to an RGB image
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape((height, width, 3))
            images.append(image)
            
            plt.close("all")
            # plt.imshow(image)
            

    ### save the images as a mp4
    save_numpy_as_gif(np.array(images), f"data/real_world/{save_name}.mp4", fps=6)

