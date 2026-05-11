import h5py
from pathlib import Path
import torch
import third_party.robogen.goal_cond_diffpo_utils as utils
import numpy as np
import matplotlib.pyplot as plt

task = 'square_d2'
ROOT = Path('/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/')
# ROOT = Path('/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/robomimic/datasets')    
path = ROOT / task / f'{task}_pcd_abs_images_flow.hdf5'

agentview_intrinsics = torch.Tensor(np.loadtxt(
    'sandbox/mino/camera_calibrations/agentview_camera_intrinsics.txt'))
world_to_agentview_extrinsics = torch.Tensor(np.loadtxt(
    'sandbox/mino/camera_calibrations/agentview_world_to_cam_extrinsics.txt'))
eye_in_hand_intrinsics = torch.Tensor(np.loadtxt(
    'sandbox/mino/camera_calibrations/eye_in_hand_camera_intrinsics.txt'))


demo = 0
with h5py.File(path, 'r') as f:
    grp = f['data'][f'demo_{demo}']['obs']
    length = len(grp['point_cloud'])

    agentview_image_84 = grp['agentview_image_84'][:2]
    gripper_pcd = grp['gripper_pcd'][:2]
    goal_gripper_pcd = grp['goal_gripper_pcd'][:2]

    pt_agentview_image_84 = torch.tensor(agentview_image_84, dtype=torch.float32).unsqueeze(0)
    pt_gripper_pcd = torch.tensor(gripper_pcd, dtype=torch.float32).unsqueeze(0)
    pt_goal_gripper_pcd = torch.tensor(goal_gripper_pcd, dtype=torch.float32).unsqueeze(0)

    pt_agentview_image_84 = pt_agentview_image_84.permute(0, 1, 4, 2, 3)  # Change to (B, C, H, W)
    
    ###############################
    # Test pytorch_project_points #
    ###############################
    coords = utils.project_points(gripper_pcd, 
                                  agentview_intrinsics.numpy(), 
                                  world_to_agentview_extrinsics.numpy(),
                                  normalization_factor=(84./512.))

    pt_coords = utils.pytorch_project_points(pt_gripper_pcd, 
                                             agentview_intrinsics,
                                             world_to_agentview_extrinsics,
                                             normalization_factor=(84./512.))
    
    print(np.allclose(coords, pt_coords.unsqueeze(0).numpy(), atol=1e-6))


    ###################################
    # Test pytorch_coords_to_2d_image #
    ###################################

    mask = utils.coords_to_2d_image(coords, agentview_image_84)
    pt_mask = utils.pytorch_coords_to_2d_image(pt_coords, pt_agentview_image_84)
    pt_mask = pt_mask[0,:,-1]
    print(np.allclose(mask, pt_mask.numpy(), atol=1e-6))

    print(mask.shape)

    ##################################
    # visualize the projected points #
    ##################################

    fig, axs = plt.subplots(1, 3, figsize=(8, 8))
    axs[0].imshow(mask[0])
    axs[1].imshow(pt_mask.numpy()[0])
    axs[2].imshow(mask[0] + pt_mask.numpy()[0])
    plt.show()    

    ########################################
    # Test coords_to_2d_image_displacement #
    ########################################
    gripper_coords = utils.project_points(gripper_pcd, 
                                  agentview_intrinsics.numpy(), 
                                  world_to_agentview_extrinsics.numpy(),
                                  normalization_factor=(84./512.),
                                  return_2d=False)
    goal_gripper_coords = utils.project_points(goal_gripper_pcd, 
                                  agentview_intrinsics.numpy(), 
                                  world_to_agentview_extrinsics.numpy(),
                                  normalization_factor=(84./512.), 
                                  return_2d=False)
    
    flow_mask = utils.coords_to_2d_image_displacements(goal_gripper_coords, agentview_image_84, gripper_coords)
    print(flow_mask.shape)

    pt_gripper_coords = utils.pytorch_project_points(pt_gripper_pcd, 
                                            agentview_intrinsics,
                                            world_to_agentview_extrinsics,
                                            normalization_factor=(84./512.),
                                            return_2d=False)
    pt_goal_gripper_coords = utils.pytorch_project_points(pt_goal_gripper_pcd, 
                                            agentview_intrinsics,
                                            world_to_agentview_extrinsics,
                                            normalization_factor=(84./512.),
                                            return_2d=False)

    pt_flow_mask = utils.pytorch_coords_to_2d_image_displacements(pt_goal_gripper_coords,
                                                                pt_agentview_image_84,
                                                                pt_gripper_coords)
    pt_flow_mask = pt_flow_mask[0].permute(0,2,3,1).numpy()
    print(pt_flow_mask.shape)
    
    print(np.allclose(flow_mask, pt_flow_mask, atol=1e-6))


    ########################
    # Visualize flow masks #
    ########################

    fig, axs = plt.subplots(3, 4, figsize=(8, 8))
    axs[0,0].imshow(flow_mask[0, :, :, 0])
    axs[0,1].imshow(flow_mask[0, :, :, 1])
    axs[0,2].imshow(flow_mask[1, :, :, 0])
    axs[0,3].imshow(flow_mask[1, :, :, 1])
    axs[1,0].imshow(pt_flow_mask[0, :, :, 0])
    axs[1,1].imshow(pt_flow_mask[0, :, :, 1])
    axs[1,2].imshow(pt_flow_mask[1, :, :, 0])
    axs[1,3].imshow(pt_flow_mask[1, :, :, 1])
    axs[2,0].imshow(pt_flow_mask[0, :, :, 0] - flow_mask[0, :, :, 0])
    axs[2,1].imshow(pt_flow_mask[0, :, :, 1] - flow_mask[0, :, :, 1])
    axs[2,2].imshow(pt_flow_mask[1, :, :, 0] - flow_mask[1, :, :, 0])
    axs[2,3].imshow(pt_flow_mask[1, :, :, 1] - flow_mask[1, :, :, 1])
    
    plt.show() 