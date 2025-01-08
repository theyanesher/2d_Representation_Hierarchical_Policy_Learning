from test_scene_translation import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffuser_actor_3d.robogen_utils import get_gripper_pos_orient_from_4_points_torch, quaternion_to_rotation_matrix_torch
from manipulation.utils import rotation_transfer_matrix_to_6D, rotation_transfer_matrix_to_6D_batch

dataset = load_train_dataset()
dataloader = get_dataloader(dataset, batch_size=1)


def unbatchify(batch):
    for key in batch.keys():
        if isinstance(batch[key], dict):
            unbatchify(batch[key])
        else:
            batch[key] = batch[key][0]

def plot_pcds(pcds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'red', 'yellow', 'green']
    for color, pcd in zip(colors,pcds):
        ax.scatter(pcd[:,0],pcd[:,1],pcd[:,2],c=color)
    ax.view_init(elev=20, azim=180)
    plt.savefig('pcds.png')

def gripper_pcd_to_10d_vector(gripper_pcd, is_open=False):
    device = gripper_pcd.device
    all_representations = []
    for pcd in gripper_pcd:
        gripper_pos, gripper_orn = get_gripper_pos_orient_from_4_points_torch(pcd)
        vec_shape = tuple(gripper_pos.shape)
        vec_shape = (*vec_shape[:-1], 1)
        if is_open:
            grip_state = torch.zeros(vec_shape, device=device)
        else: 
            grip_state = torch.ones(vec_shape, device=device)
        gripper_rot_matrix = quaternion_to_rotation_matrix_torch(gripper_orn.cpu())
        gripper_6d_pose = rotation_transfer_matrix_to_6D_batch(gripper_rot_matrix)
        representation = torch.concatenate([gripper_pos.cuda(), gripper_6d_pose.cuda(), grip_state.cuda()], axis=-1)
        all_representations.append(representation)
    all_representations = torch.stack(all_representations)
    return all_representations

for batch in iter(dataloader):
    # batch [obs, keys]
    # obs ['point_cloud', 'agent_pos', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']
    # action Tensor[1,2,10]
    # unbatchify(batch)
    # pcds = [batch['obs']['point_cloud'][0],
    #         batch['obs']['gripper_pcd'][0],
    #         batch['obs']['goal_gripper_pcd'][0],
    #         batch['obs']['agent_pos'][[0],:3]]

    # plot_pcds(pcds)
    
    # xyz = batch['obs']['goal_gripper_pcd'][:,-1]
    # gripper_state = np.ones()# batch['action'][:,-1]
    # representation = get_gripper_pos_orient_from_4_points_torch(batch['obs']['goal_gripper_pcd'].to('cuda'))
    representation = gripper_pcd_to_10d_vector(batch['obs']['goal_gripper_pcd'].to('cuda'))
    breakpoint()
    