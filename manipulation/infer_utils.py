from manipulation.utils import build_up_env, save_numpy_as_gif, save_env
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
#from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
import torch
#from pytorch3d.transforms import Rotate
def construct_env(cfg, config_file, solution_path, task_name, init_state_file):
    env, _ = build_up_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    # render=False, 
                    render=False, 
                    randomize=False,
                    obj_id=0,
                    horizon=600,
                    random_object_translation=True
            )
            
    object_name = "StorageFurniture".lower()
    env.reset()
    print("POINT CLOUD MEAN CENTERED", cfg.task.env_runner.point_cloud_mean_centered)
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                num_points=cfg.task.env_runner.num_point_in_pc,
                                                use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                use_segmask=cfg.task.env_runner.use_segmask,
                                                only_handle_points=cfg.task.env_runner.only_handle_points,
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                dense_pcd_for_goal=cfg.task.env_runner.dense_pcd_for_goal,
                                                point_cloud_mean_centered=cfg.task.env_runner.point_cloud_mean_centered
                                                )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    return env





def symmetric_orthogonalization(M):
    """Maps arbitrary input matrices onto SO(3) via symmetric orthogonalization.
    (modified from https://github.com/amakadia/svd_for_pose)

    M: should have size [batch_size, 3, 3]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.det(torch.bmm(U, Vh)).view(-1, 1, 1)
    Vh = torch.cat((Vh[:, :2, :], Vh[:, -1:, :] * det), 1)
    R = U @ Vh
    #R = Vh.T @ U.T
    return R


def flow2pose(xyz, flow, weights=None, return_transform3d=False,
        return_quaternions=False, world_frameify=False):
    
    #flow = flow.cpu()
    #xyz = xyz[:, -1, :,:]
    #import pdb; pdb.set_trace()
    flow = torch.tensor(flow.reshape(1, -1, 3)).cuda()
    xyz = torch.tensor(xyz).cuda()
    #import pdb; pdb.set_trace()
    if weights is None:
        weights = torch.ones(xyz.shape[:-1], device=xyz.device)
    ww = (weights / weights.sum(dim=-1, keepdims=True)).unsqueeze(-1)

    # xyz_mean shape: ((B,N,1), (B,N,3)) mult -> (B,N,3) -> sum -> (B,1,3)
    xyz_mean = (ww * xyz).sum(dim=1, keepdims=True)
    xyz_demean = xyz - xyz_mean  # broadcast `xyz_mean`, still shape (B,N,3)

    # As with xyz positions, find (weighted) mean of flow, shape (B,1,3).
    flow_mean = (ww * flow).sum(dim=1, keepdims=True)

    # Zero-mean positions plus zero-mean flow to find new points.
    xyz_trans = xyz_demean + flow - flow_mean  # (B,N,3)

    # Batch matrix-multiply, get X: (B,3,3), each (3x3) matrix is in SO(3).
    X = torch.bmm(xyz_demean.transpose(-2,-1),  # (B,3,N)
                  ww * xyz_trans)               # (B,N,3)

    # Rotation matrix in SO(3) for each mb item, (B,3,3).
    R = symmetric_orthogonalization(X)
    R_ret = ((symmetric_orthogonalization(X).squeeze()).T).unsqueeze(0)

    # 3D translation vector for eacb mb item, (B,3) due to squeezing.
    if world_frameify:
        t = (flow_mean + xyz_mean - torch.bmm(xyz_mean, R)).squeeze(1)
    else:
        t = flow_mean.squeeze(1)

    '''if return_transform3d:
        return Rotate(R).translate(t), matrix_to_rotation_6d(R).squeeze(), t.squeeze()'''
    '''if return_quaternions:
        quats = matrix_to_quaternion(matrix=R)
        return quats, t'''
    #import pdb; pdb.set_trace()
    #return matrix_to_rotation_6d(R_ret).squeeze(), t.squeeze()
    return matrix_to_rotation_6d(R).squeeze(), t.squeeze()



def rotation_matrix_to_euler_angles(R):
    R = R.view(3, 3)  # Ensure it's a 3x3 matrix
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])  # Compute sy
    singular = sy < 1e-6  # Check if singularity is reached

    if not singular:
        # No singularity, can compute all three angles
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        # Handle singularity case (pitch = +/-90 degrees)
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0.0
    
    return torch.tensor([x, y, z])



def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def matrix_to_rotation_6d_numpy(matrix):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))

import torch.nn.functional as F

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    #import pdb; pdb.set_trace();
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2).cpu().numpy()