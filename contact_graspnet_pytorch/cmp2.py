def build_6d_grasp(approach_dirs, base_dirs, contact_pts, thickness, use_torch=False, gripper_depth = 0.1034, device=None):
    """
    Build 6-DoF grasps + width from point-wise network predictions

    Arguments:
        approach_dirs {np.ndarray/torch.tensor} -- BxNx3 approach direction vectors
        base_dirs {np.ndarray/torch.tensor} -- BxNx3 base direction vectors
        contact_pts {np.ndarray/torch.tensor} -- BxNx3 contact points
        thickness {np.ndarray/torch.tensor} -- BxNx1 grasp width

    Keyword Arguments:
        use_tf {bool} -- whether inputs and outputs are tf tensors (default: {False})
        gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

    Returns:
        np.ndarray / torch.tensor -- BxNx4x4 grasp poses in camera coordinates
    """
    # We are trying to build a stack of 4x4 homogeneous transform matricies of size B x N x 4 x 4.
    # To do so, we calculate the rotation and translation portions according to the paper.
    # This gives us positions as shown:
    # [ R R R T ]
    # [ R R R T ]
    # [ R R R T ]
    # [ 0 0 0 1 ]                    Note that the ^ dim is 2 and the --> dim is 3
    # We need to pad with zeros and ones to get the final shape so we generate
    # ones and zeros and stack them.
    if thickness.ndim == 2:
        thickness = thickness.unsqueeze(2)  # B x N x 1
    if use_torch:
        if device is None:
            device = torch.device('cpu')
        grasp_R = torch.stack([base_dirs, torch.cross(approach_dirs,base_dirs),approach_dirs], dim=3)  # B x N x 3 x 3
        grasp_t = contact_pts + (thickness / 2) * base_dirs - gripper_depth * approach_dirs  # B x N x 3
        grasp_t = grasp_t.unsqueeze(3)  # B x N x 3 x 1
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32).to(device)  # B x N x 1 x 1
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32).to(device)  # B x N x 1 x 3
        homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
        grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4

    else:
        raise NotImplementedError("Need to test this more")
        grasps = []
        for i in range(len(contact_pts)):
            grasp = np.eye(4)

            grasp[:3,0] = base_dirs[i] / np.linalg.norm(base_dirs[i])
            grasp[:3,2] = approach_dirs[i] / np.linalg.norm(approach_dirs[i])
            grasp_y = np.cross( grasp[:3,2],grasp[:3,0])
            grasp[:3,1] = grasp_y / np.linalg.norm(grasp_y)
            # base_gripper xyz = contact + thickness / 2 * baseline_dir - gripper_d * approach_dir
            grasp[:3,3] = contact_pts[i] + thickness[i] / 2 * grasp[:3,0] - gripper_depth * grasp[:3,2]
            # grasp[0,3] = finger_width
            grasps.append(grasp)
        grasps = np.array(grasps)

    return grasps