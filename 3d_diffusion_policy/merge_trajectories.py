import os, glob
import numpy as np
import open3d as o3d
import copy
import pickle
from termcolor import cprint
from tqdm import tqdm

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

def rotation_transfer_6D_to_matrix_batch(orient):

    # orient shape = (B, 6)
    # return shape = (3, B * 3)

    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)
    
    assert orient.shape[-1] == 6

    orient = orient.reshape(-1, 2, 3)
    a1 = orient[:,0]
    a2 = orient[:,1]

    b1 = a1 / np.linalg.norm(a1, axis=-1).reshape(-1,1)
    b2 = a2 - (np.sum(a2*b1, axis=-1).reshape(-1,1) * b1)
    b2 = b2 / np.linalg.norm(b2, axis=-1).reshape(-1,1)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.hstack((b1, b2, b3))
    rotate_matrix = rotate_matrix.reshape(-1, 3).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D_batch(rotate_matrix):

    # rotate_matrix.shape = (B, 9) or (B, 3, 3)
    # return shape = (B, 6)

    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(-1, 9)
    rotate_matrix = rotate_matrix.reshape(-1, 9)

    return rotate_matrix[:,:6]

def add_pose(pose, size=0.1):

    assert len(pose) == 10

    trans = np.identity(4)
    trans[:3, 3] = pose[:3]
    trans[:3,:3] = rotation_transfer_6D_to_matrix(pose[3:9])
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coor.transform(trans)

    return coor

def add_sphere(pts, color=[1, 0, 0], radius=0.02):
    rets = []
    for pt in pts:
        pt_coor = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        pt_coor.translate(pt.reshape(-1, 1))
        pt_coor.paint_uniform_color(color)
        rets.append(pt_coor)

    return rets

def add_pcd(pts, color=[0,0,1]):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.full(pts.shape, color))

    return pcd

def combine_action(src_dir='one_traj/45526_pkl', dst_dir='one_traj/45526_pkl_post', combine_step=2, dense_steps_around_goal=3):

    assert os.path.exists(src_dir), f'{src_dir} not exists'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    
    pkl_dirs = glob.glob(f'{src_dir}/*')

    for pkl_dir in tqdm(pkl_dirs):

        traj_name = pkl_dir.split('/')[-1]

        wpt_groups = os.listdir(pkl_dir)
        wpt_groups = sorted(wpt_groups, key = lambda x: int(x.split('.')[0]))
        traj_length = len(wpt_groups)

        pickle_path = f'{pkl_dir}/{wpt_groups[0]}'
        data = pickle.load(open(pickle_path, 'rb'))
        original_goal_gripper_pcd = data['goal_gripper_pcd']
        
        goal_switch_point = 0
        goal_pcd = copy.deepcopy(original_goal_gripper_pcd[0])

        # find goal switching point and save raw data
        raw_wpt_data = [data]
        for wpt_id, wpt_group in enumerate(wpt_groups[1:]):

            pickle_path = f'{pkl_dir}/{wpt_group}'
            data = pickle.load(open(pickle_path, 'rb'))
            raw_wpt_data.append(data)

            if goal_switch_point == 0 and np.linalg.norm((data['goal_gripper_pcd'][0] - goal_pcd)) > 1e-3:
                goal_switch_point = wpt_id
        
        # combine action
        saved_data_list = []
        run_index = 0
        for wpt_id in range(traj_length):
            
            # keep dense steps around goal
            if (wpt_id > goal_switch_point - dense_steps_around_goal) and (wpt_id < goal_switch_point + dense_steps_around_goal):
                
                run_index = wpt_id
                
                saved_data_list.append({
                    'state': raw_wpt_data[wpt_id]['state'],
                    'point_cloud': raw_wpt_data[wpt_id]['point_cloud'],
                    'gripper_pcd': raw_wpt_data[wpt_id]['gripper_pcd'],
                    'goal_gripper_pcd': raw_wpt_data[wpt_id]['goal_gripper_pcd'],
                    'displacement_gripper_to_object': raw_wpt_data[wpt_id]['displacement_gripper_to_object']
                })

                if len(saved_data_list) <= 1:
                    continue
                
                last_state = saved_data_list[-2]['state']
                current_state = saved_data_list[-1]['state']

                last_rot_mat = rotation_transfer_6D_to_matrix(last_state[0, 3:9])
                current_rot_mat = rotation_transfer_6D_to_matrix(current_state[0, 3:9])
                delta_rot_mat = last_rot_mat.T @ current_rot_mat
                delta_rot_6d = rotation_transfer_matrix_to_6D(delta_rot_mat)

                action = np.zeros((1, 10))
                action[0, :3] = current_state[0, :3] - last_state[0, :3]
                action[0, 3:9] = delta_rot_6d
                action[0, 9] = current_state[0, 9] - last_state[0, 9]

                # last action will be computed in the current state
                saved_data_list[-2]['action'] = action
            
            elif wpt_id % combine_step == 0:

                run_index = wpt_id
                
                saved_data_list.append({
                    'state': raw_wpt_data[wpt_id]['state'],
                    'point_cloud': raw_wpt_data[wpt_id]['point_cloud'],
                    'gripper_pcd': raw_wpt_data[wpt_id]['gripper_pcd'],
                    'goal_gripper_pcd': raw_wpt_data[wpt_id]['goal_gripper_pcd'],
                    'displacement_gripper_to_object': raw_wpt_data[wpt_id]['displacement_gripper_to_object']
                })

                if len(saved_data_list) <= 1:
                    continue

                last_state = saved_data_list[-2]['state']
                current_state = saved_data_list[-1]['state']

                last_rot_mat = rotation_transfer_6D_to_matrix(last_state[0, 3:9])
                current_rot_mat = rotation_transfer_6D_to_matrix(current_state[0, 3:9])
                delta_rot_mat = last_rot_mat.T @ current_rot_mat
                delta_rot_6d = rotation_transfer_matrix_to_6D(delta_rot_mat)

                action = np.zeros((1, 10))
                action[0, :3] = current_state[0, :3] - last_state[0, :3]
                action[0, 3:9] = delta_rot_6d
                action[0, 9] = current_state[0, 9] - last_state[0, 9]

                # last action will be computed in the current state
                saved_data_list[-2]['action'] = action

        # for last action only
        last_action = raw_wpt_data[run_index]['action']

        last_delta_rot_mat = rotation_transfer_6D_to_matrix(last_action[0, 3:9])
        for last_few_wpt_id in range(run_index + 1, traj_length):
            
            # position
            last_action[0, :3] += raw_wpt_data[last_few_wpt_id]['action'][0, :3]

            # rotation
            current_delta_rot_mat = rotation_transfer_6D_to_matrix(raw_wpt_data[last_few_wpt_id]['action'][0, 3:9])
            last_delta_rot_mat = last_delta_rot_mat @ current_delta_rot_mat

            # gripper
            last_action[0, 9] += raw_wpt_data[last_few_wpt_id]['action'][0, 9]
            
        last_action[0, 3:9] = rotation_transfer_matrix_to_6D(last_delta_rot_mat)
        saved_data_list[-1]['action'] = last_action
        
        # save data
        for new_wpt_id, saved_data in enumerate(saved_data_list):
            target_dir = f'{dst_dir}/{traj_name}'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            new_pickle_data_save_path = f'{target_dir}/{new_wpt_id}.pkl'
            with open(new_pickle_data_save_path, "wb") as f:
                pickle.dump(saved_data, f)
                cprint(f"Saved new data to: {new_pickle_data_save_path}", "green")

if __name__=="__main__":

    src_dirs = ['one_traj/45305_pkl']
    dst_dirs = ['one_traj/45305_pkl_post']
    src_dirs = [
        "/scratch/chialiang/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point",
        "/scratch/chialiang/dp3_demo/0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action",
        "/scratch/chialiang/dp3_demo/0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0628-act3d-obj-48700-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo/0705-obj-45526",
        "/scratch/chialiang/dp3_demo/0705-obj-45661",
        "/scratch/chialiang/dp3_demo/0705-obj-45694",
        "/scratch/chialiang/dp3_demo/0705-obj-45780",
        "/scratch/chialiang/dp3_demo/0705-obj-45910",
        "/scratch/chialiang/dp3_demo/0705-obj-45961",
        "/scratch/chialiang/dp3_demo/0705-obj-46408",
        "/scratch/chialiang/dp3_demo/0705-obj-46417",
        "/scratch/chialiang/dp3_demo/0705-obj-46440",
        "/scratch/chialiang/dp3_demo/0705-obj-46490",
        "/scratch/chialiang/dp3_demo/0705-obj-46762",
        "/scratch/chialiang/dp3_demo/0705-obj-46825",
        "/scratch/chialiang/dp3_demo/0705-obj-46893",
        "/scratch/chialiang/dp3_demo/0705-obj-47235",
        "/scratch/chialiang/dp3_demo/0705-obj-47281",
        "/scratch/chialiang/dp3_demo/0705-obj-47315",
        "/scratch/chialiang/dp3_demo/0705-obj-47529",
        "/scratch/chialiang/dp3_demo/0705-obj-47669",
        "/scratch/chialiang/dp3_demo/0705-obj-47944",
        "/scratch/chialiang/dp3_demo/0705-obj-48063",
        "/scratch/chialiang/dp3_demo/0705-obj-48177",
        "/scratch/chialiang/dp3_demo/0705-obj-48356",
        "/scratch/chialiang/dp3_demo/0705-obj-48623",
        "/scratch/chialiang/dp3_demo/0705-obj-48876",
        "/scratch/chialiang/dp3_demo/0705-obj-49025",
        "/scratch/chialiang/dp3_demo/0705-obj-49062",
        "/scratch/chialiang/dp3_demo/0705-obj-49132",
        "/scratch/chialiang/dp3_demo/0705-obj-49133",
        "/scratch/chialiang/dp3_demo/0712-obj-40417",
        "/scratch/chialiang/dp3_demo/0712-obj-41085",
        "/scratch/chialiang/dp3_demo/0712-obj-41452",
        "/scratch/chialiang/dp3_demo/0712-obj-45162",
        "/scratch/chialiang/dp3_demo/0712-obj-45176",
        "/scratch/chialiang/dp3_demo/0712-obj-45194",
        "/scratch/chialiang/dp3_demo/0712-obj-45203",
        "/scratch/chialiang/dp3_demo/0712-obj-45248",
        "/scratch/chialiang/dp3_demo/0712-obj-45271",
        "/scratch/chialiang/dp3_demo/0712-obj-45290",
        "/scratch/chialiang/dp3_demo/0712-obj-45305",
        "/scratch/chialiang/dp3_demo/0725-obj-45413",
        "/scratch/chialiang/dp3_demo/0725-obj-45420",
        "/scratch/chialiang/dp3_demo/0725-obj-45427",
        "/scratch/chialiang/dp3_demo/0725-obj-45594",
        "/scratch/chialiang/dp3_demo/0725-obj-45620",
        "/scratch/chialiang/dp3_demo/0725-obj-45623",

        # g2
        "/scratch/chialiang/dp3_demo/0725-obj-45636",
        "/scratch/chialiang/dp3_demo/0725-obj-45670",
        "/scratch/chialiang/dp3_demo/0725-obj-45689",
        "/scratch/chialiang/dp3_demo/0725-obj-45696",
        "/scratch/chialiang/dp3_demo/0725-obj-45749",
        "/scratch/chialiang/dp3_demo/0725-obj-45759",
        "/scratch/chialiang/dp3_demo/0725-obj-45916",
        "/scratch/chialiang/dp3_demo/0725-obj-45936",
        "/scratch/chialiang/dp3_demo/0725-obj-45950",
        "/scratch/chialiang/dp3_demo/0725-obj-45984",
        "/scratch/chialiang/dp3_demo/0725-obj-46092",
        "/scratch/chialiang/dp3_demo/0725-obj-46130",
        "/scratch/chialiang/dp3_demo/0725-obj-46134",
        "/scratch/chialiang/dp3_demo/0725-obj-46197",
        "/scratch/chialiang/dp3_demo/0725-obj-46401",
        "/scratch/chialiang/dp3_demo/0725-obj-46456",
        "/scratch/chialiang/dp3_demo/0725-obj-46480",
        "/scratch/chialiang/dp3_demo/0725-obj-46481",
        "/scratch/chialiang/dp3_demo/0725-obj-46544",
        "/scratch/chialiang/dp3_demo/0725-obj-46641",
        "/scratch/chialiang/dp3_demo/0725-obj-47178",
        "/scratch/chialiang/dp3_demo/0725-obj-47182",
        "/scratch/chialiang/dp3_demo/0725-obj-47227",
        "/scratch/chialiang/dp3_demo/0725-obj-47577",
        "/scratch/chialiang/dp3_demo/0725-obj-47648",
        "/scratch/chialiang/dp3_demo/0725-obj-47747",
        "/scratch/chialiang/dp3_demo/0725-obj-47808",
        "/scratch/chialiang/dp3_demo/0725-obj-47976",
        "/scratch/chialiang/dp3_demo/0725-obj-48010",
        "/scratch/chialiang/dp3_demo/0725-obj-48258",
        "/scratch/chialiang/dp3_demo/0725-obj-48797",
        "/scratch/chialiang/dp3_demo/0725-obj-48379",
        "/scratch/chialiang/dp3_demo/0725-obj-48859",
        "/scratch/chialiang/dp3_demo/0725-obj-48855",
        "/scratch/chialiang/dp3_demo/0730-obj-35059",
        "/scratch/chialiang/dp3_demo/0725-obj-49188",
        "/scratch/chialiang/dp3_demo/0730-obj-41083",
        "/scratch/chialiang/dp3_demo/0730-obj-41004",
        "/scratch/chialiang/dp3_demo/0730-obj-44781",
        "/scratch/chialiang/dp3_demo/0730-obj-41529",
        "/scratch/chialiang/dp3_demo/0730-obj-44853",
        "/scratch/chialiang/dp3_demo/0730-obj-44826",
        "/scratch/chialiang/dp3_demo/0730-obj-45130",
        "/scratch/chialiang/dp3_demo/0730-obj-45092",
        "/scratch/chialiang/dp3_demo/0730-obj-45146",
        "/scratch/chialiang/dp3_demo/0730-obj-45135",
        "/scratch/chialiang/dp3_demo/0730-obj-45168",
        "/scratch/chialiang/dp3_demo/0730-obj-45164",
        "/scratch/chialiang/dp3_demo/0730-obj-45212",
        "/scratch/chialiang/dp3_demo/0730-obj-45173",
        "/scratch/chialiang/dp3_demo/0730-obj-45372",
        "/scratch/chialiang/dp3_demo/0730-obj-45213",
        "/scratch/chialiang/dp3_demo/0730-obj-45387",
        "/scratch/chialiang/dp3_demo/0730-obj-45374",
        "/scratch/chialiang/dp3_demo/0730-obj-45419",

        # g3
        "/scratch/chialiang/dp3_demo/0730-obj-45415",
        "/scratch/chialiang/dp3_demo/0730-obj-45503",
        "/scratch/chialiang/dp3_demo/0730-obj-45423",
        "/scratch/chialiang/dp3_demo/0730-obj-45524",
        "/scratch/chialiang/dp3_demo/0730-obj-45505",
        "/scratch/chialiang/dp3_demo/0730-obj-45575",
        "/scratch/chialiang/dp3_demo/0730-obj-45573",
        "/scratch/chialiang/dp3_demo/0730-obj-45612",
        "/scratch/chialiang/dp3_demo/0730-obj-45606",
        "/scratch/chialiang/dp3_demo/0730-obj-45622",
        "/scratch/chialiang/dp3_demo/0730-obj-45621",
        "/scratch/chialiang/dp3_demo/0730-obj-45638",
        "/scratch/chialiang/dp3_demo/0730-obj-45632",
        "/scratch/chialiang/dp3_demo/0730-obj-45662",
        "/scratch/chialiang/dp3_demo/0730-obj-45645",
        "/scratch/chialiang/dp3_demo/0730-obj-45676",
        "/scratch/chialiang/dp3_demo/0730-obj-45671",
        "/scratch/chialiang/dp3_demo/0730-obj-45687",
        "/scratch/chialiang/dp3_demo/0730-obj-45677",
        "/scratch/chialiang/dp3_demo/0730-obj-45710",
        "/scratch/chialiang/dp3_demo/0730-obj-45699",
        "/scratch/chialiang/dp3_demo/0730-obj-45756",
        "/scratch/chialiang/dp3_demo/0730-obj-45746",
        "/scratch/chialiang/dp3_demo/0730-obj-45784",
        "/scratch/chialiang/dp3_demo/0730-obj-45783",
        "/scratch/chialiang/dp3_demo/0730-obj-45801",
        "/scratch/chialiang/dp3_demo/0730-obj-45790",
        "/scratch/chialiang/dp3_demo/0730-obj-45853",
        "/scratch/chialiang/dp3_demo/0730-obj-45822",
        "/scratch/chialiang/dp3_demo/0730-obj-45915",
        "/scratch/chialiang/dp3_demo/0730-obj-45855",
        "/scratch/chialiang/dp3_demo/0730-obj-45949",
        "/scratch/chialiang/dp3_demo/0730-obj-45948",
        "/scratch/chialiang/dp3_demo/0730-obj-45964",
        "/scratch/chialiang/dp3_demo/0730-obj-45963",
        "/scratch/chialiang/dp3_demo/0730-obj-46019",
        "/scratch/chialiang/dp3_demo/0730-obj-46002",
        "/scratch/chialiang/dp3_demo/0730-obj-46033",
        "/scratch/chialiang/dp3_demo/0730-obj-46029",
        "/scratch/chialiang/dp3_demo/0730-obj-46044",
        "/scratch/chialiang/dp3_demo/0730-obj-46037",
        "/scratch/chialiang/dp3_demo/0730-obj-46060",
        "/scratch/chialiang/dp3_demo/0730-obj-46045",
        "/scratch/chialiang/dp3_demo/0730-obj-46108",
        "/scratch/chialiang/dp3_demo/0730-obj-46084",
        "/scratch/chialiang/dp3_demo/0730-obj-46120",
        "/scratch/chialiang/dp3_demo/0730-obj-46117",
        "/scratch/chialiang/dp3_demo/0730-obj-46145",
        "/scratch/chialiang/dp3_demo/0730-obj-46123",
        "/scratch/chialiang/dp3_demo/0730-obj-46180",
        "/scratch/chialiang/dp3_demo/0730-obj-46179",
        "/scratch/chialiang/dp3_demo/0730-obj-46230",
        "/scratch/chialiang/dp3_demo/0730-obj-46199",
        "/scratch/chialiang/dp3_demo/0730-obj-46380",
        "/scratch/chialiang/dp3_demo/0730-obj-46277",

        # g4
        "/scratch/chialiang/dp3_demo/0730-obj-46430",
        "/scratch/chialiang/dp3_demo/0730-obj-46427",
        "/scratch/chialiang/dp3_demo/0730-obj-46466",
        "/scratch/chialiang/dp3_demo/0730-obj-46439",
        "/scratch/chialiang/dp3_demo/0730-obj-46549",
        "/scratch/chialiang/dp3_demo/0730-obj-46537",
        "/scratch/chialiang/dp3_demo/0730-obj-46598",
        "/scratch/chialiang/dp3_demo/0730-obj-46556",
        "/scratch/chialiang/dp3_demo/0730-obj-46699",
        "/scratch/chialiang/dp3_demo/0730-obj-46616",
        "/scratch/chialiang/dp3_demo/0730-obj-46741",
        "/scratch/chialiang/dp3_demo/0730-obj-46700",
        "/scratch/chialiang/dp3_demo/0730-obj-46847",
        "/scratch/chialiang/dp3_demo/0730-obj-46744",
        "/scratch/chialiang/dp3_demo/0730-obj-46859",
        "/scratch/chialiang/dp3_demo/0730-obj-46856",
        "/scratch/chialiang/dp3_demo/0730-obj-46906",
        "/scratch/chialiang/dp3_demo/0730-obj-46889",
        "/scratch/chialiang/dp3_demo/0730-obj-46955",
        "/scratch/chialiang/dp3_demo/0730-obj-46944",
        "/scratch/chialiang/dp3_demo/0730-obj-47024",
        "/scratch/chialiang/dp3_demo/0730-obj-46981",
        "/scratch/chialiang/dp3_demo/0730-obj-47183",
        "/scratch/chialiang/dp3_demo/0730-obj-47089",
        "/scratch/chialiang/dp3_demo/0730-obj-47233",
        "/scratch/chialiang/dp3_demo/0730-obj-47207",
        "/scratch/chialiang/dp3_demo/0730-obj-47278",
        "/scratch/chialiang/dp3_demo/0730-obj-47252",
        "/scratch/chialiang/dp3_demo/0730-obj-47296",
        "/scratch/chialiang/dp3_demo/0730-obj-47290",
        "/scratch/chialiang/dp3_demo/0730-obj-47514",
        "/scratch/chialiang/dp3_demo/0730-obj-47438",
        "/scratch/chialiang/dp3_demo/0730-obj-47601",
        "/scratch/chialiang/dp3_demo/0730-obj-47595",
        "/scratch/chialiang/dp3_demo/0730-obj-47701",
        "/scratch/chialiang/dp3_demo/0730-obj-47632",
        "/scratch/chialiang/dp3_demo/0730-obj-47853",
        "/scratch/chialiang/dp3_demo/0730-obj-47729",
        "/scratch/chialiang/dp3_demo/0730-obj-48051",
        "/scratch/chialiang/dp3_demo/0730-obj-47926",
        "/scratch/chialiang/dp3_demo/0730-obj-48452",
        "/scratch/chialiang/dp3_demo/0730-obj-48413",
        "/scratch/chialiang/dp3_demo/0730-obj-48490",
        "/scratch/chialiang/dp3_demo/0730-obj-48467",
        "/scratch/chialiang/dp3_demo/0730-obj-48517",
        "/scratch/chialiang/dp3_demo/0730-obj-48513",
        "/scratch/chialiang/dp3_demo/0730-obj-48746",
        "/scratch/chialiang/dp3_demo/0730-obj-48721",
        "/scratch/chialiang/dp3_demo/0730-obj-49140",
        "/scratch/chialiang/dp3_demo/0730-obj-48878",
    ]

    dst_dirs = [
        "/scratch/chialiang/dp3_demo_combine_2/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point",
        "/scratch/chialiang/dp3_demo_combine_2/0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action",
        "/scratch/chialiang/dp3_demo_combine_2/0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0628-act3d-obj-48700-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-45526",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-45661",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-45694",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-45780",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-45910",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-45961",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-46408",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-46417",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-46440",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-46490",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-46762",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-46825",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-46893",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-47235",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-47281",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-47315",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-47529",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-47669",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-47944",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-48063",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-48177",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-48356",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-48623",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-48876",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-49025",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-49062",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-49132",
        "/scratch/chialiang/dp3_demo_combine_2/0705-obj-49133",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-40417",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-41085",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-41452",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45162",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45176",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45194",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45203",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45248",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45271",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45290",
        "/scratch/chialiang/dp3_demo_combine_2/0712-obj-45305", 
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45413",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45420",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45427",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45594",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45620",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45623",

        # g2
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45636",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45670",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45689",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45696",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45749",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45759",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45916",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45936",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45950",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-45984",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46092",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46130",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46134",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46197",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46401",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46456",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46480",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46481",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46544",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-46641",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47178",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47182",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47227",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47577",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47648",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47747",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47808",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-47976",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-48010",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-48258",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-48797",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-48379",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-48859",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-48855",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-35059",
        "/scratch/chialiang/dp3_demo_combine_2/0725-obj-49188",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-41083",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-41004",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-44781",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-41529",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-44853",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-44826",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45130",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45092",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45146",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45135",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45168",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45164",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45212",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45173",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45372",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45213",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45387",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45374",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45419",

        # g3
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45415",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45503",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45423",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45524",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45505",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45575",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45573",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45612",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45606",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45622",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45621",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45638",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45632",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45662",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45645",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45676",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45671",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45687",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45677",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45710",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45699",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45756",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45746",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45784",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45783",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45801",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45790",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45853",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45822",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45915",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45855",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45949",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45948",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45964",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-45963",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46019",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46002",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46033",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46029",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46044",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46037",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46060",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46045",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46108",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46084",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46120",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46117",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46145",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46123",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46180",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46179",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46230",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46199",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46380",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46277",

        # g4
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46430",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46427",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46466",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46439",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46549",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46537",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46598",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46556",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46699",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46616",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46741",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46700",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46847",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46744",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46859",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46856",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46906",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46889",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46955",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46944",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47024",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-46981",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47183",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47089",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47233",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47207",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47278",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47252",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47296",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47290",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47514",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47438",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47601",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47595",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47701",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47632",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47853",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47729",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48051",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-47926",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48452",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48413",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48490",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48467",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48517",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48513",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48746",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48721",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-49140",
        "/scratch/chialiang/dp3_demo_combine_2/0730-obj-48878",
    ]

    dense_steps_around_goal = 2
    combine_step = 2

    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        combine_action(
            src_dir=src_dir, 
            dst_dir=dst_dir, 
            combine_step=combine_step,
            dense_steps_around_goal=dense_steps_around_goal
        )
        