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

    for pkl_dir in pkl_dirs:

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
        run_gripper_action = 0
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
                    run_gripper_action = raw_wpt_data[wpt_id]['action'][0, 9]
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
                action[0, 9] = run_gripper_action
                run_gripper_action = raw_wpt_data[wpt_id]['action'][0, 9]

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
                    run_gripper_action = raw_wpt_data[wpt_id]['action'][0, 9]
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
                action[0, 9] = run_gripper_action
                run_gripper_action = raw_wpt_data[wpt_id]['action'][0, 9]

                # last action will be computed in the current state
                saved_data_list[-2]['action'] = action
            
            else:

                run_gripper_action += raw_wpt_data[wpt_id]['action'][0, 9]

        # for last action only
        last_action = raw_wpt_data[run_index]['action']
        run_gripper_action = raw_wpt_data[run_index]['action'][0, 9]
        last_delta_rot_mat = rotation_transfer_6D_to_matrix(last_action[0, 3:9])
        for last_few_wpt_id in range(run_index + 1, traj_length):
            
            # position
            last_action[0, :3] += raw_wpt_data[last_few_wpt_id]['action'][0, :3]

            # rotation
            current_delta_rot_mat = rotation_transfer_6D_to_matrix(raw_wpt_data[last_few_wpt_id]['action'][0, 3:9])
            last_delta_rot_mat = last_delta_rot_mat @ current_delta_rot_mat

            # gripper
            run_gripper_action += raw_wpt_data[last_few_wpt_id]['action'][0, 9]
            
        last_action[0, 3:9] = rotation_transfer_matrix_to_6D(last_delta_rot_mat)
        last_action[0, 9] = run_gripper_action
        saved_data_list[-1]['action'] = last_action
        
        # save data
        for new_wpt_id, saved_data in enumerate(saved_data_list):
            target_dir = f'{dst_dir}/{traj_name}'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            new_pickle_data_save_path = f'{target_dir}/{new_wpt_id}.pkl'
            with open(new_pickle_data_save_path, "wb") as f:
                pickle.dump(saved_data, f)
                cprint(f"Saving new data to: {new_pickle_data_save_path}", "green")

if __name__=="__main__":

    src_dirs = [
'0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action',
'0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point',
'0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action',
'0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
'0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
'0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
'0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
'0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
'0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
'0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
'0628-act3d-obj-48700-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',

'0705-obj-45526',
'0705-obj-45661',
'0705-obj-45694',
'0705-obj-45780',
'0705-obj-45910',
'0705-obj-45961',
'0705-obj-46408',
'0705-obj-46417',
'0705-obj-46440',
'0705-obj-46490',
'0705-obj-46762',
'0705-obj-46825',
'0705-obj-46893',
'0705-obj-47235',
'0705-obj-47281',
'0705-obj-47315',
'0705-obj-47529',
'0705-obj-47669',
'0705-obj-47944',
'0705-obj-48063',
'0705-obj-48177',
'0705-obj-48356',
'0705-obj-48623',
'0705-obj-48876',
'0705-obj-49025',
'0705-obj-49062',
'0705-obj-49132',
'0705-obj-49133',
'0712-obj-40417',
'0712-obj-41085',
'0712-obj-41452',
'0712-obj-45162',
'0712-obj-45176',
'0712-obj-45194',
'0712-obj-45203',
'0712-obj-45248',
'0712-obj-45271',
'0712-obj-45290',
'0712-obj-45305',

'0725-obj-45427',
'0725-obj-45620',
'0725-obj-45623',
'0725-obj-45636',
'0725-obj-45689',
'0725-obj-45696',
'0725-obj-45749',
'0725-obj-45759',
'0725-obj-45936',
'0725-obj-45984',
'0725-obj-46130',
'0725-obj-46197',
'0725-obj-46481',
'0725-obj-46544',
'0725-obj-47178',
'0725-obj-47182',
'0725-obj-47227',
'0725-obj-47577',
'0725-obj-47648',
'0725-obj-47747',
'0725-obj-47808',
'0725-obj-47976',
'0725-obj-48010',
'0725-obj-48258',
'0725-obj-48379',
'0725-obj-48797',
'0725-obj-48855',
'0725-obj-48859',
'0725-obj-49188','0730-obj-35059','0730-obj-41004','0730-obj-41083','0730-obj-44781','0730-obj-44826','0730-obj-44853','0730-obj-45092','0730-obj-45130','0730-obj-45135','0730-obj-45146','0730-obj-45164','0730-obj-45168','0730-obj-45173','0730-obj-45212','0730-obj-45213','0730-obj-45372','0730-obj-45374','0730-obj-45387','0730-obj-45415','0730-obj-45419','0730-obj-45423',
'0730-obj-45503',
'0730-obj-45505',
'0730-obj-45524',
'0730-obj-45573',
'0730-obj-45575',
'0730-obj-45606',
'0730-obj-45612',
'0730-obj-45621',
'0730-obj-45622',
'0730-obj-45632',
'0730-obj-45638',
'0730-obj-45645',
'0730-obj-45662',
'0730-obj-45671',
'0730-obj-45676',
'0730-obj-45677',
'0730-obj-45687',
'0730-obj-45699',
'0730-obj-45710',
'0730-obj-45746',
'0730-obj-45756',
'0730-obj-45783',
'0730-obj-45784',
'0730-obj-45790',
'0730-obj-45801',
'0730-obj-45822',
'0730-obj-45853',
'0730-obj-45855',
'0730-obj-45915',
'0730-obj-45948',
'0730-obj-45949',
'0730-obj-45963',
'0730-obj-45964',
'0730-obj-46019',
'0730-obj-46029',
'0730-obj-46033',
'0730-obj-46037',
'0730-obj-46044',
'0730-obj-46045',
'0730-obj-46060',
'0730-obj-46084',
'0730-obj-46108',
'0730-obj-46117',
'0730-obj-46120',
'0730-obj-46123',
'0730-obj-46145',
'0730-obj-46179',
'0730-obj-46180',
'0730-obj-46199',
'0730-obj-46380',
'0730-obj-46427',
'0730-obj-46430',
'0730-obj-46439',
'0730-obj-46537',
'0730-obj-46549',
'0730-obj-46556',
'0730-obj-46598',
'0730-obj-46616',
'0730-obj-46699',
'0730-obj-46700',
'0730-obj-46741',
'0730-obj-46744',
'0730-obj-46847',
'0730-obj-46856',
'0730-obj-46859',
'0730-obj-46889',
'0730-obj-46906',
'0730-obj-46944',
'0730-obj-46955',
'0730-obj-46981',
'0730-obj-47024',
'0730-obj-47089',
'0730-obj-47183',
'0730-obj-47207',
'0730-obj-47233',
'0730-obj-47252',
'0730-obj-47278',
'0730-obj-47290',
'0730-obj-47296',
'0730-obj-47438',
'0730-obj-47514',
'0730-obj-47595',
'0730-obj-47601',
'0730-obj-47632',
'0730-obj-47701',
'0730-obj-47729',
'0730-obj-47853',
'0730-obj-47926',
'0730-obj-48413',
'0730-obj-48452',
'0730-obj-48467',   
'0730-obj-48490',
'0730-obj-48513',
'0730-obj-48517',
'0730-obj-48721',
'0730-obj-48746',
'0730-obj-48878',
]


    all_objs = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd/")
    all_objs = sorted(all_objs)
    src_dirs = all_objs

    dense_steps_around_goal = 0
    combine_step = 2
    
    src_dirs = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd", x) for x in src_dirs]
    dst_dirs = [x.replace("dp3_demo_real_world_noise_pcd", "dp3_demo_real_world_noise_pcd_combined_{}_step_{}".format(combine_step, dense_steps_around_goal)) for x in src_dirs]


    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        combine_action(
            src_dir=src_dir, 
            dst_dir=dst_dir, 
            combine_step=combine_step,
            dense_steps_around_goal=dense_steps_around_goal
        )
        