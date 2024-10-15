import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import zarr
import os
from termcolor import cprint
import numpy as np
from tqdm import tqdm
import pickle

class PointNetDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, is_pickle=False, use_all_data=False):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        self.is_pickle = is_pickle
        self.use_all_data = use_all_data
        
        if only_first_stage:
            cprint('======= ONLY FIRST STAGE =======', 'red')

        if eval_episode is not None:
            cprint('======= EVAL MODE =======', 'red')
            cprint(f'Only evaluating the first observation of {eval_episode} episodes', 'red')

        self.all_zarr_paths = []
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if not self.use_all_data:
                end = min(end, 75)
            if eval_episode is not None:
                end = beg + eval_episode
            all_subfolder = all_subfolder[beg:end]
            self.all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]

        cprint('Preparing all zarr paths', 'green')
        self.episode_lengths = []
        for idx, zarr_path in enumerate(tqdm(self.all_zarr_paths)):
            if is_pickle:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
                    
                first_goal = None

                for i, substep in enumerate(all_substeps):
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    data = pickle.load(open(substep_path, 'rb'))
                    action = data['action'][:]

                    current_goal = data['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break

            
            else:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x))

                first_goal = None

                for i, substep in enumerate(all_substeps):
                    
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    group = zarr.open(substep_path, 'r')
                    src_store = group.store
                    src_root = zarr.group(src_store)

                    action = src_root['data']['action'][:]

                    current_goal = src_root['data']['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break

            if not only_first_stage and eval_episode is None:
                self.episode_lengths.append(len(all_substeps))

        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f'Finished preparing all zarr paths with total datapoints: {self.accumulated_episode_lengths[-1]}', 'green')

    def __len__(self):
        return self.accumulated_episode_lengths[-1]

    def __getitem__(self, idx):
        episode_idx = np.searchsorted(self.accumulated_episode_lengths, idx, side='right')
        start_idx = idx - self.accumulated_episode_lengths[episode_idx]

        if start_idx < 0:
            start_idx += self.episode_lengths[episode_idx]

        if self.is_pickle:
            step_path = os.path.join(self.all_zarr_paths[episode_idx], str(start_idx) + '.pkl')
            data = pickle.load(open(step_path, 'rb'))
            pointcloud = data['point_cloud'][:][0]
            gripper_pcd = data['gripper_pcd'][:][0]
            goal_gripper_pcd = data['goal_gripper_pcd'][:][0]
        else:
            zarr_path = self.all_zarr_paths[episode_idx]
            
            step_path = os.path.join(zarr_path, str(start_idx))
            group = zarr.open(step_path, 'r')
            src_store = group.store
            src_root = zarr.group(src_store)
            pointcloud = src_root['data']['point_cloud'][:][0]
            gripper_pcd = src_root['data']['gripper_pcd'][:][0]
            goal_gripper_pcd = src_root['data']['goal_gripper_pcd'][:][0]

        return pointcloud, gripper_pcd, goal_gripper_pcd
    
class PredictTwoGoalsDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, is_pickle=False, use_all_data=False):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        self.is_pickle = is_pickle
        self.use_all_data = use_all_data
        
        if only_first_stage:
            cprint('======= ONLY FIRST STAGE =======', 'red')

        if eval_episode is not None:
            cprint('======= EVAL MODE =======', 'red')
            cprint(f'Only evaluating the first observation of {eval_episode} episodes', 'red')

        self.all_zarr_paths = []
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if not self.use_all_data:
                end = min(end, 75)
            if eval_episode is not None:
                end = beg + eval_episode
            all_subfolder = all_subfolder[beg:end]
            self.all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]

        cprint('Preparing all zarr paths', 'green')
        self.episode_lengths = []
        for idx, zarr_path in enumerate(tqdm(self.all_zarr_paths)):
            if is_pickle:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
                    
                first_goal = None

                for i, substep in enumerate(all_substeps):
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    data = pickle.load(open(substep_path, 'rb'))
                    action = data['action'][:]

                    current_goal = data['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break

            
            else:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x))

                first_goal = None

                for i, substep in enumerate(all_substeps):
                    
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    group = zarr.open(substep_path, 'r')
                    src_store = group.store
                    src_root = zarr.group(src_store)

                    action = src_root['data']['action'][:]

                    current_goal = src_root['data']['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break

            if not only_first_stage and eval_episode is None:
                self.episode_lengths.append(len(all_substeps))

        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f'Finished preparing all zarr paths with total datapoints: {self.accumulated_episode_lengths[-1]}', 'green')

    def __len__(self):
        return len(self.episode_lengths) # num data points is just num trajectories

    def __getitem__(self, idx):
        episode_idx = idx
        start_idx = 0
        end_idx = self.episode_lengths[episode_idx] - 1

        if self.is_pickle:
            step_path = os.path.join(self.all_zarr_paths[episode_idx], str(start_idx) + '.pkl')
            data = pickle.load(open(step_path, 'rb'))
            pointcloud = data['point_cloud'][:][0]
            gripper_pcd = data['gripper_pcd'][:][0]
            goal_gripper_pcd = data['goal_gripper_pcd'][:][0]

            step_path_end = os.path.join(self.all_zarr_paths[episode_idx], str(end_idx) + '.pkl')
            data_end = pickle.load(open(step_path_end, 'rb'))
            goal_gripper_pcd_end = data_end['goal_gripper_pcd'][:][0]
        else:
            zarr_path = self.all_zarr_paths[episode_idx]
            
            step_path = os.path.join(zarr_path, str(start_idx))
            group = zarr.open(step_path, 'r')
            src_store = group.store
            src_root = zarr.group(src_store)
            pointcloud = src_root['data']['point_cloud'][:][0]
            gripper_pcd = src_root['data']['gripper_pcd'][:][0]
            goal_gripper_pcd = src_root['data']['goal_gripper_pcd'][:][0]

        return pointcloud, gripper_pcd, np.concatenate([goal_gripper_pcd, goal_gripper_pcd_end], axis=0)
        
def get_dataloader(all_obj_paths=None, batch_size=32, beg_ratio=0, end_ratio=0.9, shuffle=True, eval_episode=None, only_first_stage=False):
    if all_obj_paths is None:
        all_obj_paths = ['0705-obj-41510', '0705-obj-45448', '0705-obj-46462', '0705-obj-46732', '0705-obj-46801', '0705-obj-46874', '0705-obj-46922', '0705-obj-46966', '0705-obj-47570', '0705-obj-47578', '0705-obj-48700', '0705-obj-45526', '0705-obj-45661', '0705-obj-45694', '0705-obj-45780', '0705-obj-45910', '0705-obj-45961', '0705-obj-46408', '0705-obj-46417', '0705-obj-46440', '0705-obj-46490', '0705-obj-46762', '0705-obj-46825', '0705-obj-46893', '0705-obj-47235', '0705-obj-47281', '0705-obj-47315', '0705-obj-47529', '0705-obj-47669', '0705-obj-47944', '0705-obj-48063', '0705-obj-48177', '0705-obj-48356', '0705-obj-48623', '0705-obj-48876', '0705-obj-49025', '0705-obj-49062', '0705-obj-49132', '0705-obj-49133', '0712-obj-40417', '0712-obj-41085', '0712-obj-41452', '0712-obj-45162', '0712-obj-45176', '0712-obj-45194', '0712-obj-45203', '0712-obj-45248', '0712-obj-45271', '0712-obj-45290', '0712-obj-45305']
        all_obj_paths = ['/scratch/chialiang/dp3_demo/' + s for s in all_obj_paths]
    dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


save_data_name_0='0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action'
save_data_name_1='0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point'
save_data_name_2='0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action'
save_data_name_3='0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'
save_data_name_4='0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'
save_data_name_5='0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'
save_data_name_6='0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'
save_data_name_7='0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'
save_data_name_8='0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'
save_data_name_9='0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'
save_data_name_10='0628-act3d-obj-48700-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1'

save_data_name_11='0705-obj-45526'
save_data_name_12='0705-obj-45661'
save_data_name_13='0705-obj-45694'
save_data_name_14='0705-obj-45780'
save_data_name_15='0705-obj-45910'
save_data_name_16='0705-obj-45961'
save_data_name_17='0705-obj-46408'
save_data_name_18='0705-obj-46417'
save_data_name_19='0705-obj-46440'
save_data_name_20='0705-obj-46490'
save_data_name_21='0705-obj-46762'
save_data_name_22='0705-obj-46825'
save_data_name_23='0705-obj-46893'
save_data_name_24='0705-obj-47235'
save_data_name_25='0705-obj-47281'
save_data_name_26='0705-obj-47315'
save_data_name_27='0705-obj-47529'
save_data_name_28='0705-obj-47669'
save_data_name_29='0705-obj-47944'
save_data_name_30='0705-obj-48063'
save_data_name_31='0705-obj-48177'
save_data_name_32='0705-obj-48356'
save_data_name_33='0705-obj-48623'
save_data_name_34='0705-obj-48876'
save_data_name_35='0705-obj-49025'
save_data_name_36='0705-obj-49062'
save_data_name_37='0705-obj-49132'
save_data_name_38='0705-obj-49133'
save_data_name_39='0712-obj-40417'
save_data_name_40='0712-obj-41085'
save_data_name_41='0712-obj-41452'
save_data_name_42='0712-obj-45162'
save_data_name_43='0712-obj-45176'
save_data_name_44='0712-obj-45194'
save_data_name_45='0712-obj-45203'
save_data_name_46='0712-obj-45248'
save_data_name_47='0712-obj-45271'
save_data_name_48='0712-obj-45290'
save_data_name_49='0712-obj-45305'

save_data_name_50='0725-obj-45427'
save_data_name_51='0725-obj-45620'
save_data_name_52='0725-obj-45623'
save_data_name_53='0725-obj-45636'
save_data_name_54='0725-obj-45689'
save_data_name_55='0725-obj-45696'
save_data_name_56='0725-obj-45749'
save_data_name_57='0725-obj-45759'
save_data_name_58='0725-obj-45936'
save_data_name_59='0725-obj-45984'
save_data_name_60='0725-obj-46130'
save_data_name_61='0725-obj-46197'
save_data_name_62='0725-obj-46481'
save_data_name_63='0725-obj-46544'
save_data_name_64='0725-obj-47178'
save_data_name_65='0725-obj-47182'
save_data_name_66='0725-obj-47227'
save_data_name_67='0725-obj-47577'
save_data_name_68='0725-obj-47648'
save_data_name_69='0725-obj-47747'
save_data_name_70='0725-obj-47808'
save_data_name_71='0725-obj-47976'
save_data_name_72='0725-obj-48010'
save_data_name_73='0725-obj-48258'
save_data_name_74='0725-obj-48379'
save_data_name_75='0725-obj-48797'
save_data_name_76='0725-obj-48855'
save_data_name_77='0725-obj-48859'
save_data_name_78='0725-obj-49188'
save_data_name_79='0730-obj-35059'
save_data_name_80='0730-obj-41004'
save_data_name_81='0730-obj-41083'
save_data_name_82='0730-obj-44781'
save_data_name_83='0730-obj-44826'
save_data_name_84='0730-obj-44853'
save_data_name_85='0730-obj-45092'
save_data_name_86='0730-obj-45130'
save_data_name_87='0730-obj-45135'
save_data_name_88='0730-obj-45146'
save_data_name_89='0730-obj-45164'
save_data_name_90='0730-obj-45168'
save_data_name_91='0730-obj-45173'
save_data_name_92='0730-obj-45212'
save_data_name_93='0730-obj-45213'
save_data_name_94='0730-obj-45372'
save_data_name_95='0730-obj-45374'
save_data_name_96='0730-obj-45387'
save_data_name_97='0730-obj-45415'
save_data_name_98='0730-obj-45419'
save_data_name_99='0730-obj-45423'
save_data_name_100='0730-obj-45503'
save_data_name_101='0730-obj-45505'
save_data_name_102='0730-obj-45524'
save_data_name_103='0730-obj-45573'
save_data_name_104='0730-obj-45575'
save_data_name_105='0730-obj-45606'
save_data_name_106='0730-obj-45612'
save_data_name_107='0730-obj-45621'
save_data_name_108='0730-obj-45622'
save_data_name_109='0730-obj-45632'
save_data_name_110='0730-obj-45638'
save_data_name_111='0730-obj-45645'
save_data_name_112='0730-obj-45662'
save_data_name_113='0730-obj-45671'
save_data_name_114='0730-obj-45676'
save_data_name_115='0730-obj-45677'
save_data_name_116='0730-obj-45687'
save_data_name_117='0730-obj-45699'
save_data_name_118='0730-obj-45710'
save_data_name_119='0730-obj-45746'
save_data_name_120='0730-obj-45756'
save_data_name_121='0730-obj-45783'
save_data_name_122='0730-obj-45784'
save_data_name_123='0730-obj-45790'
save_data_name_124='0730-obj-45801'
save_data_name_125='0730-obj-45822'
save_data_name_126='0730-obj-45853'
save_data_name_127='0730-obj-45855'
save_data_name_128='0730-obj-45915'
save_data_name_129='0730-obj-45948'
save_data_name_130='0730-obj-45949'
save_data_name_131='0730-obj-45963'
save_data_name_132='0730-obj-45964'
save_data_name_133='0730-obj-46019'
save_data_name_134='0730-obj-46029'
save_data_name_135='0730-obj-46033'
save_data_name_136='0730-obj-46037'
save_data_name_137='0730-obj-46044'
save_data_name_138='0730-obj-46045'
save_data_name_139='0730-obj-46060'
save_data_name_140='0730-obj-46084'
save_data_name_141='0730-obj-46108'
save_data_name_142='0730-obj-46117'
save_data_name_143='0730-obj-46120'
save_data_name_144='0730-obj-46123'
save_data_name_145='0730-obj-46145'
save_data_name_146='0730-obj-46179'
save_data_name_147='0730-obj-46180'
save_data_name_148='0730-obj-46199'
save_data_name_149='0730-obj-46380'
save_data_name_150='0730-obj-46427'
save_data_name_151='0730-obj-46430'
save_data_name_152='0730-obj-46439'
save_data_name_153='0730-obj-46537'
save_data_name_154='0730-obj-46549'
save_data_name_155='0730-obj-46556'
save_data_name_156='0730-obj-46598'
save_data_name_157='0730-obj-46616'
save_data_name_158='0730-obj-46699'
save_data_name_159='0730-obj-46700'
save_data_name_160='0730-obj-46741'
save_data_name_161='0730-obj-46744'
save_data_name_162='0730-obj-46847'
save_data_name_163='0730-obj-46856'
save_data_name_164='0730-obj-46859'
save_data_name_165='0730-obj-46889'
save_data_name_166='0730-obj-46906'
save_data_name_167='0730-obj-46944'
save_data_name_168='0730-obj-46955'
save_data_name_169='0730-obj-46981'
save_data_name_170='0730-obj-47024'
save_data_name_171='0730-obj-47089'
save_data_name_172='0730-obj-47183'
save_data_name_173='0730-obj-47207'
save_data_name_174='0730-obj-47233'
save_data_name_175='0730-obj-47252'
save_data_name_176='0730-obj-47278'
save_data_name_177='0730-obj-47290'
save_data_name_178='0730-obj-47296'
save_data_name_179='0730-obj-47438'
save_data_name_180='0730-obj-47514'
save_data_name_181='0730-obj-47595'
save_data_name_182='0730-obj-47601'
save_data_name_183='0730-obj-47632'
save_data_name_184='0730-obj-47701'
save_data_name_185='0730-obj-47729'
save_data_name_186='0730-obj-47853'
save_data_name_187='0730-obj-47926'
save_data_name_188='0730-obj-48413'
save_data_name_189='0730-obj-48452'
save_data_name_190='0730-obj-48467'
save_data_name_191='0730-obj-48490'
save_data_name_192='0730-obj-48513'
save_data_name_193='0730-obj-48517'
save_data_name_194='0730-obj-48721'
save_data_name_195='0730-obj-48746'
save_data_name_196='0730-obj-48878'

save_data_name_197='0725-obj-41003'
save_data_name_198='0725-obj-45001'
save_data_name_199='0725-obj-45235'
save_data_name_200='0725-obj-45238'
save_data_name_201='0725-obj-45244'
save_data_name_202='0725-obj-45249'
save_data_name_203='0705-obj-45523'
save_data_name_204='0705-obj-46014'
save_data_name_205='0705-obj-46166'
save_data_name_206='0705-obj-46653'
save_data_name_207='0705-obj-47711'
save_data_name_208='0705-obj-48263'
save_data_name_209='0730-obj-45007'
save_data_name_210='0730-obj-45087'
save_data_name_211='0730-obj-45159'
save_data_name_212='0730-obj-45166'
save_data_name_213='0730-obj-45189'
save_data_name_214='0730-obj-45247'
save_data_name_215='0730-obj-45261'
save_data_name_216='0730-obj-45267'
save_data_name_217='0730-obj-45354'
save_data_name_218='0725-obj-45413'
save_data_name_219='0725-obj-45420'
save_data_name_220='0725-obj-45594'
save_data_name_221='0725-obj-45670'
save_data_name_222='0725-obj-45916'
save_data_name_223='0725-obj-45950'
save_data_name_224='0725-obj-46092'
save_data_name_225='0725-obj-46134'
save_data_name_226='0730-obj-46230'
save_data_name_227='0730-obj-46277'
save_data_name_228='0725-obj-46334'
save_data_name_229='0725-obj-46443'
save_data_name_230='0730-obj-46466'
save_data_name_231='0725-obj-46480'
save_data_name_232='0725-obj-46641'
save_data_name_233='0730-obj-47088'
save_data_name_234='0730-obj-47185'
save_data_name_235='0725-obj-47254'
save_data_name_236='0730-obj-47419'
save_data_name_237='0730-obj-47613'
save_data_name_238='0725-obj-47742'
save_data_name_239='0730-obj-48018'
save_data_name_240='0730-obj-48023'
save_data_name_241='0730-obj-48051'
save_data_name_242='0730-obj-48271'
save_data_name_243='0730-obj-48491'
save_data_name_244='0730-obj-48519'
save_data_name_245='0730-obj-48740'
save_data_name_246='0730-obj-49140'
save_data_name_247='0822-obj-10036'
save_data_name_248='0822-obj-10143'
save_data_name_249='0822-obj-10144'
save_data_name_250='0822-obj-10655'
save_data_name_251='0822-obj-10797'
save_data_name_252='0822-obj-10867'
save_data_name_253='0822-obj-10944'
save_data_name_254='0822-obj-11211'
save_data_name_255='0822-obj-11661'
save_data_name_256='0822-obj-11700'
save_data_name_257='0822-obj-12042'
save_data_name_258='0822-obj-12043'
save_data_name_259='0822-obj-12259'
save_data_name_260='0822-obj-12480'
save_data_name_261='0822-obj-12531'
save_data_name_262='0822-obj-12536'
save_data_name_263='0822-obj-12543'
save_data_name_264='0822-obj-12552'
save_data_name_265='0822-obj-12553'
save_data_name_266='0822-obj-12559'
save_data_name_267='0822-obj-12561'
save_data_name_268='0822-obj-12562'
save_data_name_269='0822-obj-12563'
save_data_name_270='0822-obj-12579'
save_data_name_271='0822-obj-12583'
save_data_name_272='0822-obj-12587'
save_data_name_273='0822-obj-12590'
save_data_name_274='0822-obj-12592'
save_data_name_275='0822-obj-12594'
save_data_name_276='0822-obj-12596'
save_data_name_277='0822-obj-12605'
save_data_name_278='0822-obj-12606'
save_data_name_279='0822-obj-12614'
save_data_name_280='0822-obj-12617'
save_data_name_281='0822-obj-7119'
save_data_name_282='0822-obj-7167'
save_data_name_283='0822-obj-7187'
save_data_name_284='0822-obj-7220'
save_data_name_285='0822-obj-7263'
save_data_name_286='0822-obj-7290'

save_data_name_287="0815-obj-35059" 
save_data_name_288="0815-obj-41004" 
save_data_name_289="0815-obj-44781" 
save_data_name_290="0815-obj-44826" 
save_data_name_291="0815-obj-45087" 
save_data_name_292="0815-obj-45092" 
save_data_name_293="0815-obj-45130" 
save_data_name_294="0815-obj-45135" 
save_data_name_295="0815-obj-45164" 
save_data_name_296="0815-obj-45168" 
save_data_name_297="0815-obj-45173" 
save_data_name_298="0815-obj-45189" 
save_data_name_299="0815-obj-45212" 
save_data_name_300="0815-obj-45213" 
save_data_name_301="0815-obj-45247" 
save_data_name_302="0815-obj-45261" 
save_data_name_303="0815-obj-45267" 
save_data_name_304="0815-obj-45354" 
save_data_name_305="0815-obj-45387" 
save_data_name_306="0815-obj-45413" 
save_data_name_307="0815-obj-45419" 
save_data_name_308="0815-obj-45420" 
save_data_name_309="0815-obj-45423" 
save_data_name_310="0815-obj-45503" 
save_data_name_311="0815-obj-45505" 
save_data_name_312="0815-obj-45524" 
save_data_name_313="0815-obj-45573" 
save_data_name_314="0815-obj-45575" 
save_data_name_315="0815-obj-45594" 
save_data_name_316="0815-obj-45606" 
save_data_name_317="0815-obj-45620" 
save_data_name_318="0815-obj-45623" 
save_data_name_319="0815-obj-45638" 
save_data_name_320="0815-obj-45645" 
save_data_name_321="0815-obj-45662" 
save_data_name_322="0815-obj-45677" 
save_data_name_323="0815-obj-45699" 
save_data_name_324="0815-obj-45746" 
save_data_name_325="0815-obj-45783" 
save_data_name_326="0815-obj-45790" 
save_data_name_327="0815-obj-45822" 
save_data_name_328="0815-obj-45853" 
save_data_name_329="0815-obj-45855" 
save_data_name_330="0815-obj-45936" 
save_data_name_331="0815-obj-45937" 
save_data_name_332="0815-obj-45948" 
save_data_name_333="0815-obj-45963" 
save_data_name_334="0815-obj-45964" 
save_data_name_335="0815-obj-46029" 
save_data_name_336="0815-obj-46037" 
save_data_name_337="0815-obj-46060" 
save_data_name_338="0815-obj-46092" 
save_data_name_339="0815-obj-46117" 
save_data_name_340="0815-obj-46120" 
save_data_name_341="0815-obj-46132" 
save_data_name_342="0815-obj-46179" 
save_data_name_343="0815-obj-46197" 
save_data_name_344="0815-obj-46430" 
save_data_name_345="0815-obj-46456" 
save_data_name_346="0815-obj-46537" 
save_data_name_347="0815-obj-46556" 
save_data_name_348="0815-obj-46616" 
save_data_name_349="0815-obj-46699" 
save_data_name_350="0815-obj-46744" 
save_data_name_351="0815-obj-46847" 
save_data_name_352="0815-obj-46889" 
save_data_name_353="0815-obj-46906" 
save_data_name_354="0815-obj-47024" 
save_data_name_355="0815-obj-47178" 
save_data_name_356="0815-obj-47207" 
save_data_name_357="0815-obj-47227" 
save_data_name_358="0815-obj-47252" 
save_data_name_359="0815-obj-47296" 
save_data_name_360="0815-obj-47419" 
save_data_name_361="0815-obj-47438" 
save_data_name_362="0815-obj-47577" 
save_data_name_363="0815-obj-47595" 
save_data_name_364="0815-obj-47601" 
save_data_name_365="0815-obj-47613" 
save_data_name_366="0815-obj-47632" 
save_data_name_367="0815-obj-47729" 
save_data_name_368="0815-obj-47742" 
save_data_name_369="0815-obj-47747" 
save_data_name_370="0815-obj-47808" 
save_data_name_371="0815-obj-47853" 
save_data_name_372="0815-obj-47976" 
save_data_name_373="0815-obj-48010" 
save_data_name_374="0815-obj-48023" 
save_data_name_375="0815-obj-48258" 
save_data_name_376="0815-obj-48271" 
save_data_name_377="0815-obj-48379" 
save_data_name_378="0815-obj-48413" 
save_data_name_379="0815-obj-48452" 
save_data_name_380="0815-obj-48467" 
save_data_name_381="0815-obj-48490" 
save_data_name_382="0815-obj-48513" 
save_data_name_383="0815-obj-48517" 
save_data_name_384="0815-obj-48721" 
save_data_name_385="0815-obj-48740" 
save_data_name_386="0815-obj-48746" 
save_data_name_387="0815-obj-48797" 
save_data_name_388="0815-obj-48855" 
save_data_name_389="0815-obj-48859" 
save_data_name_390="0815-obj-48878" 
save_data_name_391="0815-obj-49188" 
save_data_name_392="0826-obj-44781" 
save_data_name_393="0826-obj-45092" 
save_data_name_394="0826-obj-45135" 
save_data_name_395="0826-obj-45159" 
save_data_name_396="0826-obj-45213" 
save_data_name_397="0826-obj-45247" 
save_data_name_398="0826-obj-45261" 
save_data_name_399="0826-obj-45354" 
save_data_name_400="0826-obj-45372" 
save_data_name_401="0826-obj-45387" 
save_data_name_402="0826-obj-45423" 
save_data_name_403="0826-obj-45575" 
save_data_name_404="0826-obj-45606" 
save_data_name_405="0826-obj-45621" 
save_data_name_406="0826-obj-45638" 
save_data_name_407="0826-obj-45645" 
save_data_name_408="0826-obj-45662" 
save_data_name_409="0826-obj-45676" 
save_data_name_410="0826-obj-45677" 
save_data_name_411="0826-obj-45746" 
save_data_name_412="0826-obj-45790" 
save_data_name_413="0826-obj-45853" 
save_data_name_414="0826-obj-45855" 
save_data_name_415="0826-obj-45915" 
save_data_name_416="0826-obj-45963" 
save_data_name_417="0826-obj-45964" 
save_data_name_418="0826-obj-46002" 
save_data_name_419="0826-obj-46029" 
save_data_name_420="0826-obj-46033" 
save_data_name_421="0826-obj-46037" 
save_data_name_422="0826-obj-46044" 
save_data_name_423="0826-obj-46060" 
save_data_name_424="0826-obj-46117" 
save_data_name_425="0826-obj-46120" 
save_data_name_426="0826-obj-46179" 
save_data_name_427="0826-obj-46430" 
save_data_name_428="0826-obj-46439" 
save_data_name_429="0826-obj-46537" 
save_data_name_430="0826-obj-46616" 
save_data_name_431="0826-obj-46699" 
save_data_name_432="0826-obj-46847" 
save_data_name_433="0826-obj-46856" 
save_data_name_434="0826-obj-46859" 
save_data_name_435="0826-obj-46889" 
save_data_name_436="0826-obj-46955" 
save_data_name_437="0826-obj-47024" 
save_data_name_438="0826-obj-47088" 
save_data_name_439="0826-obj-47089" 
save_data_name_440="0826-obj-47207" 
save_data_name_441="0826-obj-47233" 
save_data_name_442="0826-obj-47252" 
save_data_name_443="0826-obj-47296" 
save_data_name_444="0826-obj-47388" 
save_data_name_445="0826-obj-47419" 
save_data_name_446="0826-obj-47595" 
save_data_name_447="0826-obj-47601" 
save_data_name_448="0826-obj-47613" 
save_data_name_449="0826-obj-47632" 
save_data_name_450="0826-obj-47701" 
save_data_name_451="0826-obj-47729" 
save_data_name_452="0826-obj-47853" 
save_data_name_453="0826-obj-48023" 
save_data_name_454="0826-obj-48271" 
save_data_name_455="0826-obj-48413" 
save_data_name_456="0826-obj-48467" 
save_data_name_457="0826-obj-48490" 
save_data_name_458="0826-obj-48491" 
save_data_name_459="0826-obj-48513" 
save_data_name_460="0826-obj-48517" 
save_data_name_461="0826-obj-48721" 
save_data_name_462="0826-obj-48740"

def get_dataloader_from_pickle(all_obj_paths=None, batch_size=32, beg_ratio=0, end_ratio=0.9, shuffle=True, eval_episode=None, only_first_stage=False):
    dataset_prefix='/scratch/chialiang/dp3_demo'
    if all_obj_paths is None:
        all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
        f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
        f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
        f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
        f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
        f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
        f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
        f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
        f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
        f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
        f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
        f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
        f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
        f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
        f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
        f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
        f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
        f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
        f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
        f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}',]
    dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage, is_pickle=True)    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_dataset_from_pickle(all_obj_paths=None, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, use_all_data=False, use_combined_action=False, dataset_prefix=None, num_train_objects=200, predict_two_goals=False):
    
    if dataset_prefix is None:
        dataset_prefix='/scratch/chialiang/dp3_demo'
        if use_combined_action:
            dataset_prefix='/scratch/chialiang/dp3_demo_combine_2_new'
    
    if all_obj_paths is None:
        if num_train_objects == 200:
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}',]
        elif num_train_objects == 300:
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}']
        elif num_train_objects == 500:
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}',
            f'{dataset_prefix}/{save_data_name_287}', f'{dataset_prefix}/{save_data_name_288}', f'{dataset_prefix}/{save_data_name_289}', f'{dataset_prefix}/{save_data_name_290}', f'{dataset_prefix}/{save_data_name_291}', f'{dataset_prefix}/{save_data_name_292}', f'{dataset_prefix}/{save_data_name_293}', f'{dataset_prefix}/{save_data_name_294}', f'{dataset_prefix}/{save_data_name_295}', f'{dataset_prefix}/{save_data_name_296}', f'{dataset_prefix}/{save_data_name_297}', f'{dataset_prefix}/{save_data_name_298}', f'{dataset_prefix}/{save_data_name_299}', f'{dataset_prefix}/{save_data_name_300}', f'{dataset_prefix}/{save_data_name_301}', f'{dataset_prefix}/{save_data_name_302}', f'{dataset_prefix}/{save_data_name_303}', f'{dataset_prefix}/{save_data_name_304}', f'{dataset_prefix}/{save_data_name_305}', f'{dataset_prefix}/{save_data_name_306}', f'{dataset_prefix}/{save_data_name_307}', f'{dataset_prefix}/{save_data_name_308}', f'{dataset_prefix}/{save_data_name_309}', f'{dataset_prefix}/{save_data_name_310}', f'{dataset_prefix}/{save_data_name_311}', f'{dataset_prefix}/{save_data_name_312}', f'{dataset_prefix}/{save_data_name_313}', f'{dataset_prefix}/{save_data_name_314}', f'{dataset_prefix}/{save_data_name_315}', f'{dataset_prefix}/{save_data_name_316}', f'{dataset_prefix}/{save_data_name_317}', f'{dataset_prefix}/{save_data_name_318}', f'{dataset_prefix}/{save_data_name_319}', f'{dataset_prefix}/{save_data_name_320}', f'{dataset_prefix}/{save_data_name_321}', f'{dataset_prefix}/{save_data_name_322}', f'{dataset_prefix}/{save_data_name_323}', f'{dataset_prefix}/{save_data_name_324}', f'{dataset_prefix}/{save_data_name_325}', f'{dataset_prefix}/{save_data_name_326}', f'{dataset_prefix}/{save_data_name_327}', f'{dataset_prefix}/{save_data_name_328}', f'{dataset_prefix}/{save_data_name_329}', f'{dataset_prefix}/{save_data_name_330}', f'{dataset_prefix}/{save_data_name_331}', f'{dataset_prefix}/{save_data_name_332}', f'{dataset_prefix}/{save_data_name_333}', f'{dataset_prefix}/{save_data_name_334}', f'{dataset_prefix}/{save_data_name_335}', f'{dataset_prefix}/{save_data_name_336}', f'{dataset_prefix}/{save_data_name_337}', f'{dataset_prefix}/{save_data_name_338}', f'{dataset_prefix}/{save_data_name_339}', f'{dataset_prefix}/{save_data_name_340}', f'{dataset_prefix}/{save_data_name_341}', f'{dataset_prefix}/{save_data_name_342}', f'{dataset_prefix}/{save_data_name_343}', f'{dataset_prefix}/{save_data_name_344}', f'{dataset_prefix}/{save_data_name_345}', f'{dataset_prefix}/{save_data_name_346}', f'{dataset_prefix}/{save_data_name_347}', f'{dataset_prefix}/{save_data_name_348}', f'{dataset_prefix}/{save_data_name_349}', f'{dataset_prefix}/{save_data_name_350}', f'{dataset_prefix}/{save_data_name_351}', f'{dataset_prefix}/{save_data_name_352}', f'{dataset_prefix}/{save_data_name_353}', f'{dataset_prefix}/{save_data_name_354}', f'{dataset_prefix}/{save_data_name_355}', f'{dataset_prefix}/{save_data_name_356}', f'{dataset_prefix}/{save_data_name_357}', f'{dataset_prefix}/{save_data_name_358}', f'{dataset_prefix}/{save_data_name_359}', f'{dataset_prefix}/{save_data_name_360}', f'{dataset_prefix}/{save_data_name_361}', f'{dataset_prefix}/{save_data_name_362}', f'{dataset_prefix}/{save_data_name_363}', f'{dataset_prefix}/{save_data_name_364}', f'{dataset_prefix}/{save_data_name_365}', f'{dataset_prefix}/{save_data_name_366}', f'{dataset_prefix}/{save_data_name_367}', f'{dataset_prefix}/{save_data_name_368}', f'{dataset_prefix}/{save_data_name_369}', f'{dataset_prefix}/{save_data_name_370}', f'{dataset_prefix}/{save_data_name_371}', f'{dataset_prefix}/{save_data_name_372}', f'{dataset_prefix}/{save_data_name_373}', f'{dataset_prefix}/{save_data_name_374}', f'{dataset_prefix}/{save_data_name_375}', f'{dataset_prefix}/{save_data_name_376}', f'{dataset_prefix}/{save_data_name_377}', f'{dataset_prefix}/{save_data_name_378}', f'{dataset_prefix}/{save_data_name_379}', f'{dataset_prefix}/{save_data_name_380}', f'{dataset_prefix}/{save_data_name_381}', f'{dataset_prefix}/{save_data_name_382}', f'{dataset_prefix}/{save_data_name_383}', f'{dataset_prefix}/{save_data_name_384}', f'{dataset_prefix}/{save_data_name_385}', f'{dataset_prefix}/{save_data_name_386}', f'{dataset_prefix}/{save_data_name_387}', f'{dataset_prefix}/{save_data_name_388}', f'{dataset_prefix}/{save_data_name_389}', f'{dataset_prefix}/{save_data_name_390}', f'{dataset_prefix}/{save_data_name_391}', f'{dataset_prefix}/{save_data_name_392}', f'{dataset_prefix}/{save_data_name_393}', f'{dataset_prefix}/{save_data_name_394}', f'{dataset_prefix}/{save_data_name_395}', f'{dataset_prefix}/{save_data_name_396}', f'{dataset_prefix}/{save_data_name_397}', f'{dataset_prefix}/{save_data_name_398}', f'{dataset_prefix}/{save_data_name_399}', f'{dataset_prefix}/{save_data_name_400}', f'{dataset_prefix}/{save_data_name_401}', f'{dataset_prefix}/{save_data_name_402}', f'{dataset_prefix}/{save_data_name_403}', f'{dataset_prefix}/{save_data_name_404}', f'{dataset_prefix}/{save_data_name_405}', f'{dataset_prefix}/{save_data_name_406}', f'{dataset_prefix}/{save_data_name_407}', f'{dataset_prefix}/{save_data_name_408}', f'{dataset_prefix}/{save_data_name_409}', f'{dataset_prefix}/{save_data_name_410}', f'{dataset_prefix}/{save_data_name_411}', f'{dataset_prefix}/{save_data_name_412}', f'{dataset_prefix}/{save_data_name_413}', f'{dataset_prefix}/{save_data_name_414}', f'{dataset_prefix}/{save_data_name_415}', f'{dataset_prefix}/{save_data_name_416}', f'{dataset_prefix}/{save_data_name_417}', f'{dataset_prefix}/{save_data_name_418}', f'{dataset_prefix}/{save_data_name_419}', f'{dataset_prefix}/{save_data_name_420}', f'{dataset_prefix}/{save_data_name_421}', f'{dataset_prefix}/{save_data_name_422}', f'{dataset_prefix}/{save_data_name_423}', f'{dataset_prefix}/{save_data_name_424}', f'{dataset_prefix}/{save_data_name_425}', f'{dataset_prefix}/{save_data_name_426}', f'{dataset_prefix}/{save_data_name_427}', f'{dataset_prefix}/{save_data_name_428}', f'{dataset_prefix}/{save_data_name_429}', f'{dataset_prefix}/{save_data_name_430}', f'{dataset_prefix}/{save_data_name_431}', f'{dataset_prefix}/{save_data_name_432}', f'{dataset_prefix}/{save_data_name_433}', f'{dataset_prefix}/{save_data_name_434}', f'{dataset_prefix}/{save_data_name_435}', f'{dataset_prefix}/{save_data_name_436}', f'{dataset_prefix}/{save_data_name_437}', f'{dataset_prefix}/{save_data_name_438}', f'{dataset_prefix}/{save_data_name_439}', f'{dataset_prefix}/{save_data_name_440}', f'{dataset_prefix}/{save_data_name_441}', f'{dataset_prefix}/{save_data_name_442}', f'{dataset_prefix}/{save_data_name_443}', f'{dataset_prefix}/{save_data_name_444}', f'{dataset_prefix}/{save_data_name_445}', f'{dataset_prefix}/{save_data_name_446}', f'{dataset_prefix}/{save_data_name_447}', f'{dataset_prefix}/{save_data_name_448}', f'{dataset_prefix}/{save_data_name_449}', f'{dataset_prefix}/{save_data_name_450}', f'{dataset_prefix}/{save_data_name_451}', f'{dataset_prefix}/{save_data_name_452}', f'{dataset_prefix}/{save_data_name_453}', f'{dataset_prefix}/{save_data_name_454}', f'{dataset_prefix}/{save_data_name_455}', f'{dataset_prefix}/{save_data_name_456}', f'{dataset_prefix}/{save_data_name_457}', f'{dataset_prefix}/{save_data_name_458}', f'{dataset_prefix}/{save_data_name_459}', f'{dataset_prefix}/{save_data_name_460}', f'{dataset_prefix}/{save_data_name_461}', f'{dataset_prefix}/{save_data_name_462}',
            ]
        else:
            raise ValueError('num_train_objects not supported')
        
    if not predict_two_goals:
        dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage, is_pickle=True, use_all_data=use_all_data)    
    else:
        dataset = PredictTwoGoalsDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage, is_pickle=True, use_all_data=use_all_data)
    return dataset