from manipulation.sim import SimpleEnv
from manipulation.primitive_api import *
import gym

handle_name_dict = {
    'busket': 'handle',
    'faucet': 'switch',
    'foldingchair': 'seat',
    'laptop': 'screen_frame',
    'stapler': 'lid',
    'toilet': 'lid',
}

class articulated(SimpleEnv):

    def __init__(self, task_name, object_name, link_name, init_angle, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = dict()
        self.object_name = object_name
        self.link_name = link_name
        self.handle_name = handle_name_dict.get(object_name, 'handle')  # default handle name
        self.init_angle = init_angle
        # default link name
        # link_1 for stapler
        # link_0 for other objects

    def execute(self):
        rgbs, final_state = approach_object_link_parallel(self, self.object_name, self.link_name, debug=False)  
        return rgbs, final_state
    
    def set_handle(self, angle=0.5):
        link_pc, _, _, _ = self.get_link_pc(self.object_name, self.link_name)
        all_handle_pos, handle_joint_id, _, _ = self.get_handle_pos(self.object_name, return_median=False, custom_joint_name=self.handle_name)
        _, handle_joint_id, _, _ = get_link_handle(all_handle_pos, handle_joint_id, link_pc, threshold=0.02)
        cur_angle = p.getJointState(self.urdf_ids[self.object_name], handle_joint_id, physicsClientId=self.id)[0]
        p.resetJointState(self.urdf_ids[self.object_name], handle_joint_id, cur_angle + angle, physicsClientId=self.id)

    def reset(self, reset_state=None, object_name=None, open_gripper_at_reset=False):
        if object_name is None:
            object_name = self.object_name
        super().reset(reset_state, object_name, open_gripper_at_reset)
        # if self.object_name == 'laptop':
        #     self.set_handle(angle=-np.pi/2)
        # if self.object_name == 'toilet':
        #     self.set_handle(angle=np.pi/6)
        if self.object_name == 'bucket' or self.object_name == 'laptop' or self.object_name == 'toilet':
            if reset_state is None and self.init_angle is not None and self.restore_state_file is None:
                self.set_handle(angle=self.init_angle) 
            
gym.register(
    id='articulated-v0',
    entry_point=articulated,
)