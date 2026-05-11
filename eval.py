"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from equi_diffpo.workspace.base_workspace import BaseWorkspace

max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d0': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

def get_ws_y_center(task_name):
    return 0.

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'equi_diffpo','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)

    # three_piece_assembly_d2
    # output_dir = '/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/outputs/2025.05.14/03.35.59_train_dp3_three_piece_assembly_d2' # no noise, 46%
    # output_dir = '/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/outputs/2025.05.13/22.29.22_train_dp3_three_piece_assembly_d2' # noise, 2%
    # output_dir = 'data/outputs/2025.06.02/21.08.37_train_dp3_three_piece_assembly_d2' # 58%
    # output_dir = 'data/outputs/2025.05.25/02.47.03_train_dp3_three_piece_assembly_d2' # 42%

    # square_d2
    # output_dir = '/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/outputs/2025.05.20/20.50.52_train_dp3_square_d2' # no noise, 64%
    # output_dir = '/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/data/outputs/2025.05.21/12.46.41_train_dp3_square_d2' # noise, 46%

    # threading_d2
    # output_dir = 'data/outputs/2025.06.03/17.13.26_train_dp3_threading_d2' # 50%

    # coffee_d2
    # output_dir = 'data/outputs/2025.06.05/21.37.45_train_dp3_coffee_d2' # 86%

    # hammer cleanup
    # output_dir = 'data/outputs/2025.06.05/22.00.46_train_dp3_hammer_cleanup_d1' # 62%

    output_dir = cfg.low_level_dir
    # nut assembly

    workspace: BaseWorkspace = cls(cfg, output_dir)
    workspace.eval()

if __name__ == "__main__":
    main()