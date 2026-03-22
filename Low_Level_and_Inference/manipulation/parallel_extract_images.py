import os
import logging
import argparse
import numpy as np
from mino_utils.command_runner import ParallelExecutor
from manipulation.extract_images_from_states import get_all_expert_angles

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Parallel CPU Image Extraction Launcher")
    
    # Path Arguments
    parser.add_argument("--folder_name", type=str, default="data/diverse_objects_all")
    parser.add_argument("--exp_name", type=str, default="test_gen_demo")
    parser.add_argument("--extract_name", type=str, required=True, help="Object ID, e.g., 41510")
    parser.add_argument("--randomize_camera", type=int, default=0, help="0: No randomization, 1: Left-Right, 2: Full Randomization")
    parser.add_argument("--observation_mode", type=str, default='image_pointmap', help="Observation mode to determine which data to extract")
    parser.add_argument("--save_path", type=str, default="data/rgb_mino_data/", help="Directory to save extracted images")
    # Execution Arguments
    parser.add_argument("--max_workers", type=int, default=32, help="Number of concurrent CPU processes")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--extract_pcds", action='store_true')

    args = parser.parse_args()

    # 1. Identify Experiment Folder
    experiment_folder = os.path.join(args.folder_name, args.extract_name, "experiment", args.exp_name)
    if not os.path.exists(experiment_folder):
        experiment_folder = os.path.join(args.folder_name, args.extract_name)
        if not os.path.exists(experiment_folder):
            log.error(f"Folder not found: {experiment_folder}")
            return

    # 2. Get trajectories and calculate Global Threshold once
    all_experiments = sorted([x for x in os.listdir(experiment_folder) 
                             if os.path.isdir(os.path.join(experiment_folder, x))])
    
    log.info(f"Calculating threshold for {len(all_experiments)} trajectories...")
    exp_paths = [os.path.join(experiment_folder, exp) for exp in all_experiments]
    _, all_angles = get_all_expert_angles(exp_paths)
    
    global_threshold = np.quantile(all_angles, 0.1) if len(all_angles) > 0 else 0.0
    log.info(f"Using Global Threshold: {global_threshold:.4f}")

    runner = ParallelExecutor(
        max_workers=args.max_workers,
        timeout=args.timeout,
        logger=log
    )

    all_commands = []
    for i in range(len(all_experiments)):
        # import pdb; pdb.set_trace();
        if args.extract_pcds:
            cmd = (
                f"pixi run python manipulation/extract_images_from_states.py "
                f"--folder_name {args.folder_name} "
                f"--exp_name {args.exp_name} --extract_name {args.extract_name} "
                f"--traj_idx {i} --angle_threshold {global_threshold} "
                f"--randomize_camera {args.randomize_camera} --observation_mode {args.observation_mode} "
                f"--save_path {args.save_path} --extract_pcds"
            )
        else:
            cmd = (
                f"pixi run python manipulation/extract_images_from_states.py "
                f"--folder_name {args.folder_name} "
                f"--exp_name {args.exp_name} --extract_name {args.extract_name} "
                f"--traj_idx {i} --angle_threshold {global_threshold} "
                f"--randomize_camera {args.randomize_camera} --observation_mode {args.observation_mode} "
                f"--save_path {args.save_path}"
            )
        all_commands.append(cmd)
    runner.add_commands(all_commands)
    runner.log_commands()

    # 5. Run
    log.info(f"Starting parallel CPU execution with {args.max_workers} workers...")
    command_log_dir = os.path.join("logs", "extraction", args.extract_name)
    runner.run(command_log_dir=command_log_dir)
    
if __name__ == "__main__":
    main()