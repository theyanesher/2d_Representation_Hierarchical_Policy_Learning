python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/square_d2/square_d2_1100.hdf5 --output data/robomimic/datasets/square_d2/square_d2_1100_pcd.hdf5 --num_workers=24

python equi_diffpo/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/square_d2/square_d2_1100_pcd.hdf5 -o data/robomimic/datasets/square_d2/square_d2_1100_pcd_abs.hdf5 -n 12

python eval.py --config-name=articubot_mimic_eval_oracle task_name=square_d2