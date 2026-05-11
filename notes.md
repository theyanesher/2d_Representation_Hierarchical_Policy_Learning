
# Square D2

To generate dataset:
```
python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/square_d2/square_d2_abs.hdf5 --output data/robomimic/datasets/square_d2/square_d2_pcd_abs_512.hdf5 --num_workers=30
```

To train:
```
torchrun --standalone --nproc_per_node=2 mimicgen_train_ddp.py \
        --config-name articubot_square_d2_ddp n_demo=1000
```

```
python eval.py --config-name=articubot_square_d2 task_name=square_d2
```

# Three Piece Assembly D2

To generate dataset:

```
python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/three_piece_assembly_d2/three_piece_assembly_d2_abs.hdf5 --output data/robomimic/datasets/three_piece_assembly_d2/three_piece_assembly_d2_pcd_abs_512.hdf5 --num_workers=30
```

To train:
```
python mimicgen_train.py --config-name articubot_three_piece_assembly_d2_train n_demo=1000
```

To eval:
```
python eval.py --config-name=articubot_three_piece_assembly_d2 task_name=three_piece_assembly_d2
```
