# Using Singularity on Autobot

### For writables:

```
singularity shell --writable -B /project_data/held/mnakuraf/tax3d-conditioned-mimicgen:/mnt/tax3d-conditioned-mimicgen \
    --nv \
    --bind /usr/local/cuda-11.8:/usr/local/cuda mimicgen_tasks/
```

```
cd /project_data/held/mnakuraf
singularity build mimicgen_tasks.sif mimicgen_tasks
```

### On Autobot
```
singularity shell --bind /project_data/held/mnakuraf/tax3d-conditioned-mimicgen/:/mnt/tax3d-conditioned-mimicgen/ \
    --bind /usr/local/cuda-11.8:/usr/local/cuda \
    --nv /project_data/held/mnakuraf/mimicgen_tasks.sif
```

### On Local
```
singularity shell --bind /data/minon/tax3d-conditioned-mimicgen/:/mnt/tax3d-conditioned-mimicgen/ \
    --bind /usr/local/cuda-12.1:/usr/local/cuda \
    --nv /data/minon/containers/mimicgen_tasks.sif
```

### Start-Up Commands
3. In Singularity, run
```
source /opt/miniconda/etc/profile.d/conda.sh
conda activate equidiff
cd /mnt/tax3d-conditioned-mimicgen/
DISPLAY_NUM=99
export DISPLAY_NUM
Xvfb :${DISPLAY_NUM} -screen 0 1024x768x24 &
export DISPLAY=:${DISPLAY_NUM}.0
ps aux | grep Xvfb
export MUJOCO_GL=glx
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1
```
 
#### No Singularity
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH