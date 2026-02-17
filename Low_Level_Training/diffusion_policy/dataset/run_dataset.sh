#!/bin/bash
export COPPELIASIM_ROOT=/project_data/held/pratik/.../CoppeliaSim_Player_V4_1_0_Ubuntu18_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
python3 "$@"