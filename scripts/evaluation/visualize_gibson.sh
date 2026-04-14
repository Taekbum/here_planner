#!/bin/bash
##################################################
### This script is to visualize a trajectory on
### a given mesh. Trajectory poses are extracted
### from the given checkpoint
##################################################

### Input arguments ###
scene=${1:-Cantwell}
trial=${2:-0}
EXP=${3:-HERE}
mesh_type=${4:-uncert_mesh}
cam_view=${5:-src/visualization/default_camera_view.json}
save_vis=${6:-1}
with_interact=${7:-0}

PROJ_DIR=${PWD}
DATASET=gibson
RESULT_DIR=${PROJ_DIR}/results
trials=(0 1 2 3 4)

if [ "$trial" == "all" ]; then
    selected_trials=${trials[@]} # Copy all trials
else
    selected_trials=($trial) # Assign the matching scene
fi

scenes=(Cantwell Denmark Eastville Elmira Eudora Greigsville Pablo Ribera Swormville)
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

### visualize trajectory ###
for scene in $selected_scenes
do
    for i in $selected_trials; do
        ### get visualization folder ###
        vis_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}/visualization
        echo "==> Visualizing [${vis_dir}]"

        python src/visualization/here_o3d_visualizer.py \
        --vis_dir $vis_dir \
        --cam_json $cam_view \
        --save_vis $save_vis \
        --mesh_type $mesh_type \
        --with_interact $with_interact
    done
done
