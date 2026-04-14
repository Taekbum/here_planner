#!/bin/bash
##################################################
### This script is to run the full HERE system 
### (active planning and active ray sampling) 
###  on the gibson dataset.
##################################################

# Input arguments
scene=${1:-Cantwell}
num_run=${2:-1}
EXP=${3:-HERE} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${4:-0}

PROJ_DIR=${PWD}
DATASET=gibson
RESULT_DIR=${PROJ_DIR}/results

##################################################
### Random Seed
###     also used to initialize agent pose 
###     from indexing a set of predefined pose/traj 
##################################################
seeds=(0 1224 4869 8964 1000) 
seeds=("${seeds[@]:0:$num_run}")

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=( Cantwell Denmark Eastville Elmira Eudora Greigsville Pablo Ribera Swormville )
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
for scene in $selected_scenes
do
    for i in "${!seeds[@]}"; do

        if [ "$scene" == "Cantwell" ]; then
            max_iter=2000
        elif [ "$scene" == "Eastville" ]; then
            max_iter=2000
        elif [ "$scene" == "Swormville" ]; then
            max_iter=2000
        else
            max_iter=1000
        fi
        echo "max iter is ${max_iter}"
        echo "run $((i + 1))" 

        seed=${seeds[$i]}

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}
        mkdir -p ${result_dir}

        ### run experiment ###
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        python src/here/main.py --cfg ${CFG} --seed ${seed} --result_dir ${result_dir} --enable_vis ${ENABLE_VIS}

        ### evaluation ###
        bash scripts/evaluation/eval_gibson.sh ${scene} ${i} ${max_iter} ${EXP}
    done
done
