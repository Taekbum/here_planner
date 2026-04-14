# HERE: Hierarchical Active Exploration of Radiance Field With Epistemic Uncertainty Minimization (RA-L 2026)

<a href='https://taekbum.github.io/here'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/pdf/2601.07242'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?si=4QR8HR8PZVSvw1hH&v=FkY7t2IeWhE&feature=youtu.be)


This is an official implementation of the paper "HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization". 

[Taekbeom Lee](https://ziyue.cool/)<sup>\*1</sup>, 
[Dabin Kim](https://huangying-zhan.github.io/)<sup>\*1</sup>, 
[Youngseok Jang]()<sup>1</sup>, 
[H. Jin Kim]()<sup>1</sup>

<sup>1</sup> Seoul National University

<sup>*</sup> Equal contribution </br>


<img src="assets/simulation.gif" width="800" height="400"> 

## Table of Contents
- [&nbsp; Installation](#installation) 
- [&nbsp; Dataset Preparation](#dataset-preparation)
- [&nbsp; Run HERE](#run_here)
- [&nbsp; Evaluation](#evaluation)
- [&nbsp; License](#license)
- [&nbsp; Acknowledgement](#acknowledgement)
- [&nbsp; Citation](#citation)

<h2 id="installation"> Installation </h2>

**NOTE**: This code has been tested on Ubuntu 20.04 and 22.04. On Ubuntu 22.04,  
A minor fix is required for ```third_parties/coslam``` -- add ``` #include <limits>``` at the top of  
```third_parties/coslam/external/NumpyMarchingCubes/marching_cubes/src/marching_cubes.cpp```. 

```
# Clone with the required third parties
git clone --recursive https://github.com/Taekbum/here_planner
cd here_planner

# Build conda environment
bash scripts/installation/build.sh

# Activate conda env
conda activate here
```

<h2 id="dataset-preparation"> Dataset Preparation   </h2>

To dowlnload the datasets, please refer to the instruction in [Gibson](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md#download-gibson-database-of-spaces) and [Matterport3D](https://niessner.github.io/Matterport/).

The download scripts are not included here as there are  __Term of Use agreements__ for using the datasets. 

After you obtain the download scripts, run following commands:

```
# Example use of the Gibson download script:
python download_mp.py -o data/MP3D --task_data gibson

# Example use of the Matterport3D download script:
python download_mp.py -o data/MP3D --task_data habitat
```

The data folder should be structed as
```
├── data
│   ├── gibson
│   │   └── Ackermanville.glb
│   │   └── ...
│   ├── MP3D
│   │   └── v1
│   │       └── tasks
│   │           └── mp3d
│   │               ├── 1LXtFkjw3qL
│   │               ├── ...
```

<h2 id="run_here"> Run HERE </h2>

We provide the script to run the full system described in the paper. 
This script also includes the upcoming [evaluation process](#evaluation).


```
# Run Gibson 
bash scripts/here/run_gibson.sh {SceneName/all} {NUM_TRIAL} {EXP_NAME} {ENABLE_VIS}

# Run MP3D 
bash scripts/here/run_mp3d.sh {SceneName/all} {NUM_TRIAL} {EXP_NAME} {ENABLE_VIS}

# examples
bash scripts/here/run_gibson.sh Denmark 1 HERE 1
bash scripts/here/run_mp3d.sh gZ6f7yhEvPG 1 HERE 0
bash scripts/here/run_gibson.sh all 5 HERE 0
```


<h2 id="evaluation"> Evaluation  </h2>

We evaluate the reconstruction using the following metrics with a threshold of 5cm: 

- Accuracy (cm)
- Completion (cm)
- Completion ratio (%) 

We also compute the mean absolute distance (MAD, in cm) between the estimated SDF values at all vertices and those of the ground-truth mesh. For these vertices, we further evaluate the calibration between the estimated uncertainty and the SDF distance using the Area Under the Sparsification Error (AUSE) metric.

### Quantitative Evaluation

```
# Evaluate Gibson result
bash scripts/evaluation/eval_gibson.sh {SceneName/all} {TrialIndex} {IterationToBeEval} {EXP_NAME}
# Evaluate MP3D result
bash scripts/evaluation/eval_mp3d.sh {SceneName/all} {TrialIndex} {IterationToBeEval} {EXP_NAME}

# Examples
bash scripts/evaluation/eval_gibson.sh Denmark 0 2000 HERE
bash scripts/evaluation/eval_mp3d.sh gZ6f7yhEvPG 0 5000 HERE
```

### Qualitative Evaluation

```
# Visualization of Mesh Evolution (Color / Uncertainty)
bash scripts/evaluation/visualize_gibson.sh {SceneName/all} {TrialIndex} {EXP_NAME} {MESH_TYPE} {CAM_VIEW} {SAVE_VIS} {INTERACT}

# examples
bash scripts/evaluation/visualize_gibson.sh Denmark 0 HERE uncert_mesh src/visualization/default_camera_view.json 1 0
bash scripts/evaluation/visualize_gibson.sh Cantwell 1 HERE color_mesh src/visualization/default_camera_view.json 0 0
```

### Evaluation Notes (Dataset Limitations)
In some scenes (e.g., Greigsville in Gibson), the ground-truth meshes are not fully watertight, which occasionally allows the agent to escape the indoor environment through holes. In such cases, exploration may fail (e.g., the agent gets stuck outside the scene), which is unrelated to the proposed method itself.  
For fair evaluation, the results reported in the paper exclude these failure cases. If a watertight mesh is used as the ground-truth indoor scene, this issue does not occur and all runs can be evaluated consistently.


<h2 id="license"> License  </h2>

This project is based on code from [NARUTO](https://github.com/oppo-us-research/NARUTO), which is licensed under the [MIT licence](LICENSE). For the third parties, please refer to their license. 

- [CoSLAM](https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE): Apache-2.0 License
- [HabitatSim](https://github.com/facebookresearch/habitat-sim/blob/main/LICENSE): MIT License
- [neural-slam-eval](https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE): Apache-2.0 License
- [python-tsp](https://github.com/fillipe-gsm/python-tsp): MIT License. Vendored under `src/planner/tare/python_tsp`; see `src/planner/tare/python_tsp/LICENSE`.


<h2 id="acknowledgement"> Acknowledgement  </h2>

We thank the authors of the following open-source projects used in our code: [NARUTO](https://github.com/oppo-us-research/NARUTO), [CoSLAM](https://github.com/HengyiWang/Co-SLAM).



<h2 id="citation"> Citation  </h2>

```
@article{lee2026here,
  title={HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization},
  author={Lee, Taekbeom and Kim, Dabin and Jang, Youngseok and Kim, H Jin},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  publisher={IEEE}
}
```
