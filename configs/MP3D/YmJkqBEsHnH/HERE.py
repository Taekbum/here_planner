import numpy as np
import os

_base_ = "../../default.py"

##################################################
### NARUTO (General)
##################################################
general = dict(
    dataset  = "MP3D",
    scene    = "YmJkqBEsHnH",
    num_iter = 5000,
    checkpoint_freq = 20,
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir = "data/",
    result_dir = "",
    cfg_dir = os.path.join("configs", general['dataset'], general['scene'])
)


##################################################
### Simulator
##################################################
if _base_.sim["method"] == "habitat":
    _base_.sim.update(
        habitat_cfg = os.path.join(dirs['cfg_dir'], "habitat.py")
    )

##################################################
### SLAM
##################################################
if _base_.slam["method"] == "coslam":
    _base_.slam.update(
        room_cfg        = f"{dirs['cfg_dir']}/coslam.yaml",   # Co-SLAM room configuration
        enable_active_planning = True,                             # enable/disable active planning
        active_ray      = True,                             # enable/disable active ray sampling
        
        SLAMData_dir = None, 

        start_c2w = np.array([
            [ 0,  0,  1,  0],
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  1],
            [ 0,  0,  0,  1]])
    )

##################################################
### Planner
##################################################
planner = dict(
    method= "here",
    up_dir = np.array([0, 0, 1]), # up direction for planning pose
    vmax=0.5,
    ang_max=0.875,
    v_ang_max=1.5,
    time_step=1,
    planning_bound = np.array([[-12.9, 4.1],[-5.5,1.3]]),
    exclude_init_uncert=True
)

##################################################
### Visualization
##################################################
visualizer = dict(
    vis_rgbd        = True,                             # visualize RGB-D

    ### mesh related ###
    mesh_vis_freq = 500,                                # mesh save frequency

    ## for visualizing ## 
    enable_all_vis       = True,                    # enable comprehensive visualization data
    save_color_mesh      = False                    # save colored mesh
)

######################################################
### Evaluate Monte Carlo Tree Search-based Planning
######################################################
viewpoint_manager = dict(
    number_x=20,
    number_y=20,
    number_z=1,
    resolution_x=2.0,
    resolution_y=2.0,
    resolution_z=5.0
)
tare = dict(
    cell_exploring_to_covered_th=1,
    cell_covered_to_exploring_th=10,
    cell_exploring_to_almost_covered_th=10,
    cell_almost_covered_to_exploring_th=20,
    cell_unknown_to_exploring_th=1,
    uncertain_threshold=4.5,
    collision_threshold=0.5,
    frontier_threshold=1, 
    run_global_planner=True
)
grid_world = dict(
    z_num=1,
    cell_height=0.,
    nearby_grid_num=3
)