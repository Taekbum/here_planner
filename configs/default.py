import os
import numpy as np

##################################################
### HERE (General)
##################################################
general = dict(
    seed     = 0,                                       # random seed
    dataset  = "gibson",                                # dataset name
    scene    = "Denmark",                               # scene name
    num_iter = 1000,                                    # number of optimization iteration
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir   = "data/",                               # data directory
    result_dir = "results/",                            # result directory
    cfg_dir    = os.path.join(                          # configuration directory
        "configs", 
        general['dataset'], 
        general['scene']
        )
)

##################################################
### Simulator
##################################################
sim = dict(
    method = "habitat"                                  # simulator method
)

## HabitatSim ##
if sim["method"] == "habitat":
    sim.update(
        habitat_cfg = os.path.join(                     # HabitatSim configuration
            dirs['cfg_dir'], 
            "habitat.py"
            ),
    )

##################################################
### SLAM Model
##################################################
slam = dict(
    method="coslam"                                     # SLAM backbone method
)

if slam["method"] == "coslam":
    slam.update(
        room_cfg = f"{dirs['cfg_dir']}/coslam.yaml",    # Co-SLAM room configuration
        enable_active_planning = True,                  # enable/disable active planning
        enable_active_ray = True,                       # enable/disable active ray sampling
        
        use_traj_pose = False,                          # use pre-defined trajectory pose
        SLAMData_dir = os.path.join(                    # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
            dirs["data_dir"], 
            "Replica", general['scene']
            ),

        start_c2w = np.eye(4),                          # initial camera pose

        ### volumes ###
        voxel_size = 0.1,                               # Voxel size for Uncertainty/SDF volume. Unit: meter
    )

    ### Active Ray Sampling ###
    if slam.get('active_ray', True):
        slam.update(
            act_ray_oversample_mul = 4,                 # oversampling multiplier
            act_ray_num_uncert_sample = 500,            # number of unceratinty samples for replacing original samples
        )


##################################################
### Planner
##################################################
planner = dict(
    method        = "here",                             # planner method
    enable_timing = False,                              # enable timing message
)

### HERE Planner ###
if planner["method"] == "here":
    planner.update(
        step_size = 0.1,                                # step size. Unit: meter

        ### Uncertainty Volume ###
        voxel_size = slam.get("voxel_size", 0.1),       # Uncertainty Volume voxel size. Unit: meter
        
        ### Goal Space (Uncertainty Aggregation) ###
        uncert_top_k        = 4000,                     # number of top-k uncertainty to be considered in Goal Space
        uncert_top_k_subset = 300,                      # subset number of top-k uncertainty to be considered in Goal Space. Choose randomly from uncert_top_k to avoid Uncertainty Point Concentration
        gs_sensing_range    = [0.5, 2],                 # meter. goal space sensing range
        safe_sdf            = 0.8,                      # Unit: voxel
        force_uncert_aggre  = False,                    # force running uncertainty aggregation
        gs_z_levels         = None,                     # goal space z levels

        ### path planning ###
        obs_per_goal            = 10,                   # maximum uncertain observation per goal
        enable_uncert_filtering = True,                 # filter uncertainty volume according to the traversability
        up_dir                  = np.array([0, 0, 1]), # up direction for planning pose

        ### Collision ###
        invalid_region_ratio_thre = 0.5,                # invalid region ratio threshold by checking ERP
        collision_dist_thre       = 0.05,               # Unit: meter

        ### Rotation planning ###
        max_rot_deg = 10,                               # degree

    )

##################################################
### Visualization
##################################################
visualizer = dict(
    vis_rgbd        = True,                             # visualize RGB-D
    
    ### mesh related ###
    mesh_vis_freq = 500,                                # mesh save frequency

    ### comprehensive visualizer ###
    method = "here"                                   # comprehensive visualizer method
)

if visualizer["method"] == "here":
    visualizer.update(
        enable_all_vis       = False,                   # enable comprehensive visualization data
        save_rgbd            = True,                    # save RGB-D data
        save_pose            = True,                    # save pose
        save_state           = True,                    # save planner state
        save_goal            = True,                    # save goal
        save_occ_map         = True,                    # save occupancy map
        save_color_mesh      = True,                    # save colored mesh
        save_uncert_mesh     = True,                    # save uncertainty mesh
        save_mesh_freq       = 5,                       # frequency to save mesh for visualization
        save_mesh_voxel_size = 0.1,                    # voxel size to save mesh for visualization
    )