import os
import sys
sys.path.append(os.getcwd())

from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.simulator import init_simulator
from src.slam import init_SLAM_model
from src.planner import init_planner
from src.data.pose_loader import PoseLoader
from src.here.cfg_loader import argument_parsing, load_cfg
from src.visualization import init_visualizer

if __name__ == '__main__':
    info_printer = InfoPrinter("HERE")
    timer = Timer()
    
    args = argument_parsing()
    main_cfg = load_cfg(args)
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)
    
    fix_random_seed(main_cfg.general.seed)

    # initialize Simulator
    sim = init_simulator(main_cfg, info_printer)

    # initialize ESLAM
    slam = init_SLAM_model(main_cfg, info_printer)
    
    # initialize planner
    planner = init_planner(main_cfg, info_printer, slam, slam.bounding_box.cpu().numpy())
    planner.set_grid_pts(slam.bounding_box.cpu().numpy())
    planner.update_sim(sim)
    planner.init_data(slam.bounding_box.cpu().numpy())
    
    ##################################################
    ### initialize visualizer
    ##################################################
    visualizer = init_visualizer(main_cfg, info_printer)
    
    pose_loader = PoseLoader(main_cfg)
    c2w_slam = pose_loader.load_init_pose()

    for i in range(main_cfg.general.num_iter):
        ##################################################
        ### update module infomation (e.g. step)
        ##################################################
        update_module_step(i, [sim, slam, planner, visualizer])
        
        ##################################################
        ### load pose and transform pose
        ##################################################
        c2w_slam = pose_loader.update_pose(c2w_slam, i)
        c2w_sim = c2w_slam.cpu().numpy().copy()
        
        ##################################################
        ### Simulation
        ##################################################
        timer.start("Simulation", "General")
        color, depth = sim.simulate(c2w_sim)
        if main_cfg.visualizer.vis_rgbd:
            visualizer.visualize_rgbd(color, depth, slam.config["cam"]["depth_trunc"])
        timer.end("Simulation")
        
        ##################################################
        ### save data for comprehensive visualization
        ##################################################
        if main_cfg.visualizer.enable_all_vis:
            visualizer.main(slam, planner, color, depth, c2w_slam)
        
        ##################################################
        ### Mapping optimization
        ##################################################
        if i % main_cfg.general.checkpoint_freq == 0:
            slam.save_ckpt(i)
        timer.start("SLAM", "General")
        new_uncert_sdf_vols = slam.online_recon_step(i, color, depth, c2w_slam)
        timer.end("SLAM")
        
        ##################################################
        ### Active Planning
        ##################################################
        if main_cfg.slam.enable_active_planning:
            timer.start("Planning", "General")
            ### update map volumes ###
            if new_uncert_sdf_vols is not None:
                uncert_sdf = new_uncert_sdf_vols
                is_new_vols = True
            else:
                is_new_vols = False
            c2w_slam = planner.main(
                uncert_sdf, 
                c2w_slam.cpu().numpy(), 
                is_new_vols
                )
            timer.end("Planning")
    
    ##################################################
    ### Save Final Mesh and Checkpoint
    ##################################################
    slam.save_mesh(main_cfg.general.num_iter, voxel_size=slam.config['mesh']['voxel_final'], suffix='_final')
    slam.save_ckpt(main_cfg.general.num_iter, suffix="_final")

    ##################################################
    ### Runtime Analysis
    ##################################################
    timer.time_analysis()