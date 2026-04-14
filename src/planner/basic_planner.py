import numpy as np
import torch

from typing import Dict, Tuple
import mmengine
from src.utils.timer import Timer

from src.slam.slam_model import SlamModel
from src.utils.general_utils import InfoPrinter
from src.planner.utils.utils import query_sdf_np


def is_collision_free(
                      pa            : np.ndarray,
                      pb            : np.ndarray,
                      sdf_map       : np.ndarray,
                      step_size     : float = 1,
                      collision_thre: float = 0.1
                    ) -> Tuple : 
    """ check if collision free between pa and pb using sdf values in between

    Args:
        pa (np.ndarray, [3])         : point A
        pb (np.ndarray, [3])         : point B
        sdf_map (np.ndarray, [X,Y,Z]): SDF volume
        step_size (float)            : rrt step size
        collision_thre (float)       : collision threshold. Unit: voxel
    
    Returns:
        Tuple: num_collision_free, complete_free
            - num_collision_free (int): number of collision-free points in between
            - complete_free (bool): is pa->pb completely collision free
    """
    ### sample points in between with a step < rrt_step_size/5 ###
    points = np.linspace(pa, pb, num=int(np.ceil(np.linalg.norm(pb - pa) / (step_size / 5))) + 1)
    if len(points) == 0: # TODO-1020: temporary
        return 1, True
    points_sdf = query_sdf_np(sdf_map, points)

    ### check collision ###
    points_sdf[points_sdf == None] = -1
    collision_check = (points_sdf > collision_thre)
    
    """
    FIXME: there can be potential issue!
    if agent moves to a location where simulator doesn't give collision but incorrect sdf map give collision
    Agent can be trapped and stayed without moving out
    """
    ### get number of collision-free points ###
    if collision_check.sum() == len(collision_check):
        num_collision_free = max((len(collision_check) - 1) // 5, 1)
        complete_free = True
    else:
        num_collision_free = (np.argmax(~collision_check) - 1) // 5
        complete_free = False
        print("collision sdf: ", np.min(points_sdf))
    return num_collision_free, complete_free

class BasicPlanner():
    def __init__(self, 
                 main_cfg: mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        self.main_cfg = main_cfg
        self.planner_cfg = main_cfg.planner
        self.info_printer = info_printer
        self.step = 0
        self.step_size = self.planner_cfg.step_size
        
        self.init_timer()
        self.state = "staying"#"planning"
        
        self.path = None
        self.initial_path = None
        self.reset_goal = False

    def set_grid_pts(self, bbox):
        self.Nx = int((bbox[0,1] - bbox[0,0]) / self.voxel_size + 0.0005) + 1
        self.Ny = int((bbox[1,1] - bbox[1,0]) / self.voxel_size + 0.0005) + 1
        self.Nz = int((bbox[2,1] - bbox[2,0]) / self.voxel_size + 0.0005) + 1
        x_range = torch.arange(self.Nx)
        y_range = torch.arange(self.Ny)
        z_range = torch.arange(self.Nz)
        self.gs_x, self.gs_y, self.gs_z = torch.meshgrid(x_range, y_range, z_range, indexing="ij")
        self.grid_pts = torch.stack([self.gs_x, self.gs_y, self.gs_z], dim=-1) * self.voxel_size + bbox[:,0]
        
        self.bbox = bbox
        
    def update_step(self, step):
        """ update step information
    
        Args:
            step (int): step size
    
        """
        self.step = step
    
    def init_timer(self):
        """ initialize timer if requested
        Attributes:
            timer (Timer): timer object
            
        """
        self.timer = Timer()
        if self.planner_cfg.get("enable_timing", False):
            self.enable_timing = True
        else:
            self.enable_timing = False
    
    def update_sim(self, sim):
        """ initialize/update a Simulator if requested
        Attributes:
            sim (Simulator): Simulator object
            
        """
        self.sim = sim

    def update_state(self, 
                     sdf_vol    : np.ndarray,
                     cur_pose   : np.ndarray,
                     is_new_vols: bool
                     ) -> None:
        if self.state == "planning":
            is_goal_reachable = self.check_goal_reachable()
            self.state = "movingToGoal" if is_goal_reachable else "staying"
        elif self.state == "movingToGoal":
            is_goal_reached = self.check_goal_reached()
            if is_goal_reached:
                self.state = "planning"
                self.reset_goal = True if is_goal_reached else False
            else:
                next_pt_loc = self.path[1][:3,3]
                
                is_collision_detected = self.detect_collision(
                                            cur_pose    = cur_pose,
                                            sdf_map     = sdf_vol,
                                            next_pt_loc = next_pt_loc,
                                            collision_thre= self.main_cfg.tare.collision_threshold
                                            )
                
                if is_collision_detected:
                    self.state = "staying"
                else:
                    self.state = "movingToGoal"
        elif self.state == "staying":
            is_new_map_received = self.check_new_map_received(is_new_vols)
            self.state = "planning" if is_new_map_received else "staying"

    def check_goal_reached(self, cur_pose) -> bool:
        """ check if goal is reached
        Returns:
            is_goal_reached (bool): is goal reached
            
        """
        is_goal_reached = len(self.path) == 0
        return is_goal_reached
    
    def compute_next_state_pose(self, 
                                slam_method: str,
                                slam       : SlamModel,
                                cur_pose   : np.ndarray,
                                ) -> np.ndarray:
        print("cur state:", self.state)
        if self.state == "planning":
            planning_out = self.uncertainty_aware_planning(slam_method, slam, cur_pose, reset_goal=self.reset_goal)
            if not self.reset_goal:
                self.reset_goal = True
            # self.is_goal_reachable = planning_out['is_goal_reachable']
            self.path = planning_out['path']
            self.initial_path = planning_out['initial_path']
            new_pose = cur_pose.copy()
        elif self.state == "movingToGoal":
            is_goal_reached = self.check_goal_reached()
            if is_goal_reached:
                self.state = "planning"
            else:
                print("path", self.path)
                next_pose = self.path[1]
                # is_collision_detected = self.detect_collision(
                #                             sdf_vol     = sdf_vol,
                #                             cur_pose    = cur_pose,
                #                             next_pt_loc = next_pt_loc
                #                             )
                # if is_collision_detected:
                #     self.state = "staying" 
                # else:
                #     self.state = "movingToGoal"

                new_pose = self.moving_to_goal(next_pose)
                self.path.pop(0)
                self.initial_path.pop(0)
        elif self.state == "staying":
            new_pose = cur_pose.copy()
        else:
            raise NotImplementedError
        
        return new_pose
    
    def uncertainty_aware_planning(self,
                                   slam_method: str,
                                   slam       : SlamModel,
                                   cur_pose   : np.ndarray,
                                   reset_goal : bool = True
                                   ) -> Dict:
        raise NotImplementedError
    
    def update_occ_grid_data(self, grid_pts):
        self.ps.update_occ_grid_data(grid_pts)
    
    def loc2vox(self, loc, bbox=None, voxel_size=None):
        """ convert metric coordinates to voxel coordinates.
    
        Args:
            loc (np.ndarray, [3])   : metric coordinates
            bbox (np.ndarray, [3,2]): bounding box corner coordinates. Use self.bbox if not provided
            voxel_size (float)      : voxel size. Unit: meter. Use self.bbox if not provided
    
        Returns:
            vox (np.ndarray, [3]): voxel coordinates
        """
        bbox = bbox if bbox is not None else self.bbox
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size

        vox = (loc - bbox[:, 0]) / voxel_size
        return vox
    
    def detect_collision(self,
                         cur_pose   : np.ndarray,
                         sdf_map,
                         next_pt_loc: np.ndarray,
                         collision_thre
                         ) -> bool:
        
        ##################################################
        ### Run simulation at next location
        ##################################################
        next_c2w_sim = cur_pose.copy()
        next_c2w_sim[:3, 3] = next_pt_loc
        
        ### simulate ERP depth at next-state pose ###
        # _, _, _, erp_depth, _ = self.sim.simulate(next_c2w_sim, return_erp=True, no_print=True)
        _, _, _, erp_depth = self.sim.simulate(next_c2w_sim, return_erp=True, no_print=True)
        
        dist_closest = erp_depth.min()
        invalid_region_ratio = (erp_depth>1e6).sum() / (erp_depth.shape[0] * erp_depth.shape[1]) # invalid depths are set as large values
        
        ##################################################
        ### check collision from SDF
        ##################################################
        cur_pt_vxl = self.loc2vox(cur_pose[:3, 3])
        next_pt_vxl = self.loc2vox(next_pt_loc)
        num_collision_free, sdf_collision_free = is_collision_free(
                                cur_pt_vxl, 
                                next_pt_vxl, 
                                sdf_map,
                                step_size=self.step_size,
                                collision_thre=collision_thre
                                )
        
        ##################################################
        ### collision detection
        ##################################################
        invalid_region_ratio_thre = self.planner_cfg.get("invalid_region_ratio_thre", 0.2) 
        if self.main_cfg.general.dataset == 'Replica':
            is_collision_detected = not(sdf_collision_free)
        elif self.main_cfg.general.dataset == 'MP3D' or 'gibson':
            is_collision_detected = invalid_region_ratio > invalid_region_ratio_thre or not(sdf_collision_free)
            if self.use_dist_closest:
                is_collision_detected = is_collision_detected or dist_closest < self.planner_cfg.collision_dist_thre
        # elif self.main_cfg.general.dataset == 'gibson':
        #     is_collision_detected = not(sdf_collision_free)
        #     # is_collision_detected = invalid_region_ratio > invalid_region_ratio_thre or not(sdf_collision_free)
        else:
            raise NotImplementedError

        ### Collision Detected ###
        if is_collision_detected:
            self.info_printer("Collision Detected!", 
                          self.step, self.__class__.__name__)   
            self.info_printer(f"    Invalid region ratio: {invalid_region_ratio:.3f}", 
                            self.step, self.__class__.__name__)
            self.info_printer(f"    SDF collision free: {sdf_collision_free}", 
                            self.step, self.__class__.__name__)
            self.info_printer(f"    Observation distance: {dist_closest*100:.3f}cm", 
                            self.step, self.__class__.__name__)
            # print(f"Collision Detected! step: {self.step}, invalid_region_ratio: {invalid_region_ratio:.3f}, " \
            #       f"sdf_collision_free: {sdf_collision_free}, observation distance: {dist_closest*100:.3f}cm")
        return is_collision_detected
    
    def moving_to_goal(self, 
                       cur_pose: np.ndarray,
                       new_pose: np.ndarray,
                       ) -> np.ndarray:     
        return new_pose
        
    def check_goal_reachable(self) -> bool:
        """ Check if goal is reachable
        Returns:
            is_goal_reachable (bool): is goal reachable
        """
        # is_goal_reachable = self.is_goal_reachable
        return True#is_goal_reachable
    
    def check_goal_reached(self) -> bool:
        """ check if goal is reached
        Returns:
            is_goal_reached (bool): is goal reached
            
        """
        is_goal_reached = len(self.path) == 1
        return is_goal_reached
    
    def check_new_map_received(self, is_new_vols):
        """ check if new map is received.
    
        Args:
            is_new_vols (bool): is new volumes/map received

        Returns:
            is_new_vols (bool): is new volumes/map received        
            
        """
        return is_new_vols