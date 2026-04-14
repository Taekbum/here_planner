import numpy as np
import torch
import copy
import time
import math
import random
from typing import Dict, List, Tuple

from .basic_planner import BasicPlanner
from .utils.kinematics import SE3Kinematics
from .utils.utils import spherical_fibonacci_sampling, is_collision_free, project_point_to_cuboid

from .tare.grid_world import GridWorld
from .tare.viewpoint_manager import ViewPointManager
from .tare.grid import OccupancyGrid
from .rotation_planning import rotation_planning, interpolate_rotation2, Mat2Rotation, Rotation2Mat, minimize_movement
from .tare.local_tsp import LocalTSPPlanner, set_path_between_pose_mat
from .tare.grid import CellStatus

import os
import pickle
import mmengine

class VisTargetPoses():
    def __init__(self, target_poses, target_uncs, topk_uncert_vxls, topk_uncert_vxls_all, seen_voxel_all_mask, cell_bounds):
        self.target_poses = target_poses
        self.target_uncs = target_uncs
        self.topk_uncert_vxls = topk_uncert_vxls
        self.topk_uncert_vxls_all = topk_uncert_vxls_all
        self.cell_bounds = cell_bounds
        self.seen_voxel_all_mask = seen_voxel_all_mask
        self.idx_sel = None
        self.lst_region_path = []
        self.lst_neighbor_path = []
        
class VerticeSplit():
    def __init__(self, vertices, cell_bound):
        self.vertices = vertices
        self.cell_bound = cell_bound 

def compute_camera_pose(A: np.ndarray, B: np.ndarray, up_dir: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """ compute camera pose given current location A and look-at location B.
    Using OpenGL (RUB) coordinate system. 
    up_dir is the up direction w.r.t world coorindate origin pose.

    Args:
        A (np.ndarray, [3])     : current location
        B (np.ndarray, [3])     : look-at location
        up_dir (np.ndarray, [3]): up direction in world coordinate
    
    Returns:
        M (np.ndarray, [3, 3]): rotation matrix
    """
    # viewing direction (backward)
    V = A - B

    ### FIXME: for edge case that target points in the same x,y position ###
    if V[0] == 0 and V[1] == 0:
        V[0] = 1e-6

    # right viewing direction
    R = np.cross(up_dir, V)

    # up viewing direction
    U = np.cross(V, R)

    # normalize
    V = V / np.linalg.norm(V)
    R = R / np.linalg.norm(R)
    U = U / np.linalg.norm(U)

    # construct pose matrix
    M = np.column_stack((R, U, V))  

    return M
    
    
class TarePlanner(BasicPlanner):
    def __init__(self, main_cfg, info_printer, slam, bound):
        super().__init__(main_cfg, info_printer)
        self.voxel_size = self.planner_cfg.voxel_size

        self.initial_position = np.zeros(3)
        
        viewpoint_config = main_cfg.viewpoint_manager
        viewpoint_resol = viewpoint_config.resolution_x
        
        # cell_size = viewpoint_number * viewpoint_resol / 10
        cell_size = viewpoint_resol
        row_num = np.ceil((bound[0][1] - bound[0][0])/cell_size).astype(np.int64)
        col_num = np.ceil((bound[1][1] - bound[1][0])/cell_size).astype(np.int64)
        
        tare_config = main_cfg.tare
        exploring_frontier_th = tare_config.frontier_threshold
        cell_exploring_to_covered_th = tare_config.cell_exploring_to_covered_th
        cell_covered_to_exploring_th = tare_config.cell_covered_to_exploring_th
        cell_exploring_to_almost_covered_th = tare_config.cell_exploring_to_almost_covered_th
        cell_almost_covered_to_exploring_th = tare_config.cell_almost_covered_to_exploring_th
        cell_unknown_to_exploring_th = tare_config.cell_unknown_to_exploring_th
        
        grid_world_config = main_cfg.grid_world
        self.subspace_world = GridWorld(row_num=row_num,
                                      col_num=col_num,
                                      level_num=grid_world_config.z_num,
                                      origin=np.array([bound[0][0], bound[1][0], bound[2][0]]),
                                      bound=bound,
                                      cell_size=cell_size,
                                      cell_height=grid_world_config.cell_height,
                                      nearby_grid_num=grid_world_config.nearby_grid_num,
                                      exploring_frontier_th=exploring_frontier_th,
                                      cell_exploring_to_covered_th=cell_exploring_to_covered_th,
                                      cell_covered_to_exploring_th=cell_covered_to_exploring_th,
                                      cell_exploring_to_almost_covered_th=cell_exploring_to_almost_covered_th,
                                      cell_almost_covered_to_exploring_th=cell_almost_covered_to_exploring_th,
                                      cell_unknown_to_exploring_th=cell_unknown_to_exploring_th
                                      )
        self.uncertain_threshold = tare_config.uncertain_threshold
        self.certain_threshold = tare_config.get("certain_threshold", 3.0)
        self.collision_threshold = tare_config.collision_threshold

        self.occupancy_map = OccupancyGrid(res=self.voxel_size, bound=bound, dimension=3, uncertainty_threshold=self.uncertain_threshold, collision_threshold=self.collision_threshold)

        self.subspace_world.set_env_bound(bound)
        self.subspace_world.set_region_graph(self.occupancy_map)
        self.region_graph_initialized = False
        
        print("Subspace generated")
        self.viewpoint_manager = ViewPointManager(bound, number_x=viewpoint_config.number_x,
                                                number_y=viewpoint_config.number_y,
                                                number_z=viewpoint_config.number_z,
                                                resolution_x=viewpoint_config.resolution_x,
                                                resolution_y=viewpoint_config.resolution_y,
                                                resolution_z=viewpoint_config.resolution_z,
                                                )
        self.run_global_planner = tare_config.run_global_planner

        # Local Planner
        robot = SE3Kinematics(vmax=self.planner_cfg.vmax, 
                              ang_max=self.planner_cfg.ang_max,
                              v_ang_max=self.planner_cfg.v_ang_max,
                              dt=self.planner_cfg.time_step
                              )
        self.robot = robot

        self.slam = slam
        self.bound = bound

        sim_max_vel = 0.1 # in simulation, maximum velocity 
        self.interval = sim_max_vel / self.planner_cfg.vmax
        
        self.local_tsp = LocalTSPPlanner(robot,
                                         interval=self.interval)

        self.binary_occupancy_map = None
        self.goal = None
        self.goal_region = None
        
        self.gp_planner_goal = False
        
        self.visited_positions = []
        self.global_path = []
        self.viz_global_path = []
        self.cluster_frontiers = []
        self.all_frontiers = []
        self.voxel_global_path = []
        self.gp_iter = 0
        
        self.rots = []
        self.cur_rot_planning_finished = False
        
        self.use_frontier_for_heuristic = True
        
        self.max_gp_iter = 10
        self.info_enough = True
        self.frontier_enough = False
        
        habitat_config = mmengine.Config.fromfile(self.main_cfg.sim.habitat_cfg)
        self.fov = np.deg2rad(habitat_config.camera.pinhole.fov).mean() #np.deg2rad(min(habitat_config.camera.pinhole.fov))
        self.fibonacci_samples = spherical_fibonacci_sampling(30)
        
        self.greedy = not self.planner_cfg.get("ablation", False)
        
        self.initial_uncert = slam.initial_uncert
        
        
        self.use_dist_closest = self.planner_cfg.get("use_dist_closest", False)
        self.exclude_init_uncert = self.planner_cfg.get("exclude_init_uncert", False)
        self.n_visible_threshold = self.planner_cfg.get("n_visible_threshold", 10)
        
    def goal_search(self, uncert, max_uncert=True, trial_num=0):
        # update occupancy grid to avoid occupied goal

        idxs = [self.gs_x_range, self.gs_y_range, self.gs_z_range]
        mesh = torch.meshgrid([torch.tensor(lst) for lst in idxs])
        occ_grid = self.binary_occupancy_map[mesh]
        uncert_downsize = uncert[mesh]
        
        # avoid occupied goal
        unsafe_tgt = ((self.gs_x<1) + (self.gs_x+1>=self.Nx) + \
                     (self.gs_y<1) + (self.gs_y+1>=self.Ny) + \
                     (self.gs_z<1) + (self.gs_z+1>=self.Nz)).numpy()  # FIXME(HY): newly added
        unsafe_tgt += occ_grid + \
                      self.binary_occupancy_map[(self.gs_x+1).clamp(0,self.Nx-1),self.gs_y,self.gs_z] + \
                      self.binary_occupancy_map[(self.gs_x-1).clamp(0,self.Nx-1),self.gs_y,self.gs_z]
        unsafe_tgt += self.binary_occupancy_map[self.gs_x,(self.gs_y+1).clamp(0,self.Ny-1),self.gs_z] + \
                      self.binary_occupancy_map[self.gs_x,(self.gs_y-1).clamp(0,self.Ny-1),self.gs_z]
        unsafe_tgt += self.binary_occupancy_map[self.gs_x,self.gs_y,(self.gs_z+1).clamp(0,self.Nz-1)] + \
                      self.binary_occupancy_map[self.gs_x,self.gs_y,(self.gs_z-1).clamp(0,self.Nz-1)]
        uncert_downsize[unsafe_tgt] = 0
        
        grid_pts_downsize = self.grid_pts[mesh]
        if max_uncert:
            max_uncert_vxl = np.argpartition(uncert_downsize, -1, axis=None)[-1]
        else:
            max_uncert_vxl = np.argpartition(uncert_downsize, -1, axis=None)[-1-trial_num]
        max_uncert_vxl = np.unravel_index(max_uncert_vxl, grid_pts_downsize.shape[:-1])
        goal = grid_pts_downsize[max_uncert_vxl[0], max_uncert_vxl[1], max_uncert_vxl[2]]
        return goal.numpy(), uncert_downsize

    def update_occupancy_map(self, uncert_sdf):
        '''
        occupancy map for global environment
        
        occupancy map: unkown 0 
                       occupied 1
                       unoccupied -1
        '''
        uncert_vol, sdf_vol = uncert_sdf
        # Filter
        uncertain_mask = uncert_vol < self.uncertain_threshold
        occupancy_map = -1.0 * np.ones(sdf_vol.shape) # initial: all are unoccupied
        inf_occupancy_map = -1.0 * np.ones(sdf_vol.shape) # inflated occupancy map, used for a-star planning
        occupied_idx = np.where(sdf_vol <= self.collision_threshold)
        occupancy_map[occupied_idx] *= -1.0 # change if occupied cell to 1.0
        occupancy_map *= uncertain_mask # for uncertain cell, multiply zero 
        binary_occupancy_map = 1 - (occupancy_map <0)

        inf_occupancy_idx = np.where(sdf_vol <= self.collision_threshold)  #+1.0)
        inf_occupancy_map[inf_occupancy_idx] *= -1.0
        bool_inf_occupancy_map = 1 - (inf_occupancy_map<0)
        
        self.occupancy_map.set_boolean_map_data(binary_occupancy_map)
        self.occupancy_map.set_boolean_inflated_map_data(bool_inf_occupancy_map)
        self.occupancy_map.set_raw_data(sdf_vol)
        self.occupancy_map.set_uncertainty_data(uncert_vol)
        self.occupancy_map.set_occupancy_data(occupancy_map)

        self.binary_occupancy_map = np.array(binary_occupancy_map,dtype=bool)
        
    
    def get_viz_global_path(self):
        return self.viz_global_path
        
    def main(self, 
             uncert_sdf_vols: List,
             cur_pose       : np.ndarray,
             is_new_vols    : bool
             ) -> torch.Tensor:
        """ Planner main function
    
        Args:
            uncert_sdf_vols (List)      : Uncertainty Volume and SDF Volume
                - uncert_vol (np.ndarray, [X,Y,Z]): uncertainty volume
                - sdf_vol (np.ndarray, [X,Y,Z])   : SDF volume
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            is_new_vols (bool)          : is uncert_sdf_vols new optimized volumes
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """     
        self.update_occupancy_map(uncert_sdf_vols)
        
        if not self.region_graph_initialized:
            self.subspace_world.update_region_graph(self.occupancy_map)
            self.region_graph_initialized = True
                    
        self.update_state(uncert_sdf_vols[1], cur_pose, is_new_vols) 
        self.info_printer(f"Current state: {self.state}", self.step, self.__class__.__name__)
        new_pose = self.compute_next_state_pose(cur_pose, uncert_sdf_vols, self.occupancy_map)
        new_pose = torch.from_numpy(new_pose).float()
        
        return new_pose

        
    def update_state(self, 
                     sdf_vol    : np.ndarray,
                     cur_pose   : np.ndarray,
                     is_new_vols: bool
                     ) -> None:
        if self.state == "planning":
            if self.info_enough:
                print("Switch To movingToGoal::planning Completed")
                self.state = "movingToGoal"
            else:
                print("Switch To planningToGlobalGoal::InformationGainIsTooLow")
                self.state = "planningToGlobalGoal"
        elif self.state == "planningToGlobalGoal":
            print("Switch To movingToGoal::planningToGlobalGoal Completed")
            self.state = "movingToGoal"
        elif self.state == "rotateOnGoal":
            if self.cur_rot_planning_finished:
                self.state = "planning"
        elif self.state == "movingToGoal":
            is_goal_reached = self.check_goal_reached(cur_pose[:3,3], self.goal[0])
            if self.gp_iter == self.max_gp_iter:
                print("Switch To RotationPlanning::GP_Iter_Max_Reached")
                return
            if is_goal_reached:
                print("Switch To RotationPlanning::GoalIsReached") # 
                self.state = "rotateOnGoal"
                return
            if len(self.path) == 0:
                print("Switch To Planning::PathIsEmpty")
                self.state = "planning"
                self.reset_goal = True
                return
                    
            next_pt_loc = self.path[0][:3,3]
            
            is_collision_detected = self.detect_collision(
                                        cur_pose    = cur_pose,
                                        sdf_map     = sdf_vol,
                                        next_pt_loc = next_pt_loc,
                                        collision_thre= self.main_cfg.tare.collision_threshold
                                        )
            
            if is_collision_detected:
                self.state = "planning"
            else:
                self.state = "movingToGoal"

        elif self.state == "staying":
            is_new_map_received = self.check_new_map_received(is_new_vols)
            self.state = "planning" if is_new_map_received else "staying"
            print("Switch To Planning::NewMapReceived")
                
    def check_goal_reached(self, cur_position, goal):
        if np.linalg.norm(cur_position[:2] - goal[:2]) < 0.1: # 
            print("Goal Reached!")
            return True
        return False

    def rotating_at_current_loc(self, cur_pose: np.ndarray, lst_rot:list) -> np.ndarray:
        """ rotating at the current location using the rotations in self.rots. 
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        rot = lst_rot.pop(0)
        new_pose = cur_pose.copy()
        new_pose[:3, :3] = rot
        return new_pose

    def compute_uncertain_voxels(self, cur_pose, uncert_sdf_vols):
        cur_vxl = self.occupancy_map.Pos2Sub(cur_pose[:3,3])
        cur_sub = self.subspace_world.subspaces.Pos2Sub(cur_pose[:3,3])
        cell_ind = self.subspace_world.subspaces.Sub2Ind(cur_sub)
        frontiers_in_cur_cell = np.array(self.subspace_world.subspaces.get_cell(cell_ind).frontiers)
        if self.planner_cfg.get("use_frontier_near_goal_for_heuristic", True) and len(frontiers_in_cur_cell) > 0:
            visible_mask = self.check_visibility(cur_vxl, frontiers_in_cur_cell, uncert_sdf_vols[1])
            lst_lookat_vxl = frontiers_in_cur_cell[visible_mask]
        else: # cell which contains goal, up / below half of cell plane gap - 
            cur_sub = self.subspace_world.subspaces.Pos2Sub(cur_pose[:3,3])
            sub_center = self.subspace_world.subspaces.Sub2Pos(cur_sub)
            sub_center[2] = cur_pose[2,3] # temporary, subspaces should be multi-plane in the future -> nearest among z_list?
            
            xyz_min = sub_center - self.subspace_world.subspaces.resolution * 0.5
            xyz_max = sub_center + self.subspace_world.subspaces.resolution * 0.5
            xyz_min[2] -= 0.5 # temporary
            xyz_max[2] += 0.5 # temporary
            xyz_min = np.maximum(xyz_min, self.bound[:,0])
            xyz_max = np.minimum(xyz_max, self.bound[:,1])
            sub_min = self.occupancy_map.Pos2Sub(xyz_min)
            sub_max = self.occupancy_map.Pos2Sub(xyz_max)
            x_indices = np.arange(sub_min[0], sub_max[0])
            y_indices = np.arange(sub_min[1], sub_max[1])
            z_indices = np.arange(sub_min[2], sub_max[2])
            xx, yy, zz = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
            
            voxels_in_cur_cell = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
            uncert_in_cur_cell = uncert_sdf_vols[0][voxels_in_cur_cell[:,0],voxels_in_cur_cell[:,1],voxels_in_cur_cell[:,2]]
            visible_mask = self.check_visibility(cur_vxl, voxels_in_cur_cell, uncert_sdf_vols[1])
            uncert_in_cur_cell[~visible_mask] = 0
            sel = np.argpartition(uncert_in_cur_cell, -100, axis=None)[-self.planner_cfg.obs_per_goal:]
            lst_lookat_vxl = voxels_in_cur_cell[sel]
        return lst_lookat_vxl
    
    def compute_next_state_pose(self, 
                                cur_pose       : np.ndarray,
                                uncert_sdf_vols: List,
                                occupancy_map
                                ) -> np.ndarray:
        cur_position = cur_pose[:3,3]
        cur_rotation = cur_pose[:3,:3]
        if self.state == "planning":
            if self.cur_rot_planning_finished:
                # It is passed from rotateOnGoal, and we want the next goal!
                if len(self.global_path) > 0:
                    while len(self.global_path)>0: 
                        self.goal = self.global_path[0]
                        self.global_path.pop(0)
                        self.goal_region = self.global_region_order[0]
                        self.global_region_order.pop(0)
                        
                        if not self.check_goal_reached(cur_pose[:3,3], self.goal[0]):
                            break
                self.cur_rot_planning_finished = False
                
            planning_out = self.uncertainty_aware_planning_v2(uncert_sdf_vols, cur_pose)
            self.is_goal_reachable = planning_out['is_goal_reachable']
            self.path = planning_out['path']
            self.info_enough = planning_out['planning_valid']

            cur_vxl = self.occupancy_map.Pos2Sub(cur_pose[:3,3])
            all_frontiers = np.array([self.occupancy_map.Pos2Sub(pos) for pos in self.all_frontiers])
            if len(all_frontiers) == 0:
                self.frontier_enough = False
            else:
                visible_mask = self.check_visibility(cur_vxl, all_frontiers, uncert_sdf_vols[1])
                lst_lookat_vxl = all_frontiers[visible_mask]
                self.frontier_enough = len(lst_lookat_vxl) > 5;
            new_pose = cur_pose.copy()
        elif self.state == "planningToGlobalGoal":
            print("PlanningToGobalGoal, Update Region Graph")   
            self.subspace_world.update_region_graph(self.occupancy_map)
            self.subspace_world.subspace_graph.update_graph_connectivity(self.occupancy_map)

            self.execute_global_planner(cur_pose, uncert_sdf_vols)#, force_run=True) 
            
            
            planning_type = 'target_pose' # 'target_pose' / 'astar'
            
            if planning_type == 'astar':
                
                # Idea: when we are stuck in local area where information gain is really small, run 2D a-star planning toward goal 
                #       & make the camera follow the path
                
                #  we need a good goal. So we iterate in cell list from global path,
                next_global_region_id = -1

                cur_vxl = self.occupancy_map.Pos2Sub(cur_pose[:3,3])

                next_global_region_id, sub_path, good_rot = self.get_next_global_region_id(cur_position, uncert_sdf_vols)
                
                if next_global_region_id == -1:
                    next_pose = cur_pose.copy()
                    print("Error")
                    return next_pose

                shortened_path = sub_path.copy()
                
                # Extract to path
                cur_p = copy.deepcopy(cur_position)
                total_position_path = []
                for i in range(1, len(shortened_path)):
                    idx = shortened_path[i].astype(int)
                    idx[2] = cur_vxl[2]
                    p = occupancy_map.Sub2Pos(idx)
                    total_position_path.append(cur_p)
                    cur_p = p
                Rot = cur_pose[:3,:3]                
                
                total_position_path = np.array(total_position_path)
                _, indices = np.unique(total_position_path, axis=0, return_index=True)
                total_position_path = total_position_path[np.sort(indices)]
                
                total_position_path = total_position_path.tolist()
                                
                path_len = len(total_position_path)
                total_rotation_path = interpolate_rotation2(Mat2Rotation(Rot), Mat2Rotation(good_rot), num_steps=path_len-1)
                total_rotation_path = [Rotation2Mat(rot) for rot in total_rotation_path]
                self.path = [np.vstack((np.hstack((total_rotation_path[i],np.array(total_position_path[i]).reshape((3,1)))), 
                                        np.array([[0.,0.,0.,1.]]))) for i in range(1, path_len)]
                
                print("TO Global Goal:  path with len ", len(self.path))

            elif planning_type == 'target_pose':
                '''
                From region-astar path, extract the target position for each region path
                And connect the each target position toward the global goal 
                And find the good target pose path connecting each region toward the goal
                '''
                rg = self.subspace_world.region_graph
                next_global_region_id, region_id_path, _ = self.get_next_global_region_id(cur_position, uncert_sdf_vols, find_voxel_path=False)
                if next_global_region_id == -1:
                    cur_region_id = self.subspace_world.subspaces.Pos2Ind(cur_position)
                    
                    global_region_order = [self.goal_region]
                    global_region_order.extend(self.global_region_order)
                    for next_global_region_id in global_region_order:
                    
                        if next_global_region_id == -1:
                            continue
                        _, region_id_path = rg.find_region_graph_astar_path(cur_region_id, next_global_region_id)
                        if len(region_id_path) > 0:
                            break
                        
                tmp_position = cur_position
                tmp_pose = copy.deepcopy(cur_pose)
                                
                if len(region_id_path) > 0:
                    self.path = []
                    indices = [i for i in range(len(region_id_path)) if rg.vertices[region_id_path[i]].uncert_score > 0] # len(indices) >= 2
                    if len(indices) > 0:
                        target_rotations = interpolate_rotation2(Mat2Rotation(cur_pose[:3, :3]), Mat2Rotation(rg.vertices[region_id_path[indices[0]]].target_pose[:3,:3]), num_steps=indices[0]+1)[1:]
                        for i in range(1, len(indices)):
                            before_rot = Mat2Rotation(rg.vertices[region_id_path[indices[i-1]]].target_pose[:3,:3])
                            after_rot = Mat2Rotation(rg.vertices[region_id_path[indices[i]]].target_pose[:3,:3])
                            num_steps = indices[i] - indices[i-1]
                            interpolated_rotations = interpolate_rotation2(before_rot, after_rot, num_steps=num_steps)[1:]
                            target_rotations.extend(interpolated_rotations)
                        n = len(region_id_path)-1-indices[-1]
                        if n > 0:
                            interpolated_rotations = interpolate_rotation2(Mat2Rotation(rg.vertices[region_id_path[indices[-1]]].target_pose[:3,:3]), Mat2Rotation(rg.vertices[region_id_path[-1]].target_pose[:3,:3]), num_steps=n)[1:]
                            target_rotations.extend(interpolated_rotations)
                        target_rotations = [compute_camera_pose(np.zeros(3), -Rotation2Mat(rot)[:3,2], up_dir=self.planner_cfg.up_dir) for rot in target_rotations]
                    else:
                        target_rotations = [cur_pose[:3,:3]] * len(region_id_path)
                        
                    # save for visualizing - 6
                    cell_bounds = [rg.vertices[id].cell_bound for id in region_id_path]
                    target_poses = []
                    target_uncs = []
                    topk_uncert_vxls = []
                    for i, region_id in enumerate(region_id_path):
                        target_pos = rg.vertices[region_id].target_pose[:3, 3]
                        target_rot = target_rotations[i]
                        target_poses.append(np.vstack((np.hstack((target_rot,target_pos.reshape((3,1)))), np.array([[0.,0.,0.,1.]])))[None])
                        target_uncs.append(rg.vertices[region_id].uncert_score)
                        topk_uncert_vxls.append(rg.vertices[region_id].valid_vxl)
                    
                for i, region_id in enumerate(region_id_path):
                    target_pos = rg.vertices[region_id].target_pose[:3, 3]
                    target_rot = target_rotations[i]  
                    target_pose = np.vstack((np.hstack((target_rot,target_pos.reshape((3,1)))), np.array([[0.,0.,0.,1.]])))
                    # target_pose = np.vstack((np.hstack((cur_rotation,target_pos.reshape((3,1)))), np.array([[0.,0.,0.,1.]])))
                    if self.occupancy_map.is_line_feasible(tmp_position.tolist(), target_pos.tolist()):
                        tmp_line_segment = set_path_between_pose_mat(self.robot, tmp_pose, target_pose)
                        tmp_position = copy.deepcopy(tmp_line_segment[-1][:3,3])
                        tmp_pose = copy.deepcopy(tmp_line_segment[-1])
                        self.path.extend(tmp_line_segment)
                    else:
                        _, uncert_aggre_outputs = self.uncertainty_aggregation_in_frustum(uncert_sdf_vols, cell_ids=[region_id], region_graph=rg, half_fov=self.fov/2, include_uncert_vxl_only=True)
                        valid_target_found = False
                        try:
                            new_target_poses = self.get_target_poses_v2([region_id], uncert_aggre_outputs, self.gs_x, self.gs_y, self.gs_z) # 
                        except:
                            new_target_poses = []
                        for new_target_pose in new_target_poses:
                            new_target_pose = new_target_pose[0]
                            new_target_pos = new_target_pose[:3,3]
                            new_target_pose[:3,:3] = target_rot # for smooth rotation & avoid to much rotation
                            if self.occupancy_map.is_line_feasible(tmp_position.tolist(), new_target_pos.tolist()):
                                valid_target_found = True
                                tmp_line_segment = set_path_between_pose_mat(self.robot, tmp_pose, new_target_pose)
                                tmp_position = copy.deepcopy(tmp_line_segment[-1][:3, 3])
                                tmp_pose = copy.deepcopy(tmp_line_segment[-1])
                                self.path.extend(tmp_line_segment)
                                break
                        if not valid_target_found:
                            # If no valid target, break and run a-star toward the goal
                            next_global_region_id, sub_path, good_rot = self.get_next_global_region_id(tmp_position, uncert_sdf_vols, n_visible_threshold=1) 
                            
                            cur_vxl = self.occupancy_map.Pos2Sub(tmp_position)
                            cur_p = copy.deepcopy(tmp_position)
                            total_position_path = []
                            for i in range(1, len(sub_path)):
                                idx = sub_path[i].astype(int)
                                idx[2] = cur_vxl[2]
                                p = occupancy_map.Sub2Pos(idx)
                                total_position_path.append(cur_p)
                                cur_p = p
                            Rot = tmp_pose[:3,:3]
                            
                            
                            total_position_path = np.array(total_position_path)
                            _, indices = np.unique(total_position_path, axis=0, return_index=True)
                            total_position_path = total_position_path[np.sort(indices)]
                            
                            total_position_path = total_position_path.tolist()
                            
                            
                            path_len = len(total_position_path)
                            if good_rot is None:
                                good_rot = copy.deepcopy(Rot)
                            total_rotation_path = interpolate_rotation2(Mat2Rotation(Rot), Mat2Rotation(good_rot), num_steps=path_len-1)
                            total_rotation_path = [Rotation2Mat(rot) for rot in total_rotation_path]
                            new_path = [np.vstack((np.hstack((total_rotation_path[i],np.array(total_position_path[i]).reshape((3,1)))), 
                                                np.array([[0.,0.,0.,1.]]))) for i in range(1, path_len)]

                            self.path.extend(new_path)
                            break
                            
                print("Planning On Global Goal FOund ")
            ################################################################ 
            else:
                raise NotImplementedError
            
            new_pose = cur_pose.copy() #self.path[0] 
            
            try:
                self.goal = (self.path[-1][:3,3].reshape((-1,)), True)
            except:
                self.goal = (new_pose[:3,3].reshape((-1,)), True)
            # return new_pose 
        
        elif self.state == "rotateOnGoal":
            def sample_weighted_points(points, ref_point, num_samples=10):
                flat_points = points.reshape(-1, 3)
                # Compute Euclidean distances
                distances = np.linalg.norm(flat_points - ref_point, axis=1)
                # Avoid division by zero (set very small value for zero distances)
                distances[distances == 0] = 1e-6  
                # Compute inverse distance weights
                weights = 1 / np.sqrt(distances)
                # Normalize weights to sum to 1 (convert to probabilities)
                probabilities = weights / np.sum(weights)
                # Sample indices based on computed probabilities
                sampled_indices = np.random.choice(flat_points.shape[0], size=num_samples, replace=False, p=probabilities)
                # Get the sampled points
                sampled_points = flat_points[sampled_indices]
                return sampled_points

            # If rotate On Goal is first called when reached the goal, 
            if len(self.rots) == 0: 
                # Update global representation
                self.update_global_representation(cur_pose[:3,3])
                self.cur_rot_planning_finished = False
                '''
                compute rotation directions  option frontier / topk uncertain voxels
                '''
                
                rotation_type = "as_local_target_pose" # "skip"
                if rotation_type == "as_local_target_pose":
                    cur_vxl = self.occupancy_map.Pos2Sub(cur_pose[:3,3])
                    region_graph = self.subspace_world.region_graph
                    region_id = region_graph.get_region_id_from_position(cur_pose[:3,3])
                    cell_ids = [region_id]
                    cell_ids.extend(region_graph.dct_neighbor_idx[region_id])
                    lst_lookat_loc = self.lookats_at_fixed_vxl(cur_vxl, uncert_sdf_vols, cell_ids=cell_ids, region_graph=region_graph, half_fov=self.fov/2) 
                elif rotation_type == "skip":
                    lst_lookat_loc = []
                else:
                    cur_vxl = self.occupancy_map.Pos2Sub(cur_pose[:3,3])
                    if self.planner_cfg.get("use_frontier_cluster_for_heuristic", True):
                        # use all frontiers
                        all_frontiers = np.array([self.occupancy_map.Pos2Sub(pos) for pos in self.all_frontiers])
                        visible_mask = self.check_visibility(cur_vxl, all_frontiers, uncert_sdf_vols[1])
                        lst_lookat_vxl = all_frontiers[visible_mask]

                    else: # 
                        # use uncertain voxels in cell current voxel exists
                        lst_lookat_vxl = self.compute_uncertain_voxels(cur_pose, uncert_sdf_vols)
                    if len(lst_lookat_vxl) > 0:
                        lst_lookat_vxl = sample_weighted_points(lst_lookat_vxl, cur_vxl, min(lst_lookat_vxl.shape[0],self.planner_cfg.obs_per_goal))
                    else:
                        lst_lookat_vxl = []
                    lst_lookat_loc = [self.occupancy_map.Sub2Pos(vxl) for vxl in lst_lookat_vxl]

                for lookat_loc in lst_lookat_loc:
                    rot = compute_camera_pose(cur_pose[:3, 3], lookat_loc, up_dir=self.planner_cfg.up_dir)
                    if math.isnan(rot.max()):
                        continue
                    self.rots.append(rot)                      
                
                if len(self.rots) > 0:
                    self.rots = rotation_planning(cur_pose[:3, :3], self.rots, self.planner_cfg.max_rot_deg)
                else:
                    self.cur_rot_planning_finished = True
                    return cur_pose
            
            new_pose = self.rotating_at_current_loc(cur_pose, self.rots)
            
            print("cur rot sequence left:", len(self.rots))
            if len(self.rots) == 0:
                self.cur_rot_planning_finished = True
            
            # return new_pose 
        
        elif self.state == "movingToGoal":  
            if len(self.path) == 0: # ex: plantoglobalgoal does not give path / no target poses for local tsp
                new_pose = cur_pose.copy()
                return new_pose
            next_pose = self.path[0]            
            new_pose = self.moving_to_goal(cur_pose, next_pose)
            self.path.pop(0)
            
        elif self.state == "staying":
            new_pose = cur_pose.copy()
        return new_pose

    def check_visibility(self, cur_sub, check_subs, sdf_vols):
        t_values = np.linspace(0, 1, 30)
        view_vec = cur_sub - check_subs # (N, 3)
        vis_test_points = cur_sub[None, :, None] - t_values * view_vec[..., None]
        vis_test_points = np.transpose(vis_test_points, (0, 2, 1)).astype(np.int64)
        vis_test_sdf = sdf_vols[vis_test_points[...,0], vis_test_points[...,1], vis_test_points[...,2]]
        vis_test_sdf = np.min(vis_test_sdf, axis=1)
        visible_mask = vis_test_sdf > 0
        return visible_mask
        
    def update_viewpoints(self, position, uncert_sdf_vols) -> int:
        self.viewpoint_manager.update_viewpoint_collision(self.occupancy_map)
        self.update_visited_positions(position)
        self.viewpoint_manager.update_viewpoints_visited(self.visited_positions)

        return

    def update_visited_positions(self, robot_position):
        existing = False
        for i in range(len(self.visited_positions)):
            if np.linalg.norm(robot_position - self.visited_positions[i]) < 1.0:
                existing = True
                break
        if not existing:
            self.visited_positions.append(robot_position)

    def extract_voxel_path(self, region_graph,tmp_key,next_key):
        _, vertex_global_path = region_graph.find_region_graph_astar_path(tmp_key, next_key)
        voxel_global_path = []
        for idx in range(len(vertex_global_path)-1):
            tmp_key, next_key = vertex_global_path[idx], vertex_global_path[idx + 1]
            idx = region_graph.dct_neighbor_idx[tmp_key].index(next_key)
            voxel_path = region_graph.dct_neighbor_path[tmp_key][idx]
            voxel_global_path.extend(voxel_path)
        return np.array(voxel_global_path)
    
    def execute_global_planner(self, cur_pose, uncert_sdf_vols, force_run=False):
        cur_position = cur_pose[:3,3] 
        self.update_viewpoints(cur_position,
                                uncert_sdf_vols)

        
        # 1. Global Planning
        if len(self.global_path) == 0 or self.gp_iter >= 5 or force_run:
            
            self.subspace_world.update_region_graph(self.occupancy_map) 
            good_goal_found = False
            global_path, only_coord_global_path, global_region_order =  self.run_global_planning(uncert_sdf_vols, cur_position)
            
            self.viz_global_path = copy.deepcopy(only_coord_global_path)
            self.global_path = global_path
            self.global_region_order = global_region_order
            # Save voxel-wise path for global planning            
            n_global_region = len(global_region_order)
            voxel_global_path = []
            if n_global_region > 1:
                for i in range(n_global_region-1):
                    voxel_global_path.extend(self.extract_voxel_path(self.subspace_world.region_graph, self.global_region_order[i], self.global_region_order[i+1]))
                if len(voxel_global_path) > 0:
                    self.voxel_global_path = np.stack(voxel_global_path) # if you use .append instead of .extend, then use np.concatenate
            
            
            if len(global_path) > 1:
                # Find current position
                while len(self.global_path)>0:
                    self.goal = self.global_path[0]
                    self.global_path.pop(0)
                    self.goal_region = self.global_region_order[0] 
                    self.global_region_order.pop(0)
                    
                    if not self.check_goal_reached(cur_pose[:3,3], self.goal[0]):
                        good_goal_found = True
                        break
                self.gp_planner_goal = True
            
            if not good_goal_found:
                print("NOT GOOD GOAL FOUND !")
                trial_num = 0
                while True:
                    goal, uncert = self.goal_search(uncert_sdf_vols[0], max_uncert=False, trial_num=trial_num)
                    trial_num += 1
                    if not self.check_goal_reached(cur_pose[:3,3], goal):
                        break
                self.goal = (goal, True)
        print("[Planner] global path: ", self.global_path)
        self.gp_iter += 1
    
    def uncertainty_aware_planning(self,
                                uncert_sdf_vols,
                                cur_pose   : np.ndarray,
                                reset_goal : bool = True,
                                goal       : np.ndarray = None):
        

        cur_position = cur_pose[:3,3]        
        cur_rotation = cur_pose[:3,:3]

        self.update_global_representation(cur_position)
        if self.run_global_planner:
            # update grid world
            self.execute_global_planner(cur_pose, uncert_sdf_vols)
        else:
            # Select goal position from the maximum information gain
            goal, _ = self.goal_search(uncert_sdf_vols[0])
            self.goal = (goal, True)

        region_graph = self.subspace_world.region_graph
        cur_index = self.occupancy_map.Pos2Sub(cur_pose[:3, 3])
        
        selected_ids = []    
        for key in region_graph.vertices.keys():
            region = region_graph.vertices[key]
            cur_cell = np.logical_and(np.all(region.lb_index <= cur_index), np.all(region.tr_index > cur_index))
            if cur_cell:
                selected_ids.append(key)
                selected_ids.extend(region_graph.dct_neighbor_idx[key])
        
        # Extract Target Poses from Neighboring Regions
        _, uncert_aggre_outputs = self.uncertainty_aggregation_in_frustum(uncert_sdf_vols, cell_ids=selected_ids, region_graph=region_graph, half_fov=self.fov/2)
        extracted_target_poses = []
        if len(uncert_aggre_outputs.keys()) > 0:
            t1 = time.time()
            target_poses = self.get_target_poses_v2(selected_ids, uncert_aggre_outputs, self.gs_x, self.gs_y, self.gs_z) 
            print(f"comp time for extracting target poses: {(time.time() - t1)*1000.0}[ms]")
            if len(target_poses) > 0:
                # extract best (position/rotation) pair from each position, sorted w.r.t score
                num_free_target_pose_from_cur_pose = 0 # If any target pose is not connected to the current pose, we want to call global planning
                for i in range(len(target_poses)):
                    cur_target_pose = target_poses[i][0]
                    lin_vel = (cur_target_pose[:3,3] - cur_pose[:3,3])
                    _, is_free = is_collision_free(cur_pose[:3,3], lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
                    if is_free:
                        num_free_target_pose_from_cur_pose += 1
                        
                extracted_target_poses = target_poses

        if len(extracted_target_poses) == 0 or num_free_target_pose_from_cur_pose == 0:
            is_valid_plannning = False
            path = [cur_pose]
        else:
            # Find a global goal projected to the local planning domain, which will works as a goal node for local TSP
            rg = self.subspace_world.region_graph    
            region_id = rg.get_region_id_from_position(cur_position)
            local_plan_id_list = copy.deepcopy(rg.dct_neighbor_idx[region_id])
            local_plan_id_list.append(region_id)
            
            # Check if there is goal from global TSP, which is outside the local planning domain
            outside_goal_exists = False
            outside_goal = None
            
            add_goal_node = False
            goal_pose = None
            
            global_path = [self.goal]
            global_region_order = [self.goal_region]
            global_path.extend(self.global_path)
            global_region_order.extend(self.global_region_order)
            
            for i in range(len(global_region_order)):
                if len([True for l in local_plan_id_list if l == global_region_order[i]]) == 0:
                    outside_goal_exists = True
                    outside_goal = global_path[i][0]
                    break
            
            outside_goal_exists = True   
            if outside_goal_exists:
                local_goal_type = 'target_pose' #'proj_pt'
                if local_goal_type == 'target_pose': 
                    dist_to_goal = []
                    n_valid_path = 0
                    lst_region_path = []
                    for id in local_plan_id_list:
                        region_path = region_graph.find_region_graph_astar_path(id, self.goal_region)[1]
                        
                        region_path_len = len(region_path)
                        
                        if region_path_len > 0:
                            lst_region_path.append(np.stack([rg.vertices[r_id].center for r_id in region_path]))
                        
                        if region_path_len == 0 and region_id != id:
                            region_path_len = 10000
                        else:
                            n_valid_path += 1
                        dist_to_goal.append(region_path_len)    

                    if n_valid_path > 0:
                        sorted_local_plan_id_list = np.array(local_plan_id_list)[np.argsort(dist_to_goal)]
                        for i, id in enumerate(sorted_local_plan_id_list):
                            in_region = [rg.get_region_id_from_position(pose[0, :3, 3]) == id for pose in extracted_target_poses]
                            if sum(in_region) > 0:
                                add_goal_node = True
                                idx_sel = np.argmax(in_region)
                                goal_pose = extracted_target_poses[idx_sel][0]
                                
                                extracted_target_poses.pop(idx_sel)
                                break
                        
                elif local_goal_type == 'proj_pt':
                    # Project the outside goal into the local planning region
                    dist_to_goal = [np.inf for _ in range(len(local_plan_id_list))]
                    proj_pt_list = [np.zeros(3) for _ in range(len(local_plan_id_list))]
                    for i, id in enumerate(local_plan_id_list):
                        cell_bound = rg.vertices[id].cell_bound
                        min_x, max_x = cell_bound[0][0], cell_bound[1][0]
                        min_y, max_y = cell_bound[0][1], cell_bound[1][1]
                        min_z, max_z = cell_bound[0][2], cell_bound[1][2]
                        proj_pt = project_point_to_cuboid(outside_goal[0], outside_goal[1], outside_goal[2], min_x, max_x, min_y, max_y, min_z, max_z)
                        dist_to_goal[i] = np.linalg.norm(outside_goal - proj_pt)
                        proj_pt_list[i] = proj_pt
                        # We want a collision-free goal node
                        if not self.occupancy_map.is_feasible_point(proj_pt[0],proj_pt[1],proj_pt[2]):
                            # Simple heursitic, to change z value, to find a collision-free goal node
                            for _ in range(100):
                                new_z = random.uniform(0,1) * (max_z-min_z) + min_z
                                if self.occupancy_map.is_feasible_point(proj_pt[0],proj_pt[1],new_z):
                                    proj_pt[2] = new_z
                                    dist_to_goal[i] = np.linalg.norm(outside_goal - proj_pt)
                                    proj_pt_list[i] = proj_pt
                                    break
                    # Find a best projected point comparing based on the distance to true global goal
                    min_index = dist_to_goal.index(min(dist_to_goal))
                    goal_position = proj_pt_list[min_index]
                            
                    goal_rot = cur_rotation 
                    goal_pose = np.vstack((np.hstack((goal_rot,goal_position.reshape((3,1)))), np.array([[0.,0.,0.,1.]])))                
                # Run a local TSP to find path with goal node
                path, is_valid_plannning = self.run_local_planning_v2(cur_pose,
                                                                goal_pose,
                                                                self.occupancy_map, 
                                                                uncert_sdf_vols[1],
                                                                extracted_target_poses,
                                                                add_goal_node=add_goal_node)  
            else:
                # If outside goal does not exist, we don't want to run a global planning loop here. 
                path, is_valid_plannning = self.run_local_planning_v2(cur_pose, 
                                                                None,
                                                                self.occupancy_map, 
                                                                uncert_sdf_vols[1],
                                                                extracted_target_poses)    
    
        planning_out = dict(
            path         = path,
            goal = self.goal[0],
            is_goal_reachable=True,
            planning_valid = is_valid_plannning
        )
        
        return planning_out

    def uncertainty_aware_planning_v2(self,
                                uncert_sdf_vols,
                                cur_pose   : np.ndarray,
                                goal       : np.ndarray = None):
        

        cur_position = cur_pose[:3,3]        
        cur_rotation = cur_pose[:3,:3]

        self.update_global_representation(cur_position)
        if self.run_global_planner:
            self.execute_global_planner(cur_pose, uncert_sdf_vols)
        else:
            goal, _ = self.goal_search(uncert_sdf_vols[0])
            self.goal = (goal, True)

        region_graph = self.subspace_world.region_graph
        
        region_id = region_graph.get_region_id_from_position(cur_pose[:3,3])
        selected_ids = [region_id]
        selected_ids.extend(region_graph.dct_neighbor_idx[region_id])
        cur_vxl = self.occupancy_map.Pos2Sub(cur_pose[:3,3])
        t0 = time.time()
        if self.greedy:
            _, uncert_aggre_outputs = self.uncertainty_aggregation_in_frustum(uncert_sdf_vols, cell_ids=selected_ids, region_graph=region_graph, half_fov=self.fov/2, only_visible_vxls=False, only_traversable_vxls=False, exclude_init_uncert=True, cur_vxl=cur_vxl) 
        else: 
            _, uncert_aggre_outputs = self.uncertainty_aggregation_ablate(uncert_sdf_vols, cell_ids=selected_ids, region_graph=region_graph, only_visible_vxls=False, only_traversable_vxls=False, cur_vxl=cur_vxl)
        print(f"uncertainty_aggregation_in_frustum takes {(time.time()-t0)*1000} ms")
        target_poses = []
        num_free_target_pose_from_cur_pose = 0
        # outside_goal_exists = not self.goal_region in selected_ids
        outside_goal_exists = not region_graph.get_region_id_from_position(self.goal[0]) in selected_ids
        if outside_goal_exists:
            if len(uncert_aggre_outputs.keys()) > 0:
                topk_uncert_vxls_all = uncert_aggre_outputs['topk_uncert_vxl'].cpu().numpy()
                if self.greedy:
                    target_poses = self.get_target_poses_v2(selected_ids, uncert_aggre_outputs, self.gs_x, self.gs_y, self.gs_z, n_visible_threshold=self.n_visible_threshold)
                else: 
                    target_poses = self.get_target_poses_ablate(selected_ids, uncert_aggre_outputs, self.gs_x, self.gs_y, self.gs_z)
                if len(target_poses) > 0:
                    
                    cell_bounds = [region_graph.vertices[id].cell_bound for id in selected_ids]
                    lst_neighbor_path = region_graph.dct_neighbor_path[region_id]

                    for i in range(len(target_poses)):
                        cur_target_pose = target_poses[i][0]
                        lin_vel = (cur_target_pose[:3,3] - cur_pose[:3,3])
                        _, is_free = is_collision_free(cur_pose[:3,3], lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
                        if is_free:
                            num_free_target_pose_from_cur_pose += 1
            
            if num_free_target_pose_from_cur_pose == 0:
                is_valid_plannning = False
                path = [cur_pose]
            else:
                distance_type = 'euclidean'
                if distance_type == 'euclidean':
                    dist_to_goal = [np.linalg.norm(self.goal[0] - region_graph.vertices[id].center) for id in selected_ids]
                    n_valid_path = len(selected_ids) # temporary - do not consider the straight line is feasible or not
                else: # 'geodesic'
                    dist_to_goal = []
                    n_valid_path = 0
                    lst_region_path = []
                    for id in selected_ids:
                        region_path = region_graph.find_region_graph_astar_path(id, self.goal_region)[1]
                        
                        region_path_len = len(region_path)
                        
                        if region_path_len > 0:
                            lst_region_path.append(np.stack([region_graph.vertices[r_id].center for r_id in region_path]))
                        
                        if region_path_len == 0 and region_id != id:
                            region_path_len = 10000
                        else:
                            n_valid_path += 1
                        dist_to_goal.append(region_path_len)    
                
                add_goal_node = False
                goal_pose = None    
                if n_valid_path > 0:
                    sorted_id_list = np.array(selected_ids)[np.argsort(dist_to_goal)]
                    for i, id in enumerate(sorted_id_list):
                        in_region = [region_graph.get_region_id_from_position(pose[0, :3, 3]) == id for pose in target_poses]
                        if sum(in_region) > 0:
                            add_goal_node = True
                            idx_sel = np.argmax(in_region)
                            goal_pose = target_poses[idx_sel][0]
                            
                            target_poses.pop(idx_sel)
                            break
                    
                path, is_valid_plannning = self.run_local_planning_v2(cur_pose,
                                                                    goal_pose,
                                                                    self.occupancy_map, 
                                                                    uncert_sdf_vols[1],
                                                                    target_poses,
                                                                    add_goal_node=add_goal_node)          
        else:
            add_goal_node = True
            goal_pose = region_graph.vertices[self.goal_region].target_pose
            
            if len(uncert_aggre_outputs.keys()) > 0:
                if self.greedy:
                    # To avoid information overlap with goal pose
                    gs_uncert_collections = uncert_aggre_outputs['gs_uncert_collections']#.clone()
                    n_voxels, n_rotation_samples = gs_uncert_collections.shape[-2:]
                    gs_aggre_uncerts = uncert_aggre_outputs['gs_aggre_uncerts']
                    gs_uncert_collections = gs_uncert_collections.reshape(list(gs_aggre_uncerts.shape[:-1]) + [n_voxels, n_rotation_samples])
                    gs_goal_sub = self.occupancy_map.Pos2Sub(goal_pose[:3,3]) // 2
                    gs_goal_sub[2] = np.round((goal_pose[2,3]-self.occupancy_map.origin[2])-1).astype(np.int64)
                    seen_voxel_mask = (gs_uncert_collections[gs_goal_sub[0], gs_goal_sub[1], gs_goal_sub[2]]>0).sum(dim=-1)
                    gs_uncert_collections = gs_uncert_collections.reshape(-1, n_voxels, n_rotation_samples)
                    gs_uncert_collections[:, seen_voxel_mask] = 0.
                    uncert_aggre_outputs['gs_uncert_collections'] = gs_uncert_collections
                    
                    target_poses = self.get_target_poses_v2(selected_ids, uncert_aggre_outputs, self.gs_x, self.gs_y, self.gs_z, num_target_poses=9, n_visible_threshold=self.n_visible_threshold)
                else: 
                    target_poses = self.get_target_poses_ablate(selected_ids, uncert_aggre_outputs, self.gs_x, self.gs_y, self.gs_z)
                    
                for i in range(len(target_poses)):
                    cur_target_pose = target_poses[i][0]
                    lin_vel = (cur_target_pose[:3,3] - cur_pose[:3,3])
                    _, is_free = is_collision_free(cur_pose[:3,3], lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
                    if is_free:
                        num_free_target_pose_from_cur_pose += 1
                lin_vel = (goal_pose[:3,3] - cur_pose[:3,3])
                _, is_free = is_collision_free(cur_pose[:3,3], lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
                if is_free:
                    num_free_target_pose_from_cur_pose += 1
            
            if num_free_target_pose_from_cur_pose == 0:
                is_valid_plannning = False
                path = [cur_pose]
            else:    
                path, is_valid_plannning = self.run_local_planning_v2(cur_pose,
                                                                    goal_pose,
                                                                    self.occupancy_map, 
                                                                    uncert_sdf_vols[1],
                                                                    target_poses,
                                                                    add_goal_node=add_goal_node) 

        planning_out = dict(
            path         = path,
            goal = self.goal[0],
            is_goal_reachable=True,
            planning_valid = is_valid_plannning
            # initial_path = initial_path
        )
        
        return planning_out


    def init_data(self, bbox: List) -> None:
        """initialize data for here planner
    
        Args:
            bbox (List, [3,2]): bounding box corners coordinates
    
        Attributes:
            gs_z_levels (List, [N])                  : Goal Space Z-levels. if not provided, unitformly samples from Z range.
            voxel_size (float)                       : voxel size
            bbox (np.ndarray, [3,2])                 : bounding box corners coordinates
            Nx/Ny/Nz (int)                           : bounding box sizes
            gs_x/y/z_range (torch.Tensor, [X/Y/Z])   : goal space X/Y/Z levels
            goal_space_pts (torch.Tensor, [X*Y*Z, 3]): goal space candidate locations. Unit: voxel
        """
        self.path = []
        self.lookat_tgts = None

        ### load config data ###
        self.gs_z_levels = self.planner_cfg.get("gs_z_levels", [5, 11, 17])
        self.voxel_size = self.main_cfg.planner.voxel_size # 0.1

        ### bounding box ###
        self.bbox = np.asarray(bbox)

        ## bounding box size (Unit: voxel) ##
        Nx = round((bbox[0][1] - bbox[0][0]) / self.voxel_size + 0.0005) + 1
        Ny = round((bbox[1][1] - bbox[1][0]) / self.voxel_size + 0.0005) + 1
        Nz = math.floor((bbox[2][1] - bbox[2][0]) / self.voxel_size + 0.0005) + 1
        Nz = self.planner_cfg.get("Nz", Nz)
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        
        
        planning_bound = self.planner_cfg.get("planning_bound", None)
        self.planning_bound = planning_bound
        if planning_bound is not None:
            self.Nx_start = round((planning_bound[0][0] - bbox[0][0]) / self.voxel_size + 0.0005)
            self.Ny_start = round((planning_bound[1][0] - bbox[1][0]) / self.voxel_size + 0.0005)
            self.Nx_end = round((planning_bound[0][1] - bbox[0][0]) / self.voxel_size + 0.0005)
            self.Ny_end = round((planning_bound[1][1] - bbox[1][0]) / self.voxel_size + 0.0005)
        else:
            self.Nx_start = 0
            self.Ny_start = 0
            self.Nx_end = self.Nx - 1
            self.Ny_end = self.Ny - 1

        ### Goal Space ###
        self.gs_x_range = torch.arange(0, self.Nx, 2)
        self.gs_y_range = torch.arange(0, self.Ny, 2) 
        if self.gs_z_levels is None:
            ### generate z-levels from 0.5 meter to bounding box's maximum Z-level ###
            self.gs_z_range = torch.arange( 
                        int(1/self.voxel_size), 
                        self.Nz, 
                        int(1/self.voxel_size))  
        else:
            self.gs_z_range = torch.tensor(self.gs_z_levels)
        self.gs_x, self.gs_y, self.gs_z = torch.meshgrid(self.gs_x_range, self.gs_y_range, self.gs_z_range, indexing="ij")
        self.goal_space_pts = torch.cat([self.gs_x.reshape(-1, 1), 
                                         self.gs_y.reshape(-1, 1), 
                                         self.gs_z.reshape(-1, 1)], dim=1).cuda().float()
        
        z_start = bbox[2][0] + 2 * self.voxel_size
        z_end = bbox[2][1] - 2 * self.voxel_size
        n_slices = int((z_end - z_start) // 0.5)
        self.frontier_z_slices = np.linspace(z_start, z_end, n_slices+1)


    def run_local_planning_v2(self, pose, goal_pose, occupancy_grid, sdf_vols, sorted_target_poses, add_goal_node=False):       
        self.local_tsp.set_occupancy_map(occupancy_grid)
        if add_goal_node:
            # Goal pose which is computed from global path, and project to the neighboring cell boundary
            pose_path = self.local_tsp.solve_local_tsp_with_global_goal(pose, sorted_target_poses, goal_pose)
        else:
            pose_path = self.local_tsp.solve_local_tsp(pose, sorted_target_poses)
        coord_path = []
        max_len = 100
        iter = 0
        for state in pose_path:
            if iter> max_len:
                break
            best_pos = state[0]
            best_R = state[1].as_matrix()
            best_pose = np.vstack((np.hstack((best_R, best_pos.reshape((3,1)))), np.array([[0.0,0.0,0.0,1.0]])))        
            coord_path.append(best_pose)
            iter += 1
        if len(coord_path) < 2:
            return coord_path, False
        return coord_path, True
        
    def run_global_planning(self, uncert_sdf_vols, robot_position):
        self.gp_iter = 0        
        self.subspace_world.update_cell_status(self.viewpoint_manager, self.occupancy_map)
        
        
        region_graph = self.subspace_world.region_graph
        if self.planning_bound is not None:
            region_graph.set_bool_planning_bound(self.planning_bound) 
        # subspaces = self.subspace_world.subspaces
        ids_exploring = [id for id in region_graph.vertices.keys() if region_graph.vertices[id].cell.get_status() == CellStatus.EXPLORING]
        if len(ids_exploring) > 0:
            is_valid, uncert_aggre_outputs = self.uncertainty_aggregation_in_frustum(uncert_sdf_vols, cell_ids=ids_exploring, region_graph=region_graph, half_fov=self.fov/2, only_visible_vxls=False, exclude_init_uncert=self.exclude_init_uncert)  #include_uncert_vxl_only=True)
            if is_valid:
                t0 = time.time()
                self.update_uncertainty_scores(ids_exploring, uncert_sdf_vols, uncert_aggre_outputs, self.gs_x.numpy(), self.gs_y.numpy(), self.gs_z.numpy(), up_dir=self.planner_cfg.up_dir)
                print(f"update_uncertainty_scores takes {(time.time()-t0)*1000} ms")        

        self.viewpoint_manager.update_candidate_viewpoint_cell_status(self.subspace_world)
        
        global_path, only_coord_global_path, global_cell_tsp_order = self.subspace_world.solve_global_tsp_rev(robot_position, self.occupancy_map)
        
        if len(global_cell_tsp_order) > 2:
            print()
        return global_path, only_coord_global_path, global_cell_tsp_order
            
    def update_global_representation(self, robot_position):
        '''
        Update grid world and keypose graph
        '''
        # update global representation
        self.subspace_world.update_neighbor_cells(robot_position)
        
        self.viewpoint_manager.update_robot_position(robot_position)
        print("[Viewpoint Manager] Update robot Position finished")
        
        self.subspace_world.update_robot_position(robot_position)
        print("Subspace world Update robot Position finished")
        
        frontier_position = self.occupancy_map.extract_frontier_voxel_vectorized(robot_position, self.frontier_z_slices)
        self.cluster_frontiers = self.occupancy_map.cluster_frontier_voxels()
        self.all_frontiers = self.occupancy_map.frontier_positions
        self.subspace_world.set_frontiers(frontier_position)

        
        if not self.subspace_world.home_set():
            self.subspace_world.set_home_position(self.initial_position)

    @torch.no_grad()
    def uncertainty_aggregation_ablate(self, 
                                   uncert_sdf_vols: List,
                                   force_running  : bool = False,
                                   cell_ids = None,
                                   region_graph = None,
                                   include_uncert_vxl_only=False,
                                   only_visible_vxls=False,
                                   only_traversable_vxls=False,
                                   cur_vxl=None
                                   ) -> Tuple[bool, Dict]: 
        """ Uncertainty Aggregation in Goal Space for Goal Search

        Args:
            uncert_sdf_vols (List)  : Uncertainty Volume and SDF Volume
                - uncert_vol (np.ndarray, [X,Y,Z]): uncertainty volume
                - sdf_vol (np.ndarray, [X,Y,Z])   : SDF volume
            force_running (bool)    : force running even goal space is invalid
        
        Returns:
            Tuple [bool, Dict]: goal_space_valid, outputs
                - outputs includes
                    - gs_aggre_uncert (torch.Tensor, [X,Y,Z]): goal space aggregated uncertainty.
                    - topk_uncert_vxl (torch.Tensor, [k, 3]): sampled top-k uncertain target observations. Unit: voxel
                    - gs_uncert_collections (torch.Tensor, [X*Y*Z, k]): individual uncertainties for each goal space candidate 
        """
        min_sensing_dist = self.planner_cfg.gs_sensing_range[0] / self.voxel_size
        max_sensing_dist = self.planner_cfg.gs_sensing_range[1] / self.voxel_size
        
        ##################################################
        ### Remove the predicted empty space from consideration
        ##################################################
        sdf = uncert_sdf_vols[1]
        uncert = np.copy(uncert_sdf_vols[0])
        ### only consider uncertainty on surface ###
        mask = (sdf >= 0) * (sdf < 0.5) 
        uncert[np.logical_not(mask)] = 0
        
        ##################################################
        ### remove goal points that are not safe to go (close to sdf surface)  (changed position)
        ##################################################
        unsafe_tgt = ((self.gs_x<1+self.Nx_start) + (self.gs_x>=self.Nx_end) + \
                      (self.gs_y<1+self.Ny_start) + (self.gs_y>=self.Ny_end) + \
                      (self.gs_z<1) + (self.gs_z+1>=self.Nz)).cuda()  # FIXME(HY): newly added
        binary_occupancy_map = self.occupancy_map.data.astype(np.bool_)
        binary_occupancy_map = torch.from_numpy(binary_occupancy_map).cuda()
        
        unsafe_tgt += binary_occupancy_map[self.gs_x,self.gs_y,self.gs_z] + \
                        binary_occupancy_map[(self.gs_x+1).clamp(self.Nx_start,self.Nx_end),self.gs_y,self.gs_z] + \
                        binary_occupancy_map[(self.gs_x-1).clamp(self.Nx_start,self.Nx_end),self.gs_y,self.gs_z]
        unsafe_tgt += binary_occupancy_map[self.gs_x,(self.gs_y+1).clamp(self.Ny_start,self.Ny_end),self.gs_z] + \
                        binary_occupancy_map[self.gs_x,(self.gs_y-1).clamp(self.Ny_start,self.Ny_end),self.gs_z]
        unsafe_tgt += binary_occupancy_map[self.gs_x,self.gs_y,(self.gs_z+1).clamp(0,self.Nz-1)] + \
                        binary_occupancy_map[self.gs_x,self.gs_y,(self.gs_z-1).clamp(0,self.Nz-1)]
        unsafe_tgt = unsafe_tgt.reshape(-1)
        
        sdf = torch.from_numpy(sdf).cuda()
        
        if cell_ids is not None:
            uncerts = []
            voxel_ids = []
            surface_masks = []
            d_x, d_y, d_z = uncert.shape
            voxel_ids_all = np.stack(np.meshgrid(np.arange(d_x), np.arange(d_y), np.arange(d_z), indexing="ij"), -1)
            if only_traversable_vxls:
                for cell_id in cell_ids:
                    region = region_graph.vertices[cell_id]
                    lb_index = region.lb_index
                    tr_index = region.tr_index
                    voxel_ids_cell = voxel_ids_all[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1, 3)
                    uncert_cell = uncert[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1)
                    
                    surface_mask = uncert_cell > 0
                    voxel_ids_surf = voxel_ids_cell[surface_mask]
                    # uncert_surf = uncert_cell[surface_mask]
                    
                    is_split, r1, r2 = region.check_and_split_region()
                    if is_split:
                        r1_center = r1.center_index
                        r2_center = r2.center_index
                        d1 = np.linalg.norm(cur_vxl - r1_center)
                        d2 = np.linalg.norm(cur_vxl - r2_center)
                        center_sel = r1_center if d1 <= d2 else r2_center
                        center_other = r2_center if d1 <= d2 else r1_center
                        dist_sel = np.linalg.norm(voxel_ids_surf - center_sel[None], axis=1)
                        dist_other = np.linalg.norm(voxel_ids_surf - center_other[None], axis=1)
                        traversable_mask = dist_sel < dist_other
                        
                        surface_mask = torch.from_numpy(surface_mask).cuda()
                        traversable_mask = torch.from_numpy(traversable_mask).cuda()
                        surface_mask = surface_mask.masked_scatter(surface_mask.clone(), traversable_mask).cpu().numpy()
                    
                    voxel_ids.append(voxel_ids_cell)
                    uncerts.append(uncert_cell)
                    surface_masks.append(surface_mask)                 
            else:
                for id in cell_ids:
                    region = region_graph.vertices[id]
                    lb_index = region.lb_index
                    tr_index = region.tr_index
                    
                    voxel_ids_cell = voxel_ids_all[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1, 3)
                    uncert_cell = uncert[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1)
                    
                    surface_mask = uncert_cell > 0
                    voxel_ids.append(voxel_ids_cell)
                    uncerts.append(uncert_cell)
                    surface_masks.append(surface_mask)
            
            surface_masks = np.concatenate(surface_masks)
            uncert_in_cell = np.concatenate(uncerts)
            voxel_ids = np.concatenate(voxel_ids)
            if surface_masks.sum() > self.planner_cfg.uncert_top_k:
                uncert_in_cell = uncert_in_cell[surface_masks]
                voxel_ids = voxel_ids[surface_masks]
            
            if only_visible_vxls:
                uncert_mask = uncert_in_cell > self.uncertain_threshold
                uncert_voxel_ids = voxel_ids[uncert_mask]
                n_total_vxl_in_cell = uncert_voxel_ids.shape[0]
                
                safe_gs_pts = self.goal_space_pts[~unsafe_tgt, None, :].repeat([1,n_total_vxl_in_cell,1])  # [safe, k, 3]
                view_vec = safe_gs_pts - torch.from_numpy(uncert_voxel_ids).float().cuda()
                dist = torch.norm(view_vec, dim=2)
                dist_mask = (dist < max_sensing_dist ) * (dist > min_sensing_dist)
                safe_mask = torch.ones_like(safe_gs_pts[:, :, 0]) == 1.
                safe_mask = safe_mask * dist_mask
                near_view_vec = view_vec[safe_mask]
                t_values = torch.linspace(0, 1, 30, device=near_view_vec.device)
                vis_test_points = safe_gs_pts[safe_mask][..., None] - t_values * near_view_vec[..., None]
                vis_test_points = vis_test_points.permute(0, 2, 1).long()
                vis_test_sdf = sdf[vis_test_points[:,:,0], vis_test_points[:,:,1], vis_test_points[:,:,2]]  # [x*y*z*near, 20]
                vis_test_sdf, _ = vis_test_sdf.min(dim=1) 
                visible_mask = vis_test_sdf > 0
                safe_mask = safe_mask.masked_scatter(safe_mask.clone(), visible_mask).sum(dim=0).bool()

                uncert_mask = torch.from_numpy(uncert_mask).cuda()
                uncert_mask = uncert_mask.masked_scatter(uncert_mask.clone(), ~safe_mask)
                uncert_in_cell[uncert_mask.cpu().numpy()] = 0.
                        

            n_vxl_in_cell = (uncert_in_cell > 0).sum()
            top_k = max(min(n_vxl_in_cell, self.planner_cfg.uncert_top_k), self.planner_cfg.uncert_top_k_subset)
            top_k_subset = self.planner_cfg.uncert_top_k_subset
            if include_uncert_vxl_only: 
                n_uncert_vxl = (uncert_in_cell > self.uncertain_threshold).sum()
                top_k = max(min(top_k, n_uncert_vxl), top_k_subset)
        else:
            top_k = self.planner_cfg.uncert_top_k
        
        if cell_ids is None:
            shuffled_indices = np.random.permutation(np.prod(uncert.shape))
            shuffled_uncert = uncert.reshape(-1)[shuffled_indices]
            topk_uncert_vxl = np.argpartition(shuffled_uncert, -top_k, axis=None)[-top_k_subset:]
            topk_uncert_vxl = shuffled_indices[topk_uncert_vxl]
            topk_uncert_vxl = np.unravel_index(topk_uncert_vxl, uncert.shape)
            topk_uncert_vxl = np.column_stack(topk_uncert_vxl)
        else:
            shuffled_indices = np.random.permutation(uncert_in_cell.shape[0])
            shuffled_uncert = uncert_in_cell[shuffled_indices]
            topk_uncert_vxl = np.argpartition(shuffled_uncert, -top_k, axis=None)[-top_k_subset:]
            topk_uncert_vxl = shuffled_indices[topk_uncert_vxl]
            topk_uncert_vxl = voxel_ids[topk_uncert_vxl]
        topk_uncert_vxl = torch.from_numpy(topk_uncert_vxl).cuda().float()

        ### move volumes to TorchTensor and cuda device ###
        uncert = torch.from_numpy(uncert).cuda()

        ##################################################
        ### only consider goal_space_pts within max sensing distance
        ##################################################
        goal_space_pts = self.goal_space_pts[:,None,:].repeat([1,top_k_subset,1])  # [x*y*z, k, 3]
        view_vec = goal_space_pts - topk_uncert_vxl  # [x*y*z, k, 3]
        dist = torch.norm(view_vec, dim=2)  # [x*y*z, k]
        dist_mask = (dist < max_sensing_dist ) * (dist > min_sensing_dist)  # we assume best view distance is between 0.5m and 2m
        if dist_mask.sum() == 0:
            self.info_printer(
                "   Warning! All high uncertainty points are far from Goal Space points",
                self.step,
                self.__class__.__name__ 
                )
        valid_mask = torch.ones_like(goal_space_pts[:, :, 0]) == 1.
        valid_mask = valid_mask * dist_mask
        
        valid_mask[unsafe_tgt,:] = False  # remove points that are not safe to go

        ##################################################
        ### visibility check
        ##################################################
        near_view_vec = view_vec[valid_mask]  # [x*y*z*near, 3]
        t_values = torch.linspace(0, 1, 30, device=near_view_vec.device)
        vis_test_points = goal_space_pts[valid_mask][..., None] - t_values * near_view_vec[..., None]  # [x*y*z*near, 3, 20]
        vis_test_points = vis_test_points.permute(0, 2, 1).long()  # [x*y*z*near, 20, 3]
        vis_test_sdf = sdf[vis_test_points[:,:,0], vis_test_points[:,:,1], vis_test_points[:,:,2]]  # [x*y*z*near, 20]
        ## Visibility check is valid only if sampled points are all with positive sdf ##
        vis_test_sdf, _ = vis_test_sdf.min(dim=1) 
        visible_mask = vis_test_sdf > 0
        if visible_mask.sum() == 0:
            self.info_printer(
                "   Warning! No visible uncertainty points.",
                self.step,
                self.__class__.__name__ 
                )
        
        ### warning message ###
        valid_mask = valid_mask.masked_scatter(valid_mask.clone(), visible_mask)
        
        if valid_mask.sum() == 0:
            self.info_printer(
                "   Warning! No valid uncertainty points. either too far or non-visible",
                self.step,
                self.__class__.__name__ 
                )
            invalid_goal_space = True
        else:
            invalid_goal_space = False

        ##################################################
        ### aggregate uncertainty
        ##################################################
        topk_uncert_vxl = topk_uncert_vxl.long()
        k_uncerts = uncert[topk_uncert_vxl[:,0], topk_uncert_vxl[:,1], topk_uncert_vxl[:,2]]  # [k]
        k_uncerts = k_uncerts[None, :].repeat(goal_space_pts.shape[0], 1)
        gs_uncert_collections = torch.zeros_like(k_uncerts)
        gs_uncert_collections[valid_mask] = k_uncerts[valid_mask]
        gs_aggre_uncerts = gs_uncert_collections.sum(dim=1)
        gs_aggre_uncerts = gs_aggre_uncerts.reshape([self.gs_x_range.shape[0], self.gs_y_range.shape[0], self.gs_z_range.shape[0]]) # [x, y, z, n_rotation_samples] 

        ### warning message ###
        if torch.count_nonzero(gs_aggre_uncerts) == 0:
            self.info_printer(
                "   Warning! No valid goal space. Going to filter search space. ",
                self.step,
                self.__class__.__name__ 
                )

        ##################################################
        ### gather outputs
        ##################################################
        outputs = {
            'gs_aggre_uncerts': gs_aggre_uncerts, 
            'topk_uncert_vxl': topk_uncert_vxl, 
            'gs_uncert_collections': gs_uncert_collections, 
        }

        if invalid_goal_space:
            if force_running:
                return True, outputs
            else:
                return False, {}
        else:
            return True, outputs
      
    @torch.no_grad()
    def uncertainty_aggregation_in_frustum(self, 
                                   uncert_sdf_vols: List,
                                   force_running  : bool = False,
                                   cell_ids = None,
                                   region_graph = None,
                                   half_fov=np.pi/6,
                                   include_uncert_vxl_only=False,
                                   only_visible_vxls=False,
                                   only_traversable_vxls=False,
                                   cur_vxl=None,
                                   exclude_init_uncert=False
                                   ) -> Tuple[bool, Dict]: 
        """ Uncertainty Aggregation in Goal Space for Goal Search

        Args:
            uncert_sdf_vols (List)  : Uncertainty Volume and SDF Volume
                - uncert_vol (np.ndarray, [X,Y,Z]): uncertainty volume
                - sdf_vol (np.ndarray, [X,Y,Z])   : SDF volume
            force_running (bool)    : force running even goal space is invalid
        
        Returns:
            Tuple [bool, Dict]: goal_space_valid, outputs
                - outputs includes
                    - gs_aggre_uncert (torch.Tensor, [X,Y,Z]): goal space aggregated uncertainty.
                    - topk_uncert_vxl (torch.Tensor, [k, 3]): sampled top-k uncertain target observations. Unit: voxel
                    - gs_uncert_collections (torch.Tensor, [X*Y*Z, k]): individual uncertainties for each goal space candidate 
        """
        min_sensing_dist = self.planner_cfg.gs_sensing_range[0] / self.voxel_size
        max_sensing_dist = self.planner_cfg.gs_sensing_range[1] / self.voxel_size
        
        ##################################################
        ### Remove the predicted empty space from consideration
        ##################################################
        sdf = uncert_sdf_vols[1]
        uncert = np.copy(uncert_sdf_vols[0])
        ### only consider uncertainty on surface ###
        mask = (sdf >= 0) * (sdf < 0.5) 
        uncert[np.logical_not(mask)] = 0
        if exclude_init_uncert:
            uncert[uncert > self.initial_uncert] = 0
        
        ##################################################
        ### remove goal points that are not safe to go (close to sdf surface)  (changed position)
        ##################################################
        unsafe_tgt = ((self.gs_x<1+self.Nx_start) + (self.gs_x>=self.Nx_end) + \
                      (self.gs_y<1+self.Ny_start) + (self.gs_y>=self.Ny_end) + \
                      (self.gs_z<1) + (self.gs_z+1>=self.Nz)).cuda()  # FIXME(HY): newly added
        binary_occupancy_map = self.occupancy_map.data.astype(np.bool_)
        binary_occupancy_map = torch.from_numpy(binary_occupancy_map).cuda()
        
        unsafe_tgt += binary_occupancy_map[self.gs_x,self.gs_y,self.gs_z] + \
                        binary_occupancy_map[(self.gs_x+1).clamp(self.Nx_start,self.Nx_end),self.gs_y,self.gs_z] + \
                        binary_occupancy_map[(self.gs_x-1).clamp(self.Nx_start,self.Nx_end),self.gs_y,self.gs_z]
        unsafe_tgt += binary_occupancy_map[self.gs_x,(self.gs_y+1).clamp(self.Ny_start,self.Ny_end),self.gs_z] + \
                        binary_occupancy_map[self.gs_x,(self.gs_y-1).clamp(self.Ny_start,self.Ny_end),self.gs_z]
        unsafe_tgt += binary_occupancy_map[self.gs_x,self.gs_y,(self.gs_z+1).clamp(0,self.Nz-1)] + \
                        binary_occupancy_map[self.gs_x,self.gs_y,(self.gs_z-1).clamp(0,self.Nz-1)]
        unsafe_tgt = unsafe_tgt.reshape(-1)
        
        sdf = torch.from_numpy(sdf).cuda()
        
        if cell_ids is not None:
            uncerts = []
            voxel_ids = []
            surface_masks = []
            d_x, d_y, d_z = uncert.shape
            voxel_ids_all = np.stack(np.meshgrid(np.arange(d_x), np.arange(d_y), np.arange(d_z), indexing="ij"), -1)
            if only_traversable_vxls:
                for cell_id in cell_ids:
                    region = region_graph.vertices[cell_id]
                    lb_index = region.lb_index
                    tr_index = region.tr_index
                    voxel_ids_cell = voxel_ids_all[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1, 3)
                    uncert_cell = uncert[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1)
                    
                    surface_mask = uncert_cell > 0
                    voxel_ids_surf = voxel_ids_cell[surface_mask]
                    
                    is_split, r1, r2 = region.check_and_split_region()
                    if is_split:
                        r1_center = r1.center_index
                        r2_center = r2.center_index
                        d1 = np.linalg.norm(cur_vxl - r1_center)
                        d2 = np.linalg.norm(cur_vxl - r2_center)
                        center_sel = r1_center if d1 <= d2 else r2_center
                        center_other = r2_center if d1 <= d2 else r1_center
                        dist_sel = np.linalg.norm(voxel_ids_surf - center_sel[None], axis=1)
                        dist_other = np.linalg.norm(voxel_ids_surf - center_other[None], axis=1)
                        traversable_mask = dist_sel < dist_other
                        
                        surface_mask = torch.from_numpy(surface_mask).cuda()
                        traversable_mask = torch.from_numpy(traversable_mask).cuda()
                        surface_mask = surface_mask.masked_scatter(surface_mask.clone(), traversable_mask).cpu().numpy()
                    
                    voxel_ids.append(voxel_ids_cell)
                    uncerts.append(uncert_cell)
                    surface_masks.append(surface_mask)                 
            else:
                for id in cell_ids:
                    region = region_graph.vertices[id]
                    lb_index = region.lb_index
                    tr_index = region.tr_index
                    
                    voxel_ids_cell = voxel_ids_all[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1, 3)
                    uncert_cell = uncert[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1)
                    
                    surface_mask = uncert_cell > 0
                    voxel_ids.append(voxel_ids_cell)
                    uncerts.append(uncert_cell)
                    surface_masks.append(surface_mask)
            
            surface_masks = np.concatenate(surface_masks)
            uncert_in_cell = np.concatenate(uncerts)
            voxel_ids = np.concatenate(voxel_ids)
            if surface_masks.sum() > self.planner_cfg.uncert_top_k:
                uncert_in_cell = uncert_in_cell[surface_masks]
                voxel_ids = voxel_ids[surface_masks]
            
            if only_visible_vxls:
                uncert_mask = uncert_in_cell > self.uncertain_threshold
                uncert_voxel_ids = voxel_ids[uncert_mask]
                n_total_vxl_in_cell = uncert_voxel_ids.shape[0]
                
                safe_gs_pts = self.goal_space_pts[~unsafe_tgt, None, :].repeat([1,n_total_vxl_in_cell,1])  # [safe, k, 3]
                view_vec = safe_gs_pts - torch.from_numpy(uncert_voxel_ids).float().cuda()
                dist = torch.norm(view_vec, dim=2)
                dist_mask = (dist < max_sensing_dist ) * (dist > min_sensing_dist)
                safe_mask = torch.ones_like(safe_gs_pts[:, :, 0]) == 1.
                safe_mask = safe_mask * dist_mask
                near_view_vec = view_vec[safe_mask]
                t_values = torch.linspace(0, 1, 30, device=near_view_vec.device)
                vis_test_points = safe_gs_pts[safe_mask][..., None] - t_values * near_view_vec[..., None]
                vis_test_points = vis_test_points.permute(0, 2, 1).long()
                vis_test_sdf = sdf[vis_test_points[:,:,0], vis_test_points[:,:,1], vis_test_points[:,:,2]]  # [x*y*z*near, 20]
                vis_test_sdf, _ = vis_test_sdf.min(dim=1) 
                visible_mask = vis_test_sdf > 0
                safe_mask = safe_mask.masked_scatter(safe_mask.clone(), visible_mask).sum(dim=0).bool()
                # uncert_in_cell[~safe_mask.cpu().numpy()] = 0.
                uncert_mask = torch.from_numpy(uncert_mask).cuda()
                uncert_mask = uncert_mask.masked_scatter(uncert_mask.clone(), ~safe_mask)
                uncert_in_cell[uncert_mask.cpu().numpy()] = 0.
                        

            n_vxl_in_cell = (uncert_in_cell > 0).sum()
            top_k = max(min(n_vxl_in_cell, self.planner_cfg.uncert_top_k), self.planner_cfg.uncert_top_k_subset)
            top_k_subset = self.planner_cfg.uncert_top_k_subset
            if include_uncert_vxl_only: 
                n_uncert_vxl = (uncert_in_cell > self.uncertain_threshold).sum()
                top_k = max(min(top_k, n_uncert_vxl), top_k_subset)
            top_k = min(uncert_in_cell.shape[0], top_k)
            top_k_subset = min(uncert_in_cell.shape[0], top_k_subset)
        else:
            top_k = self.planner_cfg.uncert_top_k
        
        if cell_ids is None:
            shuffled_indices = np.random.permutation(np.prod(uncert.shape))
            shuffled_uncert = uncert.reshape(-1)[shuffled_indices]
            topk_uncert_vxl = np.argpartition(shuffled_uncert, -top_k, axis=None)[-top_k_subset:]
            topk_uncert_vxl = shuffled_indices[topk_uncert_vxl]
            topk_uncert_vxl = np.unravel_index(topk_uncert_vxl, uncert.shape)
            topk_uncert_vxl = np.column_stack(topk_uncert_vxl)
        else:
            shuffled_indices = np.random.permutation(uncert_in_cell.shape[0])
            shuffled_uncert = uncert_in_cell[shuffled_indices]
            topk_uncert_vxl = np.argpartition(shuffled_uncert, -top_k, axis=None)[-top_k_subset:]
            topk_uncert_vxl = shuffled_indices[topk_uncert_vxl]
            topk_uncert_vxl = voxel_ids[topk_uncert_vxl]
        topk_uncert_vxl = torch.from_numpy(topk_uncert_vxl).cuda().float()

        ### move volumes to TorchTensor and cuda device ###
        uncert = torch.from_numpy(uncert).cuda()

        ##################################################
        ### only consider goal_space_pts within max sensing distance
        ##################################################
        goal_space_pts = self.goal_space_pts[:,None,:].repeat([1,top_k_subset,1])  # [x*y*z, k, 3]
        view_vec = goal_space_pts - topk_uncert_vxl  # [x*y*z, k, 3]
        dist = torch.norm(view_vec, dim=2)  # [x*y*z, k]
        dist_mask = (dist < max_sensing_dist ) * (dist > min_sensing_dist)  # we assume best view distance is between 0.5m and 2m
        if dist_mask.sum() == 0:
            self.info_printer(
                "   Warning! All high uncertainty points are far from Goal Space points",
                self.step,
                self.__class__.__name__ 
                )
        valid_mask = torch.ones_like(goal_space_pts[:, :, 0]) == 1.
        valid_mask = valid_mask * dist_mask
        
        valid_mask[unsafe_tgt,:] = False  # remove points that are not safe to go

        ##################################################
        ### visibility check
        ##################################################
        near_view_vec = view_vec[valid_mask]  # [x*y*z*near, 3]
        t_values = torch.linspace(0, 1, 30, device=near_view_vec.device)
        vis_test_points = goal_space_pts[valid_mask][..., None] - t_values * near_view_vec[..., None]  # [x*y*z*near, 3, 20]
        vis_test_points = vis_test_points.permute(0, 2, 1).long()  # [x*y*z*near, 20, 3]
        vis_test_sdf = sdf[vis_test_points[:,:,0], vis_test_points[:,:,1], vis_test_points[:,:,2]]  # [x*y*z*near, 20]
        ## Visibility check is valid only if sampled points are all with positive sdf ##
        vis_test_sdf, _ = vis_test_sdf.min(dim=1) 
        visible_mask = vis_test_sdf > 0
        if visible_mask.sum() == 0:
            self.info_printer(
                "   Warning! No visible uncertainty points.",
                self.step,
                self.__class__.__name__ 
                )
        
        ### warning message ###
        valid_mask = valid_mask.masked_scatter(valid_mask.clone(), visible_mask)
        
        
        n_rotation_samples = self.fibonacci_samples.shape[0]
        camera_dir = torch.from_numpy(self.fibonacci_samples[None, None, ...]).cuda().float()
        view_vec_normalized = -(view_vec / dist[..., None]).unsqueeze(-2)
        dot_product = (camera_dir * view_vec_normalized).sum(dim=-1)
        frustum_mask = dot_product >= np.cos(half_fov) # [x*y*z, k, n_rotation_samples]
        valid_mask = valid_mask.unsqueeze(-1).repeat(1, 1, n_rotation_samples)
        valid_mask[~frustum_mask] = False
        
        
        if valid_mask.sum() == 0:
            self.info_printer(
                "   Warning! No valid uncertainty points. either too far or non-visible",
                self.step,
                self.__class__.__name__ 
                )
            invalid_goal_space = True
        else:
            invalid_goal_space = False

        ##################################################
        ### aggregate uncertainty
        ##################################################
        topk_uncert_vxl = topk_uncert_vxl.long()
        k_uncerts = uncert[topk_uncert_vxl[:,0], topk_uncert_vxl[:,1], topk_uncert_vxl[:,2]]  # [k]
        k_uncerts = k_uncerts[None, :, None].repeat(goal_space_pts.shape[0], 1, n_rotation_samples) 
        gs_uncert_collections = torch.zeros_like(k_uncerts)
        gs_uncert_collections[valid_mask] = k_uncerts[valid_mask] # [x*y*z, k, n_rotation_samples]
        gs_aggre_uncerts = gs_uncert_collections.sum(dim=1)
        gs_aggre_uncerts = gs_aggre_uncerts.reshape([self.gs_x_range.shape[0], self.gs_y_range.shape[0], self.gs_z_range.shape[0], n_rotation_samples]) # [x, y, z, n_rotation_samples] 

        ### warning message ###
        if torch.count_nonzero(gs_aggre_uncerts) == 0:
            self.info_printer(
                "   Warning! No valid goal space. Going to filter search space. ",
                self.step,
                self.__class__.__name__ 
                )

        ##################################################
        ### gather outputs
        ##################################################
        outputs = {
            'gs_aggre_uncerts': gs_aggre_uncerts, 
            'topk_uncert_vxl': topk_uncert_vxl, 
            'gs_uncert_collections': gs_uncert_collections, 
        }

        if invalid_goal_space:
            if force_running:
                return True, outputs
            else:
                return False, {}
        else:
            return True, outputs
    
    @torch.no_grad()
    def lookats_at_fixed_vxl(self,
                             cur_vxl : np.ndarray, 
                             uncert_sdf_vols: List,
                             cell_ids = None,
                             region_graph = None,
                             half_fov=np.pi/6,
                             n_rots=5,
                             th_obs_per_rot=5,
                             th_mean_unc=1.0,
                            ) -> Tuple[bool, Dict]: 
        ##################################################
        ### Remove the predicted empty space from consideration
        ##################################################
        sdf = uncert_sdf_vols[1]
        uncert = np.copy(uncert_sdf_vols[0])
        ### only consider uncertainty on surface ###
        mask = (sdf >= 0) * (sdf < 0.5) 
        uncert[np.logical_not(mask)] = 0

        sdf = torch.from_numpy(sdf).cuda()
        uncert_in_cell = []
        voxel_ids = []
        d_x, d_y, d_z = uncert.shape
        voxel_ids_all = np.stack(np.meshgrid(np.arange(d_x), np.arange(d_y), np.arange(d_z), indexing="ij"), -1)
        for id in cell_ids:
            region = region_graph.vertices[id]
            lb_index = region.lb_index
            tr_index = region.tr_index
            uncert_in_cell.append(uncert[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1))
            voxel_ids.append(voxel_ids_all[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], lb_index[2]:tr_index[2]].reshape(-1, 3))
        uncert_in_cell = np.concatenate(uncert_in_cell)
        voxel_ids = np.concatenate(voxel_ids)
        
        ##################################################
        ### only consider goal_space_pts within max sensing distance
        ##################################################
        uncert_mask = uncert_in_cell > 1.0 
        uncert_voxel_ids = voxel_ids[uncert_mask]
        n_total_vxl_in_cell = uncert_voxel_ids.shape[0]
        
        goal_space_pts = torch.from_numpy(cur_vxl[None]).repeat([n_total_vxl_in_cell,1]).cuda()
        view_vec = goal_space_pts - torch.from_numpy(uncert_voxel_ids).float().cuda()
        view_vec_all = torch.from_numpy(cur_vxl[None]).cuda() - torch.from_numpy(voxel_ids).float().cuda()
        dist = torch.norm(view_vec, dim=-1)
        dist_all = torch.norm(view_vec_all, dim=-1)
        min_sensing_dist = self.planner_cfg.gs_sensing_range[0] / self.voxel_size
        max_sensing_dist = self.planner_cfg.gs_sensing_range[1] / self.voxel_size
        dist_mask = (dist < max_sensing_dist ) * (dist > min_sensing_dist)  # we assume best view distance is between 0.5m and 2m

        valid_mask = torch.ones_like(goal_space_pts[:, 0]) == 1.
        valid_mask = valid_mask * dist_mask

        ##################################################
        ### visibility check
        ##################################################
        near_view_vec = view_vec[valid_mask]
        t_values = torch.linspace(0, 1, 30, device=near_view_vec.device)
        vis_test_points = goal_space_pts[valid_mask][..., None] - t_values * near_view_vec[..., None]
        vis_test_points = vis_test_points.permute(0, 2, 1).long()
        vis_test_sdf = sdf[vis_test_points[:,:,0], vis_test_points[:,:,1], vis_test_points[:,:,2]]
        ## Visibility check is valid only if sampled points are all with positive sdf ##
        vis_test_sdf, _ = vis_test_sdf.min(dim=1) 
        visible_mask = vis_test_sdf > 0
        
        ### warning message ###
        valid_mask = valid_mask.masked_scatter(valid_mask.clone(), visible_mask)
        
        uncert_mask = torch.from_numpy(uncert_mask).cuda()
        uncert_mask = uncert_mask.masked_scatter(uncert_mask.clone(), ~valid_mask)
        uncert_in_cell[uncert_mask.cpu().numpy()] = 0.
        
        ##################################################
        ### get top-k uncertainty target points
        ##################################################
        n_vxl_in_cell = (uncert_in_cell > 0).sum()
        top_k_subset = min(n_vxl_in_cell, self.planner_cfg.uncert_top_k_subset)
        top_k = min(n_vxl_in_cell, self.planner_cfg.uncert_top_k)

        shuffled_indices = np.random.permutation(uncert_in_cell.shape[0])
        shuffled_uncert = uncert_in_cell[shuffled_indices]
        topk_uncert_vxl = np.argpartition(shuffled_uncert, -top_k, axis=None)[-top_k_subset:]
        topk_uncert_vxl = shuffled_indices[topk_uncert_vxl]
        view_vec_topk = view_vec_all[topk_uncert_vxl]
        dist_topk = dist_all[topk_uncert_vxl]
        # topk_uncert_vxl = np.argpartition(uncert, -top_k, axis=None)[-top_k_subset:]
        topk_uncert_vxl = voxel_ids[topk_uncert_vxl]
        topk_uncert_vxl = torch.from_numpy(topk_uncert_vxl).cuda().float()

        # ### move volumes to TorchTensor and cuda device ###
        uncert = torch.from_numpy(uncert).cuda()
        
        
        n_rotation_samples = self.fibonacci_samples.shape[0]
        camera_dir = torch.from_numpy(self.fibonacci_samples[None, ...]).cuda().float()
        view_vec_normalized = -(view_vec_topk / dist_topk[:, None]).unsqueeze(-2)
        dot_product = (camera_dir * view_vec_normalized).sum(dim=-1)
        frustum_mask = dot_product >= np.cos(half_fov)

        ##################################################
        ### aggregate uncertainty
        ##################################################
        topk_uncert_vxl = topk_uncert_vxl.long()
        k_uncerts = uncert[topk_uncert_vxl[:,0], topk_uncert_vxl[:,1], topk_uncert_vxl[:,2]]  # [k]
        k_uncerts = k_uncerts[:, None].repeat(1, n_rotation_samples) 
        uncert_collections = torch.zeros_like(k_uncerts)
        uncert_collections[frustum_mask] = k_uncerts[frustum_mask]
        
        lookat_tgt_locs = []
        for _ in range(n_rots):
            valid_mask = uncert_collections > 0
            aggre_uncerts = uncert_collections.sum(dim=0)
            max_uncert_rot_ind = torch.argmax(aggre_uncerts)
            seen_voxel_mask = valid_mask[:, max_uncert_rot_ind].clone()
            uncert_collections_cloned = uncert_collections.clone()
            
            n_seen_vxl = seen_voxel_mask.sum()
            seen_unc_aggre = uncert_collections_cloned[seen_voxel_mask, max_uncert_rot_ind].sum()
            if n_seen_vxl < th_obs_per_rot:
                break

            uncert_collections[seen_voxel_mask, :] = 0.
            
            if seen_unc_aggre/n_seen_vxl < th_mean_unc:
                continue
            
            lookat_tgt_loc = self.occupancy_map.Sub2Pos(cur_vxl) + self.fibonacci_samples[max_uncert_rot_ind]
            lookat_tgt_locs.append(lookat_tgt_loc)
        
        return lookat_tgt_locs
        
    def get_target_poses_v2(self, selected_ids, uncert_aggre_outputs, gs_x, gs_y, gs_z, num_target_poses=10, mean_unc_threshold=1.0, n_visible_threshold=10, up_dir=np.array([0, 0, 1])):
        gs_aggre_uncerts = uncert_aggre_outputs['gs_aggre_uncerts']
        gs_uncert_collections = uncert_aggre_outputs['gs_uncert_collections'].clone()
        n_voxels, n_rotation_samples = gs_uncert_collections.shape[-2:]
        
        gs_uncert_collections_selected = []
        gs_x_selected = []
        gs_y_selected = []
        gs_z_selected = []
        for id in selected_ids:
            region = self.subspace_world.region_graph.vertices[id]
            lb_index = region.lb_index // 2
            tr_index = region.tr_index // 2
                        
            gs_uncert_collections_sel = gs_uncert_collections.reshape(list(gs_aggre_uncerts.shape[:-1]) + [n_voxels, n_rotation_samples]) \
                [lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], ...]
            gs_uncert_collections_selected.append(gs_uncert_collections_sel.reshape(-1, n_voxels, n_rotation_samples))
            
            gs_x_sel = gs_x[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :]
            gs_y_sel = gs_y[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :]
            gs_z_sel = gs_z[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :]
            gs_x_selected.append(gs_x_sel.reshape(-1))
            gs_y_selected.append(gs_y_sel.reshape(-1))
            gs_z_selected.append(gs_z_sel.reshape(-1))
            
        gs_uncert_collections_selected = torch.cat(gs_uncert_collections_selected)
        gs_x_selected = np.concatenate(gs_x_selected)
        gs_y_selected = np.concatenate(gs_y_selected)
        gs_z_selected = np.concatenate(gs_z_selected)
        
        valid_mask = gs_uncert_collections_selected > 0
        
        target_poses = []
        th_obs_per_rot = 1 if num_target_poses == 1 else 5 
        for i in range(num_target_poses):
            ret = self.pose_from_aggregated_uncertainty_v2(gs_uncert_collections_selected, gs_x_selected, gs_y_selected, gs_z_selected, up_dir=up_dir, th_obs_per_rot=th_obs_per_rot)
            if ret is None:
                break
            target_pose, target_unc, (max_gs_ind, max_rot_indices) = ret
            
            seen_voxel_mask = torch.stack([valid_mask[max_gs_ind, :, max_rot_ind] for max_rot_ind in max_rot_indices]).sum(dim=0).bool()
            n_visible_vxl = seen_voxel_mask.sum().item()
            if n_visible_vxl < n_visible_threshold or target_unc < mean_unc_threshold:
                break
            
            target_poses.append(target_pose)
            
            valid_mask[:, seen_voxel_mask] = False
            gs_uncert_collections_selected[:, seen_voxel_mask] = 0.
            
        return target_poses
    
    def pose_from_aggregated_uncertainty_v2(self, gs_uncert_collections, gs_x, gs_y, gs_z, rots_per_target=5, th_obs_per_rot=10, th_mean_unc=1.0, up_dir=np.array([0, 0, 1])):
        gs_aggre_uncerts = gs_uncert_collections.sum(dim=1) # [x*y*z, n_rotation_samples]
        max_uncert_ind = torch.argmax(gs_aggre_uncerts).cpu().numpy()
        max_uncert_ind = np.unravel_index(max_uncert_ind, gs_aggre_uncerts.shape)
        max_uncert_ind = np.array(max_uncert_ind)
        max_gs_ind = max_uncert_ind[0]
        
        goal_vxl = np.array([gs_x[max_gs_ind],
                             gs_y[max_gs_ind],
                             gs_z[max_gs_ind]])
        target_loc = self.occupancy_map.Sub2Pos(goal_vxl)
        
        cur_uncert_collections = gs_uncert_collections[max_gs_ind] # [k, n_rotation_samples]
        
        max_rot_inds = [] #max_uncert_ind[1]
        target_poses = []
        target_uncs = 0.
        for _ in range(rots_per_target):
            valid_mask = cur_uncert_collections > 0
            cur_aggre_uncerts = cur_uncert_collections.sum(dim=0)
            max_uncert_rot_ind = torch.argmax(cur_aggre_uncerts)
            seen_voxel_mask = valid_mask[:, max_uncert_rot_ind].clone()
            cur_uncert_collections_cloned = cur_uncert_collections.clone()
            
            n_seen_vxl = seen_voxel_mask.sum()
            seen_unc_aggre = cur_uncert_collections_cloned[seen_voxel_mask, max_uncert_rot_ind].sum()
            if n_seen_vxl < th_obs_per_rot:
                break

            valid_mask[seen_voxel_mask, :] = False
            cur_uncert_collections[seen_voxel_mask, :] = 0.
            
            if seen_unc_aggre/n_seen_vxl < th_mean_unc:
                continue
            
            lookat_tgt_loc = target_loc + self.fibonacci_samples[max_uncert_rot_ind]
            target_pose = np.eye(4)
            target_pose[:3, 3] = target_loc
            target_pose[:3, :3] = compute_camera_pose(target_loc, lookat_tgt_loc, up_dir=up_dir)
            target_poses.append(target_pose)
            max_rot_inds.append(max_uncert_rot_ind)
            target_uncs += seen_unc_aggre
        
        if len(target_poses) == 0:
            return
        
        target_poses = np.stack(target_poses)    
        target_uncs = (target_uncs/len(target_poses)).item()
        
        return target_poses, target_uncs, (max_gs_ind, max_rot_inds)
    
    def get_target_poses_ablate(self, selected_ids, uncert_aggre_outputs, gs_x, gs_y, gs_z, up_dir=np.array([0, 0, 1]), th_mean_unc=1.0):
        sh = list(uncert_aggre_outputs['gs_aggre_uncerts'].shape)
        topk_uncert_vxl = uncert_aggre_outputs['topk_uncert_vxl']
        gs_uncert_collections = uncert_aggre_outputs['gs_uncert_collections']
        n_voxels = gs_uncert_collections.shape[-1]
        
        gs_uncert_collections_selected = []
        gs_x_selected = []
        gs_y_selected = []
        gs_z_selected = []
        for id in selected_ids:
            region = self.subspace_world.region_graph.vertices[id]
            lb_index = region.lb_index // 2
            tr_index = region.tr_index // 2
                        
            gs_uncert_collections_sel = gs_uncert_collections.reshape(sh + [n_voxels]) \
                [lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], ...]
            gs_uncert_collections_selected.append(gs_uncert_collections_sel.reshape(-1, n_voxels))
            
            gs_x_sel = gs_x[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :]
            gs_y_sel = gs_y[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :]
            gs_z_sel = gs_z[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :]
            gs_x_selected.append(gs_x_sel.reshape(-1))
            gs_y_selected.append(gs_y_sel.reshape(-1))
            gs_z_selected.append(gs_z_sel.reshape(-1))
            
        gs_uncert_collections_selected = torch.cat(gs_uncert_collections_selected)
        gs_aggre_uncerts = gs_uncert_collections_selected.sum(dim=1).cpu().numpy()
        gs_x_selected = np.concatenate(gs_x_selected)
        gs_y_selected = np.concatenate(gs_y_selected)
        gs_z_selected = np.concatenate(gs_z_selected)
        
        num_target_poses = len(selected_ids)
        top_k = np.minimum(gs_aggre_uncerts.shape[0], 9*num_target_poses)
        
        shuffled_indices = np.random.permutation(gs_aggre_uncerts.shape[0])
        shuffled_aggre = gs_aggre_uncerts[shuffled_indices]
        max_uncert_inds = np.argpartition(shuffled_aggre, -top_k, axis=None)[-num_target_poses:]
        max_uncert_inds = shuffled_indices[max_uncert_inds]
        
        goal_vxls = np.stack([gs_x_selected[max_uncert_inds],
                                    gs_y_selected[max_uncert_inds],
                                    gs_z_selected[max_uncert_inds]], axis=-1)
        goal_locs = np.stack([self.occupancy_map.Sub2Pos(goal_vxl) for goal_vxl in goal_vxls])
        
        uncert_obs_from_goals = gs_uncert_collections_selected[max_uncert_inds]
        
        lst_lookat_target_vals, lst_lookat_tgt_vxls = uncert_obs_from_goals.topk(k=self.planner_cfg.obs_per_goal, largest=True, dim=-1) 
        target_poses = []
        for i in range(num_target_poses):
            if (uncert_obs_from_goals[i] > 0).sum() < self.planner_cfg.obs_per_goal:
                continue
            
            lookat_target_vals = lst_lookat_target_vals[i]
            lookat_tgt_vxls = lst_lookat_tgt_vxls[i]
            if lookat_target_vals.mean() < th_mean_unc:
                continue
            
            lookat_tgt_vxls = lookat_tgt_vxls[:(lookat_target_vals > 0).sum()]
            lookat_tgt_vxls = topk_uncert_vxl[lookat_tgt_vxls].cpu().numpy()
            target_pose = []
            for ii, lookat_tgt_vxl in enumerate(lookat_tgt_vxls):
                pose = np.eye(4)
                pose[:3, 3] = goal_locs[i]
                lookat_tgt_loc = self.occupancy_map.Sub2Pos(lookat_tgt_vxl)
                pose[:3, :3] = compute_camera_pose(goal_locs[i], lookat_tgt_loc, up_dir=up_dir)

                target_pose.append(pose)
            
            if len(target_pose) > 1:
                ref_rot = Mat2Rotation(target_pose[0][:3, :3])
                lst_rot = [Mat2Rotation(pose[:3, :3]) for pose in target_pose[1:]]
            
                sorted_rotations = minimize_movement(lst_rot, ref_rot, min_ang=self.planner_cfg.ang_max)
                target_pose = [np.vstack((np.hstack((Rotation2Mat(rot), target_pose[0][:3, 3:])), np.array([[0.,0.,0.,1.]]))) for rot in sorted_rotations]
            
            target_pose = np.stack(target_pose)
            target_poses.append(target_pose)
            
        return target_poses


    def update_uncertainty_scores(self, selected_ids, uncert_sdf_vols, uncert_aggre_outputs, gs_x, gs_y, gs_z, aggregate_in_cur_cell=False, up_dir=np.array([0, 0, 1])):
        region_graph = self.subspace_world.region_graph
        
        sdf = uncert_sdf_vols[1]
        uncert = np.copy(uncert_sdf_vols[0])
        ### only consider uncertainty on surface ###
        mask = (sdf >= 0) * (sdf < 0.5) 
        uncert[np.logical_not(mask)] = 0
        t00 = time.time()
        gs_aggre_uncerts = uncert_aggre_outputs['gs_aggre_uncerts']
        topk_uncert_vxl = uncert_aggre_outputs['topk_uncert_vxl']
        gs_uncert_collections = uncert_aggre_outputs['gs_uncert_collections']
        n_voxels, n_rotation_samples = gs_uncert_collections.shape[-2:]
        t01 = time.time()
        gs_uncert_collections = gs_uncert_collections.reshape(list(gs_aggre_uncerts.shape[:-1]) + [n_voxels, n_rotation_samples])
        for id in selected_ids:
            region = region_graph.vertices[id]
            lb_index_cuda = torch.from_numpy(region.lb_index).cuda()
            tr_index_cuda = torch.from_numpy(region.tr_index).cuda()

            in_cell = torch.logical_and(torch.all(topk_uncert_vxl>=lb_index_cuda, dim=-1), torch.all(topk_uncert_vxl<tr_index_cuda, dim=-1))

            cur_uncert_vxl = topk_uncert_vxl[in_cell]
            n_vxl_cur = cur_uncert_vxl.shape[0]
            
            if n_vxl_cur == 0:
                region.update_uncertainty_score(0)
                region.target_pose[:3, 3] = region.center
                continue
            
            if aggregate_in_cur_cell:
                lb_index = lb_index_cuda // 2
                tr_index = tr_index_cuda // 2
                gs_uncert_collections_sel = gs_uncert_collections \
                    [lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :, in_cell].reshape(-1, n_vxl_cur, n_rotation_samples)
                gs_x_sel = gs_x[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :].reshape(-1)
                gs_y_sel = gs_y[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :].reshape(-1)
                gs_z_sel = gs_z[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :].reshape(-1)
            else:
                gs_uncert_collections_sel = []
                aggregate_ids = [id]
                aggregate_ids.extend(region_graph.dct_neighbor_idx[id])
                # aggregate_ids.extend(region_graph.dct_diagonal_idx[id])
                gs_x_sel = []
                gs_y_sel = []
                gs_z_sel = []
                for agg_id in aggregate_ids:
                    n_region = region_graph.vertices[agg_id]
                    lb_index = torch.from_numpy(n_region.lb_index // 2).cuda()
                    tr_index = torch.from_numpy(n_region.tr_index // 2).cuda()
                    gs_uncert_collections_sel.append(gs_uncert_collections \
                        [lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :, in_cell].reshape(-1, n_vxl_cur, n_rotation_samples))
                    gs_x_sel.append(gs_x[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :].reshape(-1))
                    gs_y_sel.append(gs_y[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :].reshape(-1))
                    gs_z_sel.append(gs_z[lb_index[0]:tr_index[0], lb_index[1]:tr_index[1], :].reshape(-1))
                    
                gs_uncert_collections_sel = torch.cat(gs_uncert_collections_sel)
                gs_x_sel = np.concatenate(gs_x_sel)
                gs_y_sel = np.concatenate(gs_y_sel)
                gs_z_sel = np.concatenate(gs_z_sel)
            
            valid_mask = gs_uncert_collections_sel > 0
            valid_vxl_mask = valid_mask.sum(dim=(0, 2)).bool()
            valid_vxl = cur_uncert_vxl[valid_vxl_mask].cpu().numpy()
            
            uncert_score = uncert[valid_vxl[:, 0], valid_vxl[:, 1], valid_vxl[:, 2]].sum()
            
            region.update_uncertainty_score(uncert_score, valid_vxl=valid_vxl)
            if valid_vxl.shape[0] == 0:
                region.target_pose[:3, 3] = region.center
            else:
                gs_uncert_aggre_sel = gs_uncert_collections_sel.sum(axis=1)
                max_uncert_ind = torch.argmax(gs_uncert_aggre_sel).item()
                max_uncert_ind = np.unravel_index(max_uncert_ind, gs_uncert_aggre_sel.shape)
                max_uncert_ind = np.array(max_uncert_ind)
                max_gs_ind = max_uncert_ind[0]
                max_rot_ind = max_uncert_ind[1]
                
                goal_vxl = np.array([gs_x_sel[max_gs_ind],
                                     gs_y_sel[max_gs_ind],
                                     gs_z_sel[max_gs_ind]])
                target_loc = self.occupancy_map.Sub2Pos(goal_vxl)
                lookat_tgt_loc = target_loc + self.fibonacci_samples[max_rot_ind]
                
                region.target_pose[:3, 3] = self.occupancy_map.Sub2Pos(goal_vxl)
                region.target_pose[:3,:3] = compute_camera_pose(target_loc, lookat_tgt_loc, up_dir=up_dir)
    
    
    def get_next_global_region_id(self, cur_pos, uncert_sdf_vols, find_voxel_path=True, n_visible_threshold=10):
        cur_region_id = self.subspace_world.subspaces.Pos2Ind(cur_pos)
        region_graph = self.subspace_world.region_graph   
        
        global_region_order = [self.goal_region]
        global_region_order.extend(self.global_region_order)
        
        selected_ids = []
        for id in global_region_order:
            selected_ids.append(id)
            selected_ids.extend(region_graph.dct_neighbor_idx[id])
        selected_ids = np.unique(selected_ids)
        
        is_valid, uncert_aggre_outputs = self.uncertainty_aggregation_in_frustum(uncert_sdf_vols, cell_ids=selected_ids, region_graph=region_graph, half_fov=self.fov/2)
        if not is_valid:
            return -1, [], None
        
        for region_id in global_region_order:
            n_region_ids = copy.deepcopy(region_graph.dct_neighbor_idx[region_id])
            n_region_ids.append(region_id)
            target_poses = self.get_target_poses_v2(n_region_ids, uncert_aggre_outputs, self.gs_x, self.gs_y, self.gs_z, num_target_poses=1, n_visible_threshold=n_visible_threshold) 
            if len(target_poses) == 0:
                continue
            poses_in_cell = target_poses[0] 
            
            next_global_goal = poses_in_cell[0, :3, 3]
            good_rot = poses_in_cell[0, :3, :3]
            # Divide the method into two types, 1) find voxel path (original) and 2) find region id path. Region id path extracts sequence of region id instead of voxel-wise path. 
            if find_voxel_path:
                # Find voxel-wise path with a-star planning
                sub_path, is_path_found = self.subspace_world.astar_planning_region_mask(cur_pos, 
                                                                                        next_global_goal, 
                                                                                        region_graph.vertices.keys(), 
                                                                                        self.occupancy_map)
                if is_path_found:
                    return region_id, sub_path, good_rot    
            else:
                # Find region-id path
                shortest_dist, region_id_path = region_graph.find_region_graph_astar_path(cur_region_id, region_id)
                if shortest_dist > 0:
                    return region_id, region_id_path, None
        return -1, [], None