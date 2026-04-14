import numpy as np
import copy
from scipy.spatial import KDTree as kdtree
from .grid import CellStatus
from .viewpoint import ViewPoint
from .grid import OccupancyGrid
from .rolling_grid import RollingGrid

class ViewPointManager(object):
    def __init__(self, 
                 bound,
                 number_x=80,
                 number_y=80,
                 number_z=40,
                 resolution_x=0.5,
                 resolution_y=0.5,
                 resolution_z=0.5,
                 connectivity_height_diff_thr=0.25,
                 viewpoint_collision_margin=0.5,
                 viewpoint_collision_margin_zplus=0.5,
                 viewpoint_collision_margin_zminus=0.5,
                 collision_grid_zscale=2.0,
                 collision_grid_resolution_x=0.5,
                 collision_grid_resolution_y=0.5,
                 collision_grid_resolution_z=0.5,
                 collision_frame_count_max=3,
                 collision_point_thr=3
                ):
        self.number_x = number_x
        self.number_y = number_y
        self.number_z = number_z
        self.number = [number_x, number_y, number_z]
        self.rollover_step_size = np.array(self.number) / 5.0
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.resolution = [resolution_x, resolution_y, resolution_z]
        self.connectivity_height_diff_thr = connectivity_height_diff_thr
        self.viewpoint_collision_margin = viewpoint_collision_margin
        self.viewpoint_collision_margin_zplus = viewpoint_collision_margin_zplus
        self.viewpoint_collision_margin_zminus = viewpoint_collision_margin_zminus
        self.collision_grid_zscale = collision_grid_zscale
        self.collision_grid_resolution_x = collision_grid_resolution_x
        self.collision_grid_resolution_y = collision_grid_resolution_y
        self.collision_grid_resolution_z = collision_grid_resolution_z
        collision_grid_resolution = [collision_grid_resolution_x, collision_grid_resolution_y, collision_grid_resolution_z]
        self.collision_grid_resolution = collision_grid_resolution
        self.collision_frame_count_max = collision_frame_count_max
        self.collision_point_thr = collision_point_thr

        self.collision_grid_size = np.ones(shape=(3,),dtype=np.int64)
        self.neighbor_range = 3.0
        self.dimension = 2
        
        self.initialized = False

        for i in range(self.dimension):
            self.collision_grid_size[i] = np.ceil(self.number[i]*self.resolution[i]+self.viewpoint_collision_margin*2) / collision_grid_resolution[i]
                    
        self.viewpoint_number = number_x * number_y * number_z
        self.__candidate_indices = set()
        self.grid = RollingGrid([self.number_x, self.number_y, self.number_z], bound)
        self.origin = np.zeros(3)
        
        self.viewpoints = dict()
        for x in range(self.number_x):
            for y in range(self.number_y):
                for z in range(self.number_z):
                    sub = np.array([x,y,z])
                    ind = self.grid.Sub2Ind(sub)
                    self.viewpoints[ind] = ViewPoint()

        self.graph_index_map = [None] * self.viewpoint_number
        for i in range(len(self.graph_index_map)):
            self.graph_index_map[i] = -1

        self.compute_connected_neighbor_indices()
        self.compute_in_range_neighbor_indices()
        # self.get_collision_correspondence()
        
    def compute_connected_neighbor_indices(self):
        self.connected_neighbor_indices = [[]] * self.viewpoint_number
        self.connected_neighbor_dist = [[]] * self.viewpoint_number
        idx_addon = []
        for x in range(-1,2):
            for y in range(-1,2):
                for z in range(-1,2):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    idx_addon.append(np.array([x,y,z]))
        for x in range(self.number_x):
            for y in range(self.number_y):
                for z in range(self.number_z):
                    sub = np.array([x,y,z])
                    ind = self.grid.Sub2Ind(sub)
                    for i in range(len(idx_addon)):
                        neighbor_sub = sub + idx_addon[i]
                        if self.grid.in_range(neighbor_sub):
                            self.connected_neighbor_indices[ind].append(self.grid.Sub2Ind(neighbor_sub))
                            dist = np.sqrt(self.resolution_x * self.resolution_x * np.abs(idx_addon[i][0])
                                                  + self.resolution_y * self.resolution_y * np.abs(idx_addon[i][1])
                                                  + self.resolution_z * self.resolution_z * np.abs(idx_addon[i][2])
                                                  )
                            self.connected_neighbor_dist[ind].append(dist) 
                               
    def compute_in_range_neighbor_indices(self):
        viewpoint_array = []
        for i in range(self.viewpoint_number):
            sub = self.grid.Ind2Sub(i)
            point = np.zeros(3)
            point[0] = sub[0]*self.resolution_x + self.resolution_x * 0.5
            point[1] = sub[1]*self.resolution_y + self.resolution_y * 0.5
            point[2] = sub[2]*self.resolution_z + self.resolution_z * 0.5
            viewpoint_array.append(point)
            
        viewpoint_array = np.array(viewpoint_array)
        self.viewpoint_array = viewpoint_array
        kdt = kdtree(data=viewpoint_array)
        self.kdt = kdt
        self.kdt_2d = kdtree(data=viewpoint_array[:,:2])
        self.in_range_neighbor_indices = [set()]*self.viewpoint_number

        for i in range(self.viewpoint_number):
            point = viewpoint_array[i]
            indices = kdt.query_ball_point(point, r=self.neighbor_range)
            for ind in indices:
                self.in_range_neighbor_indices[i].add(ind)

    def update_viewpoint_collision(self, occupancy_grid:OccupancyGrid):
        self.update_viewpoint_collision_with_sdf(occupancy_grid)
        
    def update_viewpoint_collision_with_sdf(self, occupancy_grid:OccupancyGrid):        
        for idx in self.viewpoints.keys():
            viewpoint = self.viewpoints[idx]
            p = viewpoint.position
            if occupancy_grid.is_feasible_point(p[0], p[1], p[2]):
                viewpoint.set_in_collision(False)
            else:
                viewpoint.set_in_collision(True)

    def viewpoint_in_collision(self, idx):
        return self.viewpoints[idx].is_in_collision()
            
    def update_viewpoints_visited(self, visited_positions):
        if not self.initialized:
            return
        for position in visited_positions:
            if not self.in_local_planning_horizon(position):
                continue
            viewpoint_sub = self.get_viewpoint_sub(position)
            if self.grid.in_range(viewpoint_sub):
                viewpoint_ind = self.grid.Sub2Ind(viewpoint_sub)
                self.set_viewpoint_visited(viewpoint_ind, True)
                for neighbor_viewpoint_ind in self.in_range_neighbor_indices[viewpoint_ind]:
                    assert self.grid.in_range(neighbor_viewpoint_ind)
                    self.set_viewpoint_visited(neighbor_viewpoint_ind, True)
        
    def get_viewpoint_sub(self, robot_position):
        diff = robot_position - self.origin
        sub = [0, 0, 0]
        for i in range(self.dimension):
            if diff[i] > 0:
                sub[i] = int(diff[i] / self.resolution[i])
            else:
                sub[i] = -1
        return sub
                   
    def get_candidate_indices(self):
        return self.__candidate_indices

    def get_viewpoint_cell_ind(self, viewpoint_ind, use_array_ind=True):
        array_ind = self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].get_cell_ind()

    def set_viewpoint_in_exploring_cell(self, viewpoint_ind, in_exploring_cell, use_array_ind=False):
        array_ind = self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        self.viewpoints[array_ind].set_in_exploring_cell(in_exploring_cell)
        
    def update_candidate_viewpoint_cell_status(self, grid_world):
        for ind in self.__candidate_indices:
            cell_ind = self.get_viewpoint_cell_ind(ind)
            if grid_world.ind_in_bound(cell_ind):
                cell_status = grid_world.get_cell_status(cell_ind)
                if cell_status ==CellStatus.UNSEEN or cell_status == CellStatus.EXPLORING:
                    self.set_viewpoint_in_exploring_cell(ind, True)
                    print(f"viewpoint in exploring cell!   viewpoint candidate: {ind} in cell {cell_ind}")
                else:
                    self.set_viewpoint_in_exploring_cell(ind, False)
            else:
                print(f"ViewPointManager::UpdateCandidateViewPointCellStatus: cell ind {cell_ind} out of bound")
                       
    def update_origin(self):
        for i in range(self.dimension):
            self.origin[i] = self.robot_position[i] - (self.resolution[i] * self.number[i]) * 0.5
            
    def update_robot_position(self, robot_position):
        self.robot_position = robot_position
        if not self.initialized:
            self.initialized = True
            self.update_origin()
            for x in range(self.number_x):
                for y in range(self.number_y):
                    for z in range(self.number_z):
                        ind = self.grid.Sub2Ind([x,y,z])
                        position = copy.deepcopy(self.origin)
                        position[0] += x * self.resolution[0] + self.resolution[0] * 0.5
                        position[1] += y * self.resolution[1] + self.resolution[1] * 0.5
                        position[2] = self.robot_position[2]
                        self.set_viewpoint_position(ind, position, True)
                        self.reset_viewpoint(ind, True)
        
        diff = robot_position - self.origin
        robot_grid_sub = np.zeros((3,), dtype=np.int64)
        for i in range(self.dimension):
            if diff[i] > 0:
                robot_grid_sub[i] = int(diff[i] / (self.rollover_step_size[i] * self.resolution[i]))
            else:
                robot_grid_sub[i] = -1
        
        sub_diff = np.zeros((3,), dtype=np.int64)
        for i in range(self.dimension):
            sub_diff[i] = (self.number[i] / self.rollover_step_size[i] ) * 0.5 - robot_grid_sub[i]
        
        if sub_diff[0] == 0 and sub_diff[1] == 0 and sub_diff[2] == 0:
            self.viewpoint_array = np.array([self.viewpoints[array_ind].position for array_ind in self.viewpoints.keys()])
            self.kdt_2d = kdtree(data=self.viewpoint_array[:,:2])
            return False
        
        rollover_step = np.zeros((3,), dtype=np.int64)
        if np.abs(sub_diff[0]) > 0:
            rollover_step[0] = self.rollover_step_size[0] *  np.sign(sub_diff[0]) * np.abs(sub_diff[0])
        if np.abs(sub_diff[1]) > 0:
            rollover_step[1] = self.rollover_step_size[1] *  np.sign(sub_diff[1]) * np.abs(sub_diff[1])
        if np.abs(sub_diff[2]) > 0:
            rollover_step[2] = self.rollover_step_size[2] *  np.sign(sub_diff[2]) * np.abs(sub_diff[2])
        
        self.grid.roll(copy.deepcopy(rollover_step))
        print(f"rolling x: {rollover_step[0]:.3f}  y: {rollover_step[1]:.3f}  z: {rollover_step[2]:.3f}")
        # reset viewpoint
        print("reset viewpoint")
        
        self.origin += -1.0 * rollover_step * np.array(self.resolution)
        
        self.updated_viewpoint_indices = self.grid.get_updated_indices()
        for ind in self.updated_viewpoint_indices:
            assert self.grid.in_range(ind)
            sub = self.grid.Ind2Sub(ind)
            new_position = self.origin + sub * np.array(self.resolution) + np.array(self.resolution) * 0.5
            new_position[2] = robot_position[2]
            self.set_viewpoint_position(ind, new_position)
            self.reset_viewpoint(ind)

        self.viewpoint_array = np.array([self.viewpoints[array_ind].position for array_ind in self.viewpoints.keys()])
        self.kdt_2d = kdtree(data=self.viewpoint_array[:,:2])
        print("reset viewpoint finished")
        return True
            
    def reset_viewpoint(self, viewpoint_ind, use_array_ind=True):
        array_ind = self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        self.viewpoints[array_ind].reset()
        
    def set_viewpoint_position(self, ind, position, use_array_ind=True):
        array_ind = self.get_viewpoint_array_ind(ind, use_array_ind)
        self.viewpoints[array_ind].set_position(position)
        
    def get_viewpoint_array_ind(self, viewpoint_ind:int, use_array_ind:bool):
        # assert self.__grid.in_range(viewpoint_ind)
        if use_array_ind:
            return viewpoint_ind
        else:
            return self.grid.get_array_ind(viewpoint_ind)
                
    def is_viewpoint_candidate(self, viewpoint_ind, use_array_ind=False):
        array_ind = self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].is_candidate()
    
    def viewpoint_selected(self, viewpoint_ind, use_array_ind=False):
        array_ind = self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].is_selected()
    
    def viewpoint_visited(self, viewpoint_ind, use_array_ind=False):
        array_ind = self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].is_visited()
        
    def get_viewpoint_covered_point_num(self, viewpoint_ind, use_array_ind=False):
        array_ind =  self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].get_covered_point_num()
        
    def get_viewpoint_covered_frontier_point_num(self, viewpoint_ind, use_array_ind=False):
        array_ind =  self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].get_covered_frontier_point_num()
    
    def get_viewpoint_position(self, viewpoint_ind, use_array_ind=False):
        array_ind =  self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].get_position()

    def set_viewpoint_visited(self, viewpoint_ind, visited, use_array_ind=False):
        array_ind = self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        return self.viewpoints[array_ind].set_visited(visited)
                
    def get_viewpoint_ind(self, position):
        sub = self.get_viewpoint_sub(position)
        if self.grid.in_range(sub):
            return self.grid.Sub2Ind(sub)
        else:
            return -1
    
    def set_viewpoint_cell_ind(self, viewpoint_ind, cell_ind, use_array_ind=False):
        array_ind =  self.get_viewpoint_array_ind(viewpoint_ind, use_array_ind)
        self.viewpoints[array_ind].set_cell_ind(cell_ind)
    
    def in_local_planning_horizon(self, position:np.ndarray):
        viewpoint_ind = self.get_viewpoint_ind(position)
        if self.grid.in_range(viewpoint_ind):
            max_z_diff = max(self.resolution[0], self.resolution[1]) * 2.0
            viewpoint_position = self.get_viewpoint_position(viewpoint_ind)
            if abs(viewpoint_position[2] - position[2]) < max_z_diff \
                and (self.is_viewpoint_candidate(viewpoint_ind) or self.viewpoint_in_collision(viewpoint_ind)):
                return True
        return False
    