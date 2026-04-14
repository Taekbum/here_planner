import numpy as np
import math
import copy
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any

from .graph import Graph
from .grid import Cell, Grid
from ..utils.utils import raycast, check_occupancy, connected_component_labelling, connected_component_3d_labelling
from .astar import restricted_astar_search
import time
import torch

FREE=1;
UNCERTAIN=0;

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

def Pos2Sub(pos, resolution, origin):
    sub = np.zeros((len(pos),))
    for i in range(len(pos)):
        if pos[i] >= origin[i] - 0.5*resolution:
            sub[i] = int((pos[i] - origin[i]) * resolution)
        else:
            sub[i] = -1
    return sub
def Sub2Pos(sub, resolution, origin):
    pos = np.zeros((3,))
    for i in range(len(sub)):
        pos[i] = origin[i] + sub[i] * resolution + resolution * 0.5
    return pos

class Region(object):
    '''
    Node object for graph structure
    Contain uncertain / free region based on its type
    '''
    def __init__(self, cell:Cell, id, color, masked_indices=None, voxel_res=0.1, origin=None, bound=None, z_ref_idx=10):
        self.type = UNCERTAIN # 0: uncertain, 1: free
        self.cell = cell # its parent cell
        # self.center = cell.get_position()[:2]
        # Compute Voxel Index Set
        self.color = color
        self.id = id
        
        self.voxel_res = voxel_res
        self.origin = origin
        self.bound = bound
        self.cell_bound = cell.cell_bound
        print("cell bound: ", self.cell_bound)
        # True: uncertainty of all pixels in the cell is low
        # This variable is to decide whether we want to split this cell, or update connectivity with neighboring cells
        self.lst_pixel_index = []
        self.masked_indices = np.array(masked_indices).astype(int) # Inside the cell (square), mask its corresponding pixels 

        if not self.masked_indices is None:
            self.lb_index = np.min(self.masked_indices, axis=0)
            self.tr_index = np.max(self.masked_indices, axis=0)

        # self.center_index = self.masked_indices[masked_indices.shape[0]//2,:]
        mean_index = np.mean(self.masked_indices, axis=0)
        dists = np.linalg.norm((self.masked_indices - mean_index), axis=-1)
        self.center_index = self.masked_indices[np.argmin(dists)]
        # self.center_index = np.median(self.masked_indices, axis=0) 
        self.center = Sub2Pos([self.center_index[0], self.center_index[1], self.center_index[2]], self.voxel_res, self.origin) # assuming self.center_index is reasonable geometric center of 3D region
        
        self.unocc_indices = copy.deepcopy(self.masked_indices)
        self.occ_indices = []
        
        self.z_ref_idx = z_ref_idx
        
        self.region_initialized = False
        self.target_poses = []
        self.target_unc = []
        self.topk_uncert_vxl = None
        
        self.uncert_score = 0
        self.target_pose = np.eye(4)
        self.target_pose[:3, 3] = self.center
        
        
        self.is_in_planning_bound = True
        self.valid_vxl = None
    
    def is_inside(self, pos):        
        return np.all(self.cell_bound[0] <= pos) and np.all(self.cell_bound[1] > pos) 
    
    def get_position(self):
        return self.center
    
    def update_occupancy_region(self, occupancy_map):
        indices = self.masked_indices  # Convert to NumPy array

        occupancy_data = occupancy_map.occupancy_data
        # occupancy_data = occupancy_map.data
        bool_inf_data = occupancy_map.inf_data
        
        
        self.occupancy_data = occupancy_data
        self.bool_inf_data = bool_inf_data
        self.unocc_indices = indices[occupancy_data[indices[:,0],indices[:,1],indices[:,2]] <=0]
        self.inf_unocc_indices = indices[bool_inf_data[indices[:,0],indices[:,1],indices[:,2]] ==0]
        new_occ_indices = indices[occupancy_data[indices[:,0],indices[:,1],indices[:,2]] >0]
        new_inf_occ_indices = indices[bool_inf_data[indices[:,0],indices[:,1],indices[:,2]] == 1]
        
        occ_data_changed = False
        if len(new_occ_indices) != len(self.occ_indices):
            # If there is possibility that region is divided, 
            occ_data_changed = True
            
        self.occ_indices = new_occ_indices
        self.inf_occ_indices = new_inf_occ_indices
        
        if np.any(np.all(self.occ_indices == self.center_index, axis=1)) and len(self.unocc_indices) > 0:
            # self.center_index = self.unocc_indices[self.unocc_indices.shape[0]//2,:]
            # self.center_index = np.median(self.inf_unocc_indices, axis=0) 
            mean_index = np.mean(self.unocc_indices, axis=0)
            dists = np.linalg.norm((self.unocc_indices - mean_index), axis=-1)
            self.center_index = self.unocc_indices[np.argmin(dists)]
            self.center = Sub2Pos([self.center_index[0], self.center_index[1], self.center_index[2]], self.voxel_res, self.origin) # assuming self.center_index is reasonable geometric center of 3D region
        
        if not self.region_initialized:
            self.lb_index = occupancy_map.Pos2Sub([self.cell_bound[0][0],
                                                self.cell_bound[0][1],
                                                self.cell_bound[0][2]])
            self.lb_index[0], self.lb_index[1], self.lb_index[2] = max(0, self.lb_index[0]), max(0, self.lb_index[1]), max(0, self.lb_index[2])
            self.tr_index = occupancy_map.Pos2Sub([self.cell_bound[1][0],
                                                self.cell_bound[1][1],
                                                self.cell_bound[1][2]])
            self.tr_index[0], self.tr_index[1], self.tr_index[2] = min(occupancy_data.shape[0], self.tr_index[0]), \
                                                                min(occupancy_data.shape[1], self.tr_index[1]), \
                                                                min(occupancy_data.shape[2], self.tr_index[2])
                                                                
            self.update_goal_space() 
            self.update_voxel_range()

        return occ_data_changed

    def update_goal_space(self):
        self.Nx, self.Ny, self.Nz = self.tr_index - self.lb_index
        self.gs_x_range = torch.arange(0, self.Nx, 2)
        self.gs_y_range = torch.arange(0, self.Ny, 2)
        self.gs_z_range = torch.arange(int(1/self.voxel_res), self.Nz, int(1/self.voxel_res))
        self.gs_x, self.gs_y, self.gs_z = torch.meshgrid(self.gs_x_range, self.gs_y_range, self.gs_z_range, indexing="ij")
        self.goal_space_pts = torch.cat([self.gs_x.reshape(-1, 1), 
                                         self.gs_y.reshape(-1, 1), 
                                         self.gs_z.reshape(-1, 1)], dim=1).cuda().float() + torch.from_numpy(self.lb_index).cuda()
        
    def update_voxel_range(self, include_neighbor_cells=True):
        if include_neighbor_cells:
            x_min = max(0, self.lb_index[0] - self.Nx)
            y_min = max(0, self.lb_index[1] - self.Ny)
            z_min = max(0, self.lb_index[2] - self.Nz)
            x_max = min(self.occupancy_data.shape[0], self.tr_index[0] + self.Nx)
            y_max = min(self.occupancy_data.shape[1], self.tr_index[1] + self.Ny)
            z_max = min(self.occupancy_data.shape[2], self.tr_index[2] + self.Nz)
            self.voxel_min = np.array([x_min, y_min, z_min])
            self.voxel_max = np.array([x_max, y_max, z_max])
        else:
            self.voxel_min = self.lb_index
            self.voxel_max = self.tr_index
        
    def check_and_split_region_2d(self):
        x_min_idx = min(self.masked_indices[:,0])
        y_min_idx = min(self.masked_indices[:,1])
        z_min_idx = min(self.masked_indices[:,2])
        
        x_len = max(self.masked_indices[:,0]) - x_min_idx + 1
        y_len = max(self.masked_indices[:,1]) - y_min_idx + 1
        z_len = max(self.masked_indices[:,2]) - z_min_idx + 1
        
        # idx_x = self.masked_indices[..., 0]
        # idx_y = self.masked_indices[..., 1]

        mask_in_ref_z = self.masked_indices[self.masked_indices[:,2] == self.z_ref_idx]
        occ_in_ref_z = self.occ_indices[self.occ_indices[:,2] == self.z_ref_idx]

        if len(occ_in_ref_z) > 0:
            o_idx_x = occ_in_ref_z[..., 0]
            o_idx_y = occ_in_ref_z[..., 1]
            
            m_idx_x = mask_in_ref_z[..., 0]
            m_idx_y = mask_in_ref_z[..., 1]
            labels_in = np.zeros((x_len,y_len)).astype(np.int32)
            labels_in[m_idx_x-x_min_idx, m_idx_y-y_min_idx] = 1
            labels_in[o_idx_x-x_min_idx, o_idx_y-y_min_idx] = 0
            # labels_in[idx_x-x_min_idx, idx_y-y_min_idx] = 1 - (self.occupancy_data[idx_x, idx_y, self.z_ref_idx] == 1)
            
            is_split, first_region_indices, second_region_indices = connected_component_labelling(labels_in)
            # first_region_indices = np.vstack((first_region_indices,))
        else:
            is_split = False
            
        if is_split:
            third_column = np.arange(z_len)
            
            first_repeated_array = np.repeat(first_region_indices, repeats=z_len, axis=0)
            tiled_third_column = np.tile(third_column, reps=len(first_region_indices))
            result_first_region_indices = np.column_stack((first_repeated_array, tiled_third_column))
            
            second_repeated_array = np.repeat(second_region_indices, repeats=z_len, axis=0)
            second_tiled_third_column = np.tile(third_column, reps=len(second_region_indices))
            result_second_region_indices = np.column_stack((second_repeated_array, second_tiled_third_column))
            
            result_first_region_indices += np.min(self.masked_indices, axis=0)
            result_second_region_indices += np.min(self.masked_indices, axis=0)
            
            result_first_region_indices = np.vstack((result_first_region_indices, self.occ_indices))
            r1 = Region(self.cell, self.id*10+1, self.color, result_first_region_indices, self.voxel_res, self.origin, self.bound, self.z_ref_idx)
            r2 = Region(self.cell, self.id*10+2, 256-self.color, result_second_region_indices, self.voxel_res, self.origin, self.bound, self.z_ref_idx)
            return True, r1, r2

        return False, None, None
    
    def check_and_split_region(self):
        '''
        Split the region into separate regions 
        '''
        x_min_idx = min(self.masked_indices[:,0])
        y_min_idx = min(self.masked_indices[:,1])
        z_min_idx = min(self.masked_indices[:,2])
        # make local 3d space
        x_len = max(self.masked_indices[:,0]) - x_min_idx + 1
        y_len = max(self.masked_indices[:,1]) - y_min_idx + 1
        z_len = max(self.masked_indices[:,2]) - z_min_idx + 1
        # labels_in = np.zeros((x_len, y_len, z_len), dtype=np.int32)
        
        idx_x = self.masked_indices[..., 0]
        idx_y = self.masked_indices[..., 1]
        idx_z = self.masked_indices[..., 2]
        # labels_in = self.bool_inf_data[idx_x, idx_y, idx_z].reshape((x_len,y_len,z_len)).astype(np.int32)
        labels_in = np.zeros((x_len,y_len,z_len)).astype(np.int32)
        labels_in[idx_x-x_min_idx, idx_y-y_min_idx, idx_z-z_min_idx] = self.occupancy_data[idx_x, idx_y, idx_z] == 1
        # labels_in = self.occupancy_data[idx_x, idx_y, idx_z].reshape((x_len,y_len,z_len)).astype(np.int32)
        
        is_split, first_region_indices, second_region_indices = connected_component_3d_labelling(labels_in)
        if is_split:
            first_region_indices += np.min(self.masked_indices, axis=0)
            second_region_indices += np.min(self.masked_indices, axis=0)
            r1 = Region(self.cell, self.id*10+1, self.color, first_region_indices, self.voxel_res, self.origin, self.bound, self.z_ref_idx)
            r2 = Region(self.cell, self.id*10+2, 256-self.color, second_region_indices, self.voxel_res, self.origin, self.bound, self.z_ref_idx)
            return True, r1, r2
        return False, None, None
        # if not is_split:
        #     return []
        # else:
        #     lst_new_region = []
        #     for i in range(num_label):
        #         r = Region(self.cell, self.id + "-" + str(i), masked_image=mask_image)
        #         lst_new_region.append(r)
        #     return lst_new_region
        

    def update_uncertainty_score(self, uncert_score, valid_vxl=None):
        self.uncert_score = uncert_score
        self.valid_vxl = valid_vxl
        
class RegionGraph(Graph):
    def __init__(self, grid:Grid, occupancy_map, split_allowed=False):
        super().__init__()
        
        self.split_allowed = split_allowed
        self.grid = grid
        self.occupancy_map = occupancy_map
        
        self.sub_bound = occupancy_map.Pos2Sub(grid.bound[:,1])
        
        self.cell_number = grid.cell_number
        self.vertices = dict()
        
        
        self.voxel_res = grid.voxel_res
        self.origin = grid.get_origin()
        self.bound = grid.bound
        # self.bound = bound
        
        sub_ref = Pos2Sub(0.5*(self.bound[:,0]+self.bound[:,1]), 1.0/self.voxel_res, self.origin).astype(int)
        self.z_ref_idx = sub_ref[2]
        
        self.rgb_values = np.random.randint(0, 256, size=(2*self.cell_number, 3))

        id = 0
        t0 = time.time()
        for i, cell in enumerate(self.grid.cells):
            region_bound = cell.get_cell_bound()
            pos_lb, pos_rt = region_bound
            
            if pos_rt[0] < self.bound[0][0] or pos_rt[1] < self.bound[1][0] or pos_lb[0] > self.bound[0][1] or pos_lb[1] > self.bound[1][1]:
                continue
            
            pos_lb = np.maximum(pos_lb, self.bound[:,0])
            pos_rt = np.minimum(pos_rt, self.bound[:,1]) + 0.0005
            
            sub_lb = Pos2Sub(pos_lb, 1.0/self.voxel_res, self.origin)
            sub_tr = Pos2Sub(pos_rt, 1.0/self.voxel_res, self.origin)
            # rows, cols, zs = np.meshgrid(np.arange(sub_lb[0], sub_tr[0] + 1), np.arange(sub_lb[1], sub_tr[1] + 1), np.arange(sub_lb[2], sub_tr[2]+1), indexing='ij')
            rows, cols, zs = np.meshgrid(np.arange(sub_lb[0], sub_tr[0]), np.arange(sub_lb[1], sub_tr[1]), np.arange(sub_lb[2], sub_tr[2]), indexing='ij')
            indices = np.array(list(zip(rows.ravel(), cols.ravel(), zs.ravel())))
            r = Region(cell, i, masked_indices=indices, voxel_res=grid.voxel_res, color=self.rgb_values[id,:], origin=self.origin, bound=self.bound, z_ref_idx=self.z_ref_idx)
            self.vertices[i] = r
            # self.vertices.append(r)
            id += 1

        print(f"initializing region takes {(time.time()-t0)*1000} ms")
        self.edges = np.eye(len(self.vertices))
        
        self.dct_connected = dict()
        self.dct_neighbor_idx = dict()
        self.dct_neighbor_path = dict()
        
        
        self.dct_diagonal_connected = dict()
        self.dct_diagonal_idx = dict()
        self.dct_diagonal_path = dict()

        self.compute_neighbor_idx()
        self.compute_diagonal_idx() 
        print()
        
   
    def get_region_id_from_position(self, pos):
        for key in self.vertices.keys():
            if self.vertices[key].is_inside(pos):
                return key
        return None
         
    def get_region_status(self, idx):
        return self.vertices[idx].cell.get_status()
    
    def set_bool_planning_bound(self, planning_bound):
        for key in self.vertices.keys():
            region = self.vertices[key]
            region.is_in_planning_bound = np.logical_and(region.target_pose[:2, 3] >= planning_bound[:2, 0], region.target_pose[:2, 3] <= planning_bound[:2, 1]).all()
    
    def is_in_planning_bound(self, idx):
        return self.vertices[idx].is_in_planning_bound
     
    def get_frontier_score(self, idx):
        return self.vertices[idx].cell.get_frontier_score()
    
    def get_visit_count(self, idx):
        return self.vertices[idx].cell.get_visit_count()
    
    def update_region_graph(self, occupancy_map):
        self.occupancy_map = occupancy_map

        new_regions = []
        new_neighbor_idx = []
        new_neighbor_path = []
        for key in self.vertices.keys():
            if len(self.dct_neighbor_idx[key]) > 0:
                occ_updated = self.vertices[key].update_occupancy_region(occupancy_map)            
                if occ_updated:
                    
                    if self.split_allowed:
                        # is_split, r1, r2 = self.vertices[key].check_and_split_region()
                        
                        is_split, r1, r2 = self.vertices[key].check_and_split_region_2d()
                        if is_split:
                            r1.id = self.vertices[key].id
                            r1.update_occupancy_region(occupancy_map)
                            r2.update_occupancy_region(occupancy_map)
                            # self.vertices[key] = r1
                            # self.dct_neighbor_path[key] = [[]]*len(self.dct_neighbor_idx[key])

                            # Update r1 neighbor
                            lb_r1, tr_r1 = r1.lb_index, r1.tr_index
                            lb_r2, tr_r2 = r2.lb_index, r2.tr_index
                            r1_new_neighbor, r2_new_neighbor = [], []
                            r1_new_path, r2_new_path = [], []
                            for n_id in self.dct_neighbor_idx[key]:
                                lb_idx, tr_idx = self.vertices[n_id].lb_index, self.vertices[n_id].tr_index
                                # Check overlap
                                # if lb_r1[0] <= lb_idx[0] and lb_r1[1] <= lb_idx[1] and tr_r1[0] >= lb_idx[0] and tr_r1[1] >= lb_idx[1]:
                                overlap_num1, overlap_num2 = 0, 0
                                if lb_r1[0] == lb_idx[0]:
                                    overlap_num1 += 1
                                if lb_r1[0] == tr_idx[0]:
                                    overlap_num1 += 1
                                if tr_r1[0] == lb_idx[0]:                                 
                                    overlap_num1 += 1
                                if tr_r1[0] == tr_idx[0]:                                 
                                    overlap_num1 += 1
                                
                                if lb_r1[1] == lb_idx[1]:
                                    overlap_num1 += 1
                                if lb_r1[1] == tr_idx[1]:
                                    overlap_num1 += 1
                                if tr_r1[1] == lb_idx[1]:                                 
                                    overlap_num1 += 1
                                if tr_r1[1] == tr_idx[1]:                                 
                                    overlap_num1 += 1

                                if lb_r2[0] == lb_idx[0]:
                                    overlap_num2 += 1
                                if lb_r2[0] == tr_idx[0]:
                                    overlap_num2 += 1
                                if tr_r2[0] == lb_idx[0]:                                 
                                    overlap_num2 += 1
                                if tr_r2[0] == tr_idx[0]:                                 
                                    overlap_num2 += 1
                                
                                if lb_r2[1] == lb_idx[1]:
                                    overlap_num2 += 1
                                if lb_r2[1] == tr_idx[1]:
                                    overlap_num2 += 1
                                if tr_r2[1] == lb_idx[1]:                                 
                                    overlap_num2 += 1
                                if tr_r2[1] == tr_idx[1]:                                 
                                    overlap_num2 += 1
                                
                                if overlap_num1 >= 2:
                                    r1_new_neighbor.append(n_id)
                                    r1_new_path.append([])
                                if overlap_num2 >= 2:
                                    r2_new_neighbor.append(n_id)
                                    r2_new_path.append([])

                            self.vertices[key] = r1
                            self.dct_neighbor_idx[key] = r1_new_neighbor
                            self.dct_neighbor_path[key] = r1_new_path
                            self.dct_connected[key] = [True]*len(r1_new_neighbor)

                            # self.vertices[max(self.vertices.keys())+1] = r2
                            new_regions.append(r2)
                            new_neighbor_idx.append(r2_new_neighbor)
                            new_neighbor_path.append(r2_new_path)

        if self.split_allowed:
            # Update 
            prev_num_region = len(self.vertices.keys())
            max_key = max(self.vertices.keys())
            for i in range(len(new_regions)):
                self.vertices[max_key+i+1] = new_regions[i]
                self.vertices[max_key+i+1].id = max_key+i+1
                self.dct_neighbor_idx[max_key+i+1] = new_neighbor_idx[i]
                self.dct_neighbor_path[max_key+i+1] = new_neighbor_path[i]
                self.dct_connected[max_key+i+1] = [True]*len(new_neighbor_idx[i])
            
            num_region = len(self.vertices.keys())
            
            print(f"Num region updated from  {prev_num_region} to {num_region}")
        
    def is_connected(self, region1_id, region2_id):
        # Check if the region is in neighbor
        if region2_id in self.dct_neighbor_idx[region1_id]:
            i = self.dct_neighbor_idx[region1_id].index(region2_id)
            return True, len(self.dct_neighbor_path[region1_id][i])
        else:
            dist, lst_id = self.find_region_graph_astar_path(region1_id, region2_id)
            if dist < 0:
                return False, 0
            else:
                # print("a star distance: ", dist)
                return True, dist

    def find_region_graph_astar_path(self, from_node_idx, to_node_idx):
        shortest_dist = 0.0
        cell_number = len(self.vertices.keys())
        g = {k: math.inf for k in self.vertices.keys()}
        f = {k: math.inf for k in self.vertices.keys()}
        prev = {k: -1 for k in self.vertices.keys()}
        in_pg = {k: False for k in self.vertices.keys()}
        # g = [math.inf] * cell_number
        # f = [math.inf] * cell_number
        # prev = [-1] * cell_number
        # in_pg = [False] * cell_number
        g[from_node_idx] = 0
        f[from_node_idx] = np.linalg.norm(self.vertices[from_node_idx].center - self.vertices[to_node_idx].center) * 10.0

        queue = PriorityQueue()
        queue.put(PrioritizedItem(f[from_node_idx], from_node_idx))
        in_pg[from_node_idx] = True
        
        found_path = False
        while not queue.empty():
            u_idx = queue.get().item
            # in_pg[u_idx] = False
            if u_idx == to_node_idx:
                shortest_dist = g[u_idx]
                found_path = True
                break
            for i in range(len(self.dct_neighbor_idx[u_idx])):
                # if not self.lst_connected[4][i]: # 
                d = len(self.dct_neighbor_path[u_idx][i]) # distance as number of voxels for path
                v_idx = self.dct_neighbor_idx[u_idx][i]
                if g[v_idx] > g[u_idx] + d:
                    prev[v_idx] = u_idx
                    g[v_idx] = g[u_idx] + d
                    f[v_idx] = g[v_idx] + np.linalg.norm(self.vertices[v_idx].center - self.vertices[to_node_idx].center) * 10.0
                    if not in_pg[v_idx]:
                        queue.put(PrioritizedItem(f[v_idx], v_idx))
                        in_pg[v_idx] = True
        
        if not found_path:
            return -1, []
        
        node_indices = []
        u_idx = to_node_idx
        if prev[u_idx] != -1 or u_idx == from_node_idx:
            while u_idx != -1:
                node_indices.append(u_idx)
                u_idx = prev[u_idx]
        node_indices.reverse()
        return shortest_dist, node_indices
    
    def update_connectivity(self):
        
        for key in self.vertices.keys():
            region = self.vertices[key]
            lst_neighbor_idx = self.dct_neighbor_idx[key]
            p1 = region.center
            sub_i = self.occupancy_map.Pos2Sub(p1).tolist()
            sub_i[2] = self.z_ref_idx
            
            
            new_lst_neighbor_idx = []
            new_lst_neighbor_path = []
            for i, neighbor_idx in enumerate(lst_neighbor_idx):
                # If raycast between centers is feasible, we do not need to run a-star search
                n_region = self.vertices[neighbor_idx]
                
                raycast_cells = self.dct_neighbor_path[key][i]
                is_raycast_feasible = True
                
                # If current neighbor path is not inside the region's masked indices, then it is not true
                if len(raycast_cells) == 0:
                    is_raycast_feasible = False
                elif np.sum(self.occupancy_map.occupancy_data[raycast_cells[:,0], raycast_cells[:,1], raycast_cells[:,2]] == 1)>0:
                    is_raycast_feasible = False
                else:
                    new_lst_neighbor_idx.append(neighbor_idx)
                    new_lst_neighbor_path.append(raycast_cells)
                
                # if raycast is infeasible, we search for a star path
                if not is_raycast_feasible:
                    # If previous path is still free space, then we do not need to run another a-star search
                    
                    # Check if A-star path can be found
                    a_star_feasible = False
                    
                    sub_lb1 = region.lb_index
                    sub_lb2 = n_region.lb_index
                    sub_tr1 = region.tr_index
                    sub_tr2 = n_region.tr_index
                                        
                    idx_min = np.minimum(sub_lb1, sub_lb2)
                    idx_max = np.maximum(sub_tr1, sub_tr2)
                    idx_min = np.maximum(idx_min, np.array([0,0,self.z_ref_idx-3],dtype=int))
                    idx_max = np.minimum(idx_max, np.array([self.sub_bound[0],
                                                            self.sub_bound[1],
                                                            self.z_ref_idx+3],dtype=int))
                    
                    idx_start = self.occupancy_map.Pos2Sub([region.center[0], region.center[1],0])
                    idx_goal = self.occupancy_map.Pos2Sub([n_region.center[0], n_region.center[1],0])
                    idx_start[2] = self.z_ref_idx
                    idx_goal[2] = self.z_ref_idx
                    occ_idx = np.vstack((region.inf_occ_indices, n_region.inf_occ_indices))
                    

                    astar_path = restricted_astar_search(idx_min, idx_max, idx_start, idx_goal, occ_idx, planner_type='bi-astar', diagonal=False)
                    a_star_feasible = len(astar_path) > 0
                    if a_star_feasible:
                        new_lst_neighbor_idx.append(neighbor_idx)
                        new_lst_neighbor_path.append(copy.deepcopy(astar_path))
            self.dct_neighbor_idx[key] = new_lst_neighbor_idx                  
            self.dct_connected[key] = [True]*len(new_lst_neighbor_idx)
            self.dct_neighbor_path[key] = new_lst_neighbor_path           

    def compute_neighbor_idx(self):
        for i in self.vertices.keys():
            lst_neighbor_idx = []
            lst_connected = []
            
            pos = self.vertices[i].center
            sub = self.grid.Pos2Sub(pos)

            sub_1 = copy.deepcopy(sub)
            sub_1[0] += 1
            if self.grid.in_range(sub_1):
                ind = self.grid.Sub2Ind(sub_1)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(ind)
                    lst_connected.append(True)

            sub_2 = copy.deepcopy(sub)
            sub_2[0] += -1
            if self.grid.in_range(sub_2):
                ind = self.grid.Sub2Ind(sub_2)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(self.grid.Sub2Ind(sub_2))
                    lst_connected.append(True)

            sub_3 = copy.deepcopy(sub)
            sub_3[1] += 1
            if self.grid.in_range(sub_3):
                ind = self.grid.Sub2Ind(sub_3)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(self.grid.Sub2Ind(sub_3))
                    lst_connected.append(True)

            sub_4 = copy.deepcopy(sub)
            sub_4[1] += -1

            if self.grid.in_range(sub_4):
                ind = self.grid.Sub2Ind(sub_4)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(self.grid.Sub2Ind(sub_4))
                    lst_connected.append(True)

            lst_neighbor_path = []
            
            p1 = self.vertices[i].center
            sub_i = self.occupancy_map.Pos2Sub(p1).tolist()
            sub_i[2] = self.z_ref_idx
            for idx in lst_neighbor_idx:
                n_region = self.vertices[idx]
                n_p = n_region.center
                sub_j = self.occupancy_map.Pos2Sub(n_p).tolist()
                sub_j[2] = self.z_ref_idx
                
                raycast_cells = raycast(sub_i, sub_j)
                raycast_cells = np.array(raycast_cells,dtype=int)
                
                lst_neighbor_path.append(raycast_cells)
                
            self.dct_neighbor_idx[i] = lst_neighbor_idx
            self.dct_connected[i] = lst_connected
            self.dct_neighbor_path[i] = lst_neighbor_path
            
    def compute_diagonal_idx(self):
        for i in self.vertices.keys():
            lst_neighbor_idx = []
            lst_connected = []
            
            pos = self.vertices[i].center
            sub = self.grid.Pos2Sub(pos)

            sub_1 = copy.deepcopy(sub)
            sub_1[0] += 1
            sub_1[1] += 1
            if self.grid.in_range(sub_1):
                ind = self.grid.Sub2Ind(sub_1)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(ind)
                    lst_connected.append(True)

            sub_2 = copy.deepcopy(sub)
            sub_2[0] += 1
            sub_2[1] += -1
            if self.grid.in_range(sub_2):
                ind = self.grid.Sub2Ind(sub_2)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(self.grid.Sub2Ind(sub_2))
                    lst_connected.append(True)

            sub_3 = copy.deepcopy(sub)
            sub_3[0] += -1
            sub_3[1] += 1
            if self.grid.in_range(sub_3):
                ind = self.grid.Sub2Ind(sub_3)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(self.grid.Sub2Ind(sub_3))
                    lst_connected.append(True)

            sub_4 = copy.deepcopy(sub)
            sub_4[0] += -1
            sub_4[1] += -1
            if self.grid.in_range(sub_4):
                ind = self.grid.Sub2Ind(sub_4)
                if ind in self.vertices.keys() and ind != i:
                    lst_neighbor_idx.append(self.grid.Sub2Ind(sub_4))
                    lst_connected.append(True)

            lst_neighbor_path = []
            
            p1 = self.vertices[i].center
            sub_i = self.occupancy_map.Pos2Sub(p1).tolist()
            sub_i[2] = self.z_ref_idx
            for idx in lst_neighbor_idx:
                n_region = self.vertices[idx]
                n_p = n_region.center
                sub_j = self.occupancy_map.Pos2Sub(n_p).tolist()
                sub_j[2] = self.z_ref_idx
                
                raycast_cells = raycast(sub_i, sub_j)
                raycast_cells = np.array(raycast_cells,dtype=int)
                
                lst_neighbor_path.append(raycast_cells)
                
            self.dct_diagonal_idx[i] = lst_neighbor_idx
            self.dct_diagonal_connected[i] = lst_connected
            self.dct_diagonal_path[i] = lst_neighbor_path
            

class GridGraph(Graph):
    def __init__(self, grid:Grid):
        super().__init__()
        self.grid = grid
        self.occupancy_map = None
        self.cell_number = grid.cell_number
        self.positions = [None for _ in range(grid.cell_number)]
        for i in range(self.cell_number):
            self.positions[i]  = grid.get_cell_position(i)

        self.lst_connected = [[] for _ in range(self.cell_number)]
        self.distances = [[] for _ in range(self.cell_number)]
        self.neighbor_idx = [[] for _ in range(self.cell_number)]

        self.is_initialized = False
        self.compute_neighbor_idx()

    def compute_neighbor_idx(self):
        for i in range(self.cell_number):
            pos = self.grid.get_cell_position(i)
            sub = self.grid.Pos2Sub(pos)

            sub_1 = copy.deepcopy(sub)
            sub_1[0] += 1
            if self.grid.in_range(sub_1):
                pos_1 = self.grid.Sub2Pos(sub_1)
                self.neighbor_idx[i].append(self.grid.Sub2Ind(sub_1))
                self.distances[i].append(np.linalg.norm(pos_1 - pos))
                self.lst_connected[i].append(False)

            sub_2 = copy.deepcopy(sub)
            sub_2[0] += -1
            if self.grid.in_range(sub_2):
                pos_2 = self.grid.Sub2Pos(sub_2)
                self.neighbor_idx[i].append(self.grid.Sub2Ind(sub_2))
                self.distances[i].append(np.linalg.norm(pos_2 - pos))
                self.lst_connected[i].append(False)

            sub_3 = copy.deepcopy(sub)
            sub_3[1] += 1
            if self.grid.in_range(sub_3):
                pos_3 = self.grid.Sub2Pos(sub_3)
                self.neighbor_idx[i].append(self.grid.Sub2Ind(sub_3))
                self.distances[i].append(np.linalg.norm(pos_3 - pos))
                self.lst_connected[i].append(False)
            sub_4 = copy.deepcopy(sub)
            sub_4[1] += -1

            if self.grid.in_range(sub_4):
                pos_4 = self.grid.Sub2Pos(sub_4)
                self.neighbor_idx[i].append(self.grid.Sub2Ind(sub_4))
                self.distances[i].append(np.linalg.norm(pos_4 - pos))
                self.lst_connected[i].append(False)
             
    def update_graph_connectivity(self, occupancy_map):
        self.occupancy_map = occupancy_map
        if not self.is_initialized:
            for i in range(self.cell_number):
                self.positions[i]  = self.grid.get_cell_position(i)
            self.is_initialized = True

        for i in range(self.cell_number):
            pos_i = self.positions[i]
            start_sub = occupancy_map.Pos2Sub([pos_i[0], pos_i[1], 1.0]).tolist()
            if not occupancy_map.in_range(start_sub):
                continue
            for j in range(len(self.lst_connected[i])):
                # If previously we know that two cells are connected, skip checking

                # if not self.lst_connected[i][j]:
                #     continue
                pos_j = self.positions[self.neighbor_idx[i][j]]
                end_sub = occupancy_map.Pos2Sub([pos_j[0], pos_j[1], 1.0]).tolist()
                
                if not occupancy_map.in_range(end_sub):
                    continue
                
                connected = True
                raycast_cells = raycast(start_sub, end_sub)
                if len(raycast_cells) > 1:
                    # not_connected = True
                    # Check occlusion in ray
                    num_occ = 0
                    for k in range(1, len(raycast_cells)):
                        if occupancy_map.is_occupied(np.array(raycast_cells[k]), include_unknown=True, include_collision_threshold=False):
                            num_occ += 1
                    if num_occ > 0:
                        connected = False

                self.lst_connected[i][j] = connected
                # self.lst_connected[self.neighbor_idx[i][j]][i] = connected
                self.distances[i][j] = np.linalg.norm(pos_i-pos_j)

    def update_graph_connectivity_vectorized(self, occupancy_map):
        t0 = time.time()
        self.occupancy_map = occupancy_map
        
        start_subs = np.stack([occupancy_map.Pos2Sub([pos_i[0], pos_i[1], 1.0]) for pos_i in self.positions])
        in_range_start = np.all(np.concatenate([start_subs[:, :occupancy_map.dimension] >= 0,
                                                start_subs[:, :occupancy_map.dimension] < occupancy_map.size[:occupancy_map.dimension]], axis=-1), axis=-1)
        in_range_start = np.tile(in_range_start[:, None], (1, 4)).reshape(-1)

        start_grid_subs = np.stack([self.grid.Pos2Sub([pos_i[0], pos_i[1], 1.0]) for pos_i in self.positions])
        end_grid_subs = start_grid_subs[:, None] + np.array([[[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]]) # [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0] is same order as lst_connected is defined
        sh = end_grid_subs.shape
        end_grid_subs_flat = end_grid_subs.reshape(-1, 3)
        end_pos_flat = np.stack([self.grid.Sub2Pos(end_sub) for end_sub in end_grid_subs_flat])
        end_subs_flat = np.stack([occupancy_map.Pos2Sub([pos_i[0], pos_i[1], 1.0]) for pos_i in end_pos_flat])
        in_range_end = np.all(np.concatenate([end_subs_flat[:, :occupancy_map.dimension] >= 0,
                                              end_subs_flat[:, :occupancy_map.dimension] < occupancy_map.size[:occupancy_map.dimension]], axis=-1), axis=-1)
        in_range_end_grid = np.all(np.concatenate([end_grid_subs_flat[:, :self.grid.dimension] >= 0,
                                                   end_grid_subs_flat[:, :self.grid.dimension] < self.grid.size[:self.grid.dimension]], axis=-1), axis=-1)
        in_range_end_grid = in_range_end_grid.reshape(sh[:-1])
        
        in_range = np.logical_and(in_range_start, in_range_end)
        start_subs_flat = np.tile(start_subs[:, None], (1, 4, 1)).reshape(-1, 3)
        start_subs_flat = start_subs_flat[in_range]
        end_subs_flat = end_subs_flat[in_range]
        
        is_connected = np.zeros_like(in_range)
        _is_connected = np.logical_not(check_occupancy(start_subs_flat, end_subs_flat, occupancy_map, 
                                       include_unknown=True, include_collision_threshold=False))
        is_connected[in_range] = _is_connected
        is_connected = is_connected.reshape(sh[:-1]) # (n_positions, 4)
        is_connected = is_connected.astype(np.bool_)
        t1 = time.time()
        print(f"update_graph_connectivity_vectorized before allocating takes {(t1-t0)*1000} ms")
        
        distances = copy.deepcopy(self.distances)
        lst_connected = copy.deepcopy(self.lst_connected)
        for i in range(self.cell_number):
            idx_connect = 0
            for j in range(4):
                if in_range_end_grid[i, j]:
                    lst_connected[i][idx_connect] = is_connected[i, j]
                    
                    # isn't it duplicate?
                    pos_i = self.positions[i]
                    pos_j = self.positions[self.neighbor_idx[i][idx_connect]]
                    distances[i][idx_connect] = np.linalg.norm(pos_i-pos_j)
                    
                    idx_connect += 1
        
        return distances, lst_connected
        
    def is_connected(self, idx1, idx2, pos1, pos2, z_ref)->bool:
        try:
            i = self.neighbor_idx[idx1].index(idx2)
            return self.lst_connected[idx1][i]
        except:
            # Check los
            start_sub = self.occupancy_map.Pos2Sub([pos1[0], pos1[1], z_ref]).tolist()
            end_sub = self.occupancy_map.Pos2Sub([pos2[0], pos2[1], z_ref]).tolist()

            occlude = False
            raycast_cells = raycast(start_sub, end_sub)
            if len(raycast_cells) > 1:
                occlude = False
                # Check occlusion in ray
                for k in range(1, len(raycast_cells)):
                    if self.occupancy_map.is_occupied(raycast_cells[k], include_unknown=False):
                        occlude = True
                        break                        
            if occlude:
                return False
            return True
    
    def node_index_in_range(self, node_idx):
        if node_idx >=0 and node_idx < self.cell_number:
            return True
        return False
    
    def astar_search(self,
                   from_node_idx:int,
                   to_node_idx:int):
        assert self.node_index_in_range(from_node_idx)
        assert self.node_index_in_range(to_node_idx)
        
        shortest_dist = 0.0
        g = [math.inf] * len(self.lst_connected)
        f = [math.inf] * len(self.lst_connected)
        prev = [-1] * len(self.lst_connected)
        in_pg = [False] * len(self.lst_connected)
        g[from_node_idx] = 0
        f[from_node_idx] = np.linalg.norm(self.positions[from_node_idx] - self.positions[to_node_idx])

        queue = PriorityQueue()
        queue.put(PrioritizedItem(f[from_node_idx], from_node_idx))
        in_pg[from_node_idx] = True
        
        found_path = False
        while not queue.empty():
            u_idx = queue.get().item
            # in_pg[u_idx] = False
            if u_idx == to_node_idx:
                shortest_dist = g[u_idx]
                found_path = True
                break
                        
            for i in range(len(self.lst_connected[u_idx])):
                if not self.lst_connected[u_idx][i]:
                    continue
                d = self.distances[u_idx][i]
                v_idx = self.neighbor_idx[u_idx][i]
                if g[v_idx] > g[u_idx] + d:
                    prev[v_idx] = u_idx
                    g[v_idx] = g[u_idx] + d
                    f[v_idx] = g[v_idx] + np.linalg.norm(self.positions[v_idx] - self.positions[to_node_idx])
                    if not in_pg[v_idx]:
                        queue.put(PrioritizedItem(f[v_idx], v_idx))
                        in_pg[v_idx] = True
        
        if not found_path:
            return -1, []
        
        node_indices = []
        u_idx = to_node_idx
        if prev[u_idx] != -1 or u_idx == from_node_idx:
            while u_idx != -1:
                node_indices.append(u_idx)
                u_idx = prev[u_idx]
        node_indices.reverse()
        return shortest_dist, node_indices
    