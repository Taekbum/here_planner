import time
import math
import copy
import numpy as np
from enum import Enum
from multipledispatch import dispatch
from ..utils.utils import raycast
from sklearn.cluster import DBSCAN

class CellStatus(Enum):
    UNSEEN = 0
    EXPLORING = 1
    COVERED = 2
    NOGO = 3

class Cell(object):
    def __init__(self, center, voxel_size, cell_bound):
        self.center = np.array(center)
        self.voxel_size = voxel_size

        self.cell_bound = cell_bound
        
        self.robot_position_set = False
        self.visit_count = 0
        self.keypose_id = 0
        self.robot_position = np.zeros((3,))
        self.status = CellStatus.UNSEEN
        self.viewpoint_indices = []
        self.connected_cell_indices = []
        self.roadmap_connection_point_set = False
        self.keypose_graph_node_indices = []

        self.frontiers = []

    def get_frontier_score(self):
        # print()
        # print("Frontier Score: ", len(self.frontiers))
        return len(self.frontiers)
    
    def clear_frontier(self):
        self.frontiers.clear()
                
    def add_viewpoint(self, viewpoint_ind):
        self.viewpoint_indices.append(viewpoint_ind)
    
    def add_frontier(self, frontier_ind):
        # print("[ADD FRONTIER]")
        self.frontiers.append(frontier_ind)
        
    def clear_viewpoint_indices(self):
        self.viewpoint_indices = []
                
    def get_status(self):
        return self.status
    
    def set_status(self, status:CellStatus):
        self.status = status
    
    def get_viewpoint_indices(self):
        return self.viewpoint_indices
        
    def get_graph_node_indices(self):
        return self.keypose_graph_node_indices
    
    def get_position(self):
        return self.center
    
    def set_position(self, position):
        self.center = position
    
    def get_cell_bound(self):
        return self.cell_bound
    
    def set_robot_position(self, robot_pos):
        self.robot_position = robot_pos
        self.robot_position_set = True
            
    def add_visit_count(self):
        self.visit_count += 1
    
    def get_visit_count(self):
        return self.visit_count
                
    def reset(self):
        self.status = CellStatus.UNSEEN
        self.robot_position = np.zeros((3,))
        self.visit_count = 0
        self.viewpoint_indices = []
        self.connected_cell_indices = []
        self.keypose_graph_node_indices = []
    
    
    
class Grid(object):
    def __init__(self, size, bound, voxel_res=0.1, origin=np.zeros((3,)), resolution=np.ones((3,)), dimension=2, build_cell_graph=False):
        self.origin = origin
        self.size = size
        self.resolution = resolution
        self.dimension = dimension
        self.bound = bound
        self.voxel_res = voxel_res
        
        self.inv_resolution = np.zeros_like(self.resolution)
        for i in range(self.dimension):
            self.inv_resolution[i] = 1.0 / self.resolution[i]
        # self.cell_number = size[0] * size[1] * size[2]
        
        self.cells = []
        self.subs = []
        
        for i in range(size[0] * size[1] * size[2]):
            center = self.Ind2Pos(i)
            pos_lb = (self.Ind2Pos(i) - 0.5 * self.resolution).astype(np.float32) 
            pos_rt = (self.Ind2Pos(i) + 0.5 * self.resolution).astype(np.float32)
            if type(bound) != float: # we only want this case for subspaces
                pos_lb[2] = bound[2][0]
                pos_rt[2] = bound[2][1]
                
                
                pos_rt[0] = min(pos_rt[0], bound[0][1])
                pos_rt[1] = min(pos_rt[1], bound[1][1])

            pos_lb[0] = max(pos_lb[0], origin[0])
            pos_lb[1] = max(pos_lb[1], origin[1])
            pos_lb[2] = max(pos_lb[2], origin[2])
            
            
            # if pos_rt[0] < bound[0][0] or pos_rt[1] < bound[1][0] or pos_lb[0] > bound[0][1] or pos_lb[1] > bound[1][1]:
            #     continue
            cell = Cell(center, voxel_res, (pos_lb, pos_rt))
            self.cells.append(cell)
            self.subs.append(self.ind2sub(i))
        self.cell_number = len(self.cells)
            
    
    def get_cell_number(self):
        return self.cell_number
    
    @dispatch(int)
    def get_cell_value(self, ind):
        return self.cells[ind]
    
    @dispatch(int, int, int)
    def get_cell_value(self, x, y, z):
        index = self.Sub2Ind(x, y, z)
        return self.cells[index]
        
    @dispatch(int)
    def get_cell(self, idx)->Cell:
        return self.cells[idx]

    @dispatch(int,int,int)    
    def get_cell(self, i, j, k)->Cell:
        index = self.Sub2Ind(np.array([i,j,k]))
        return self.get_cell(index)
        
    def get_size(self):
        return self.size
    
    def get_origin(self):
        return self.origin
    
    @dispatch(list)
    def in_range(self, sub:list)->bool:
        for i in range(self.dimension):
            if not (sub[i] >=0 and sub[i] < self.size[i]):
                return False
        return True

    @dispatch(np.ndarray)
    def in_range(self, sub:np.ndarray)->bool:
        for i in range(self.dimension):
            if not (sub[i] >=0 and sub[i] < self.size[i]):
                return False
        return True

    @dispatch(int)
    def in_range(self, ind):
        return ind>=0 and ind < self.cell_number
    
    def Ind2Sub(self, ind:int)->np.ndarray:
        return self.subs[ind]

    def Pos2Sub(self, pos):
        sub = np.array([0, 0, 0])
        for i in range(self.dimension):
            if pos[i] > self.origin[i]:
                sub[i] = int((pos[i] - self.origin[i]) * self.inv_resolution[i])
            else:
                sub[i] = -1
        return sub

    def Pos2Ind(self, pos):
        sub = self.Pos2Sub(pos)
        return self.Sub2Ind(sub)

    @dispatch(int,int,int)
    def Sub2Ind(self, x, y, z)->int:
        ind = (x + (y * self.size[0]) + (z * self.size[0] * self.size[1]))
        if type(ind) == int:
            return ind
        else:
            return ind.item()

    @dispatch(list)    
    def Sub2Ind(self, sub):
        if type(sub[0]) == int:
            ind = self.Sub2Ind(sub[0], sub[1], sub[2])
        else:
            ind = self.Sub2Ind(sub[0].item(), sub[1].item(), sub[2].item())
        if type(ind) == int:
            return ind
        else:
            return ind.item()

    @dispatch(np.ndarray)    
    def Sub2Ind(self, sub):
        ind = self.Sub2Ind(sub[0].item(), sub[1].item(), sub[2].item())
        if type(ind) == int:
            return ind
        else:
            return ind.item()

    @dispatch(np.ndarray)
    def Sub2Pos(self, sub):
        pos = np.zeros((3,))
        for i in range(self.dimension):
            pos[i] = self.origin[i] + sub[i] * self.resolution[i] + self.resolution[i] * 0.5
        return pos

    @dispatch(int,int,int)
    def Sub2Pos(self, x, y, z):
        return self.Sub2Pos(np.array([x,y,z]))

    def set_cell_with_ind(self, ind, cell):
        self.cells[ind] = cell
        
    def set_cell(self, i, j, k, cell):
        ind = self.Sub2Ind(i,j,k)
        self.cells[ind] = cell
    # def set_cell(self, ind, cell):
    #     self.cells[ind] = cell
    
    def ind2sub(self, ind:int):
        sub = np.zeros((3,),dtype=np.int64)
        sub[2] = ind / (self.size[0] * self.size[1])
        ind += -1.0 * (sub[2] * self.size[0] * self.size[1])
        sub[1] = ind / self.size[0]
        sub[0] = ind % self.size[0]
        return sub
    
    def Ind2Pos(self, ind):
        return self.Sub2Pos(self.ind2sub(ind))
    
    def get_cell_position(self, ind):
        return self.cells[ind].center
                
class OccupancyGrid(Grid):
    def __init__(self, res, bound=None, size=None, origin=None, dimension=3, uncertainty_threshold=4.0, collision_threshold=0.5):
        init_value = 0.0
        resolution = np.ones((3,), dtype=np.float32) * res 

        self.unknown = 0
        self.free = -1
        self.occupied = 1
        self.frontier_positions = []
        self.frontier_indices = []
        self.uncertainty_threshold = uncertainty_threshold
        self.collision_threshold = collision_threshold
        
        if bound is not None and size is None:
            x_len = bound[0][1] - bound[0][0]
            y_len = bound[1][1] - bound[1][0]
            z_len = bound[2][1] - bound[2][0]
            size = np.array([math.floor(x_len / res), math.floor(y_len/res),math.floor(z_len/res)])
            origin = np.array([bound[0][0], bound[1][0], bound[2][0]])
            super().__init__(size, init_value, origin=origin, resolution=resolution, dimension=dimension)
        else:
            super().__init__(size, init_value, origin=origin, resolution=resolution, dimension=dimension)
    
    def set_boolean_map_data(self, data):
        self.data = data
        # Create a grid object from the numpy array
        # self.grid = Grid(matrix=self.data, inverse=True)
    def set_boolean_inflated_map_data(self, data):
        self.inf_data = data
        
    def set_raw_data(self, data):
        self.raw_data = data
    
    def set_occupancy_data(self, data):
        self.occupancy_data = data
        
    def set_uncertainty_data(self, u_data):
        self.uncertainty_data = u_data
    
    @dispatch(list)      
    def is_occupied(self, sub, include_unknown=False):
        if include_unknown:
            return (self.occupancy_data[sub[0]][sub[1]][sub[2]] == self.occupied)
        else:
            return self.data[sub[0]][sub[1]][sub[2]] == self.occupied

    @dispatch(np.ndarray)
    def is_occupied(self, sub, include_unknown=False, include_collision_threshold=True):
        if include_unknown:
            if include_collision_threshold:
                return self.occupancy_data[sub[0]][sub[1]][sub[2]] == self.occupied
            else:
                if self.raw_data[sub[0]][sub[1]][sub[2]] < 0.00 and self.uncertainty_data[sub[0]][sub[1]][sub[2]] < 3.0:
                    return True
                else:
                    return False                    
        else:
            return self.data[sub[0]][sub[1]][sub[2]] == self.occupied
        
    def is_feasible_point(self, x, y, z):
        sub = self.Pos2Sub([x,y,z])
        in_range = self.in_range(sub)
        if not in_range:
            return False
        is_occ = self.is_occupied(sub)
        return not is_occ
    
    def is_line_feasible(self, pt1, pt2):
        sub1 = self.Pos2Sub(pt1)
        sub2 = self.Pos2Sub(pt2)
        in_range = self.in_range(sub2)
        if not in_range:
            return False
        raycast_cells = raycast(sub1.tolist(), sub2.tolist())
        feasible = True

        if len(raycast_cells) > 1:
            # Check occlusion in ray
            for k in range(1, len(raycast_cells)):
                if self.is_occupied(raycast_cells[k], include_unknown=True):
                # viewpoint_ind = view_pt_manager.collision_grid.Sub2Ind(np.array(raycast_cells[i]))
                    feasible = False
                    break
        return feasible
    
    def cluster_frontier_voxels(self):
        self.frontier_clusters = []
        for frontier_with_single_z in self.frontier_with_z_positions:
            if len(frontier_with_single_z) == 0:
                continue
            np_val = np.array(frontier_with_single_z)
            clustering = DBSCAN(eps=0.2, min_samples=1).fit(np_val)
            
            num_cluster = np.max(clustering.labels_)+1
            cluster_lst = [np.zeros(2).copy() for _ in range(num_cluster)]
            cluster_num = [0] * num_cluster
            for i, coord in enumerate(frontier_with_single_z):
                id = clustering.labels_[i]
                if id == -1:
                    break
                cluster_lst[id] += copy.deepcopy(coord[:2])
                cluster_num[id] += 1

            for id in range(num_cluster):
                if cluster_num[id] > 0:
                    cluster_lst[id] /= cluster_num[id]
                    self.frontier_clusters.append(np.array([cluster_lst[id][0],cluster_lst[id][1],coord[2]]))

        return self.frontier_clusters
    
    def extract_frontier_voxel_vectorized(self, position, z_list): # 
        print("Inside Frontier voxel")
        
        self.frontier_with_z_positions = []
        
        z_subs = np.array([self.Pos2Sub([position[0],position[1],z])[2] for z in z_list])
        cur_subs = np.stack(np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), z_subs, indexing='ij'), -1).reshape(-1, 3)
        neigh_subs = cur_subs[:, None] + np.array([[[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]])
        neigh_subs_flat = neigh_subs.reshape(-1, 3)
        sh_neigh = neigh_subs.shape
        
        neigh_occ_flat = np.zeros(neigh_subs_flat.shape[0])
        in_range = np.all(np.concatenate([neigh_subs_flat[:, :self.dimension] >= 0, 
                                          neigh_subs_flat[:, :self.dimension] < self.size[:self.dimension]], axis=-1), axis=-1)
        neigh_subs_in_range = neigh_subs_flat[in_range]
        neigh_occ_flat[in_range] = self.occupancy_data[neigh_subs_in_range[:, 0], neigh_subs_in_range[:, 1], neigh_subs_in_range[:, 2]]
        neigh_occ_flat[~in_range] = self.occupied
        
        neigh_occ = neigh_occ_flat.reshape(sh_neigh[:-1])
        curr_occ = self.occupancy_data[cur_subs[:, 0], cur_subs[:, 1], cur_subs[:, 2]]
        
        curr_unknown = curr_occ == self.unknown
        neigh_free_any = np.any(neigh_occ == self.free, axis=-1)
        is_frontier = np.logical_and(curr_unknown, neigh_free_any)
        self.frontier_indices = cur_subs[is_frontier]
        self.frontier_positions = self.origin + self.frontier_indices * self.resolution + self.resolution * 0.5
        
        for z_sub in z_subs:
            z_mask = self.frontier_indices[:, 2] == z_sub
            frontier_with_single_z = self.frontier_positions[z_mask]
            self.frontier_with_z_positions.append(list(frontier_with_single_z))
        
        self.frontier_indices = list(self.frontier_indices)
        self.frontier_positions = list(self.frontier_positions)
        return self.frontier_positions
    
    def check_los_free(self, start_sub, end_sub, include_unknown=False):
        occlude = False
        if start_sub[0] == end_sub[0] and start_sub[1] == end_sub[1]:
            return True
        raycast_cells = raycast(start_sub, end_sub)
        if len(raycast_cells) > 1:
            occlude = False
            # Check occlusion in ray
            for k in range(1, len(raycast_cells)):
                if self.is_occupied(np.array(raycast_cells[k]), include_unknown=include_unknown, include_collision_threshold=False):
                # viewpoint_ind = view_pt_manager.collision_grid.Sub2Ind(np.array(raycast_cells[i]))
                    occlude = True
                    break
        if not occlude:
            return True
        return False     
    
    def get_uncertainty_image(self, uncertain_threshold, z_ref_idx):
        u_array = self.uncertainty_data[:,:,z_ref_idx]
        u_array = np.array(u_array) / uncertain_threshold
        u_array = 1 - np.minimum(u_array, 1)
        u_array = np.stack([(u_array * 255).astype(np.uint8)] * 3, axis=-1)
        return u_array