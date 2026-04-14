import math
import copy
import numpy as np
from .grid import Grid, CellStatus
from .viewpoint_manager import ViewPointManager
from .grid_graph import GridGraph, RegionGraph
from .python_tsp.heuristics.lin_kernighan import solve_tsp_lin_kernighan
from .python_tsp.heuristics.local_search import solve_tsp_local_search
from .python_tsp.heuristics.lin_kernighan import _solve_tsp_brute_force
from .python_tsp.exact.brute_force import solve_tsp_brute_force
from .astar import restricted_astar_search
import time


class GridWorld(object):
    def __init__(self, row_num=1, 
            col_num=1, 
            level_num=1, 
            origin=None,
            bound=None,
            voxel_size=0.1,
            cell_size=6.0, 
            cell_height=6.0, 
            nearby_grid_num=5,
            exploring_frontier_th=10,
            cell_exploring_to_covered_th=1,
            cell_covered_to_exploring_th=10,
            cell_exploring_to_almost_covered_th=10,
            cell_almost_covered_to_exploring_th=20,
            cell_unknown_to_exploring_th=1            
            ):
        self.robot_position = np.zeros((3,))
        self.origin = origin
        self.row_num = row_num
        self.col_num = col_num
        self.level_num = level_num
        self.cell_size = cell_size
        self.cell_height = cell_height     
        self.nearby_grid_num = nearby_grid_num   
        self.exploring_frontier_th = exploring_frontier_th
        grid_size = np.array([self.row_num, self.col_num, self.level_num])
        # grid_origin = np.array([origin[0], origin[1], 1.0])
        grid_origin = origin
        grid_resolution = np.array([self.cell_size, self.cell_size, self.cell_height])
        
        s_time = time.time()
        self.subspaces = Grid(size=grid_size, bound=bound, voxel_res=voxel_size, origin=grid_origin, resolution=grid_resolution, dimension=2)

        

        print("subspace gen time: ", time.time() - s_time)
        self.home_position = np.zeros((3,))
        self.cur_keypose_graph_node_position = np.zeros((3,))
        self.set_home = False
        self.return_home = False
        
        self.neighbor_cell_indices = []
        self.almost_covered_cell_indices = []

        self.cell_exploring_to_covered_th = cell_exploring_to_covered_th
        self.cell_covered_to_exploring_th = cell_covered_to_exploring_th
        self.cell_exploring_to_almost_covered_th = cell_exploring_to_almost_covered_th        
        self.cell_almost_covered_to_exploring_th = cell_almost_covered_to_exploring_th
        self.cell_unknown_to_exploring_th = cell_unknown_to_exploring_th
        
        self.initialized = False
        
        self.prev_robot_cell_ind = -1
        self.cur_robot_cell_ind = -1
        self.cur_keypose_graph_node_ind = 0
        
        self.use_keypose_graph = False
        
        self.min_add_point_num_small = 8
        self.min_add_point_num_big = 16
        self.min_add_frontier_point_num = 8
        
        self.frontiers = None
    
    def set_frontiers(self, frontiers):
        # Before get new frontiers
        # First, reset al frontiers
        for cell_ind in range(self.subspaces.cell_number):
            self.clear_frontiers_to_cell(cell_ind)
        self.frontiers = frontiers
        
    def home_set(self):
        return self.set_home
    
    def set_home_position(self, home_position):
        self.set_home = True
        self.home_position = home_position
    
    def is_initialized(self):
        return self.initialized
    
    def ind_in_bound(self, ind):
        return self.subspaces.in_range(ind)
    
    def get_cell_ind(self, position):
        sub = self.subspaces.Pos2Sub(position)
        if self.subspaces.in_range(sub):
            return self.subspaces.Sub2Ind(sub)
        else:
            return -1
    def set_env_bound(self, bound):
        self.bound = bound
        self.subspace_graph = GridGraph(self.subspaces)
        
    def set_region_graph(self, occupancy_map):
        self.region_graph = RegionGraph(self.subspaces, occupancy_map)

    def update_region_graph(self, occupancy_map):
        self.region_graph.update_region_graph(occupancy_map)
        self.region_graph.update_connectivity()
        
    def update_robot_position(self, robot_position):
        self.robot_position = robot_position
        robot_cell_ind = self.get_cell_ind(robot_position)
        if not self.cur_robot_cell_ind == robot_cell_ind:
            self.prev_robot_cell_ind = robot_cell_ind
            self.cur_robot_cell_ind = robot_cell_ind
    
    def update_neighbor_cells(self, robot_position):
        if not self.initialized:
            self.initialized = True

            # Update cell centers
            for i in range(self.row_num):
                for j in range(self.col_num):
                    for k in range(self.level_num):
                        subspace_center_pos = self.subspaces.Sub2Pos(i, j, k)
                        subspace_center_pos[2] = 1.0
                        self.subspaces.get_cell(i,j,k).set_position(subspace_center_pos)
                        # self.subspaces.get_cell(i,j,k).SetRoadmapConnectionPoint(subspace_center_pos)
        
        # Get neighbor cells
        prev_neighbor_cell_indices = copy.deepcopy(self.neighbor_cell_indices)
        self.neighbor_cell_indices = []
        N = math.floor(self.nearby_grid_num / 2)
        M = 0
        self.get_neighbor_cell_indices(robot_position, np.array([N,N,M]))
        
        for cell_ind in self.neighbor_cell_indices:
            if not cell_ind in prev_neighbor_cell_indices:
                self.subspaces.get_cell(cell_ind).add_visit_count()

    def get_cell_sub(self, position):
        return self.subspaces.Pos2Sub(position)
        
    def get_neighbor_cell_indices(self, robot_position:np.ndarray, neighbor_range:np.ndarray):
        center_cell_sub = self.get_cell_sub(robot_position)
        row_idx = 0
        col_idx = 0
        level_idx = 0
        for i in range(-neighbor_range[0], neighbor_range[0]+1):
            for j in range(-neighbor_range[1], neighbor_range[1]+1):
                row_idx = center_cell_sub[0] + i
                col_idx = center_cell_sub[1] + j
                for k in range(-neighbor_range[2], neighbor_range[2]+1):
                    level_idx = center_cell_sub[2] + k
                    sub = np.array([row_idx, col_idx, level_idx])
                    if self.subspaces.in_range(sub):
                        ind = self.subspaces.Sub2Ind(sub)
                        self.neighbor_cell_indices.append(ind)

    def get_cell_status(self, cell_ind:int):
        assert self.subspaces.in_range(cell_ind)
        return self.subspaces.get_cell(cell_ind).get_status()
        
    def update_cell_status(self, viewpoint_manager:ViewPointManager, occupancy_map):
        exploring_count = 0
        unseen_count = 0
        covered_count = 0
        for i in range(self.subspaces.get_cell_number()):
            if self.subspaces.get_cell(i).get_status() == CellStatus.EXPLORING:
                exploring_count += 1
            elif self.subspaces.get_cell(i).get_status() == CellStatus.UNSEEN:
                unseen_count += 1
            elif self.subspaces.get_cell(i).get_status() == CellStatus.COVERED:
                covered_count += 1
        
        for cell_ind in self.neighbor_cell_indices:
            self.subspaces.get_cell(cell_ind).clear_viewpoint_indices()
        
        for viewpoint_ind in viewpoint_manager.get_candidate_indices():
            viewpoint_position = viewpoint_manager.get_viewpoint_position(viewpoint_ind)
            sub = self.subspaces.Pos2Sub(viewpoint_position)
            if self.subspaces.in_range(sub):
                cell_ind = self.subspaces.Sub2Ind(sub)
                self.add_viewpoint_to_cell(cell_ind, viewpoint_ind)
                viewpoint_manager.set_viewpoint_cell_ind(viewpoint_ind, cell_ind)
            else:
                print(f"[ERROR] Grid Sub Out of Bound:  {sub[0]}, {sub[1]}, {sub[2]}")
        
        if self.frontiers is not None:
            # Update Frontiers
            for frontier_pos in self.frontiers:
                sub_occ = occupancy_map.Pos2Sub(frontier_pos) # 
                sub = self.subspaces.Pos2Sub(frontier_pos)
                cell_ind = self.subspaces.Sub2Ind(sub)
                self.add_frontier_to_cell(cell_ind, sub_occ) # 
            
        
        print(f"Exploring {exploring_count} cells, Unseen {unseen_count} cells, Covered {covered_count} cells")

        print(f"robot pos: {self.robot_position}")
        for cell_ind in range(self.subspaces.cell_number):
            candidate_count = 0
            selected_viewpoint_count = 0
            above_small_threshold_count = 0
            above_big_threshold_count = 0
            above_frontier_threshold_count = 0
            highest_score = -1
            for viewpoint_ind in self.subspaces.get_cell(cell_ind).get_viewpoint_indices():
                candidate_count += 1
                if viewpoint_manager.viewpoint_selected(viewpoint_ind):
                    selected_viewpoint_count += 1
                if viewpoint_manager.viewpoint_visited(viewpoint_ind):
                    continue
                score = viewpoint_manager.get_viewpoint_covered_point_num(viewpoint_ind)
                frontier_score = viewpoint_manager.get_viewpoint_covered_frontier_point_num(viewpoint_ind)
                if score > highest_score:
                    highest_score = score
                if score > self.min_add_point_num_small:
                    above_small_threshold_count += 1
                if score > self.min_add_point_num_big:
                    above_big_threshold_count += 1
                if frontier_score > self.min_add_frontier_point_num:
                    above_frontier_threshold_count += 1
            if self.subspaces.get_cell(cell_ind).get_status() == CellStatus.EXPLORING:
                center = self.subspaces.get_cell(cell_ind).center
                sub = occupancy_map.Pos2Sub(center)
                if (self.subspaces.get_cell(cell_ind).get_frontier_score() < self.exploring_frontier_th): 
                    self.subspaces.get_cell(cell_ind).set_status(CellStatus.COVERED)
                    print(f"cell ind {cell_ind}: EXPLORING To COVERED")
            # Covered to Exploring
            elif self.subspaces.get_cell(cell_ind).get_status() == CellStatus.COVERED \
                and (self.subspaces.get_cell(cell_ind).get_frontier_score() > self.exploring_frontier_th):
                self.subspaces.get_cell(cell_ind).set_status(CellStatus.EXPLORING)
                self.almost_covered_cell_indices.append(cell_ind)
                print(f"cell ind {cell_ind}: COVERED To EXPLORING")
            # Exploring to Almost covered
            elif self.subspaces.get_cell(cell_ind).get_status() != CellStatus.COVERED and selected_viewpoint_count > 0:
                # if path went to this cell (selected viewpoint)
                self.subspaces.get_cell(cell_ind).set_status(CellStatus.EXPLORING)
                try:
                    cur_idx = self.almost_covered_cell_indices.index(cell_ind)
                    self.almost_covered_cell_indices.pop(cur_idx)
                except:
                    pass
                print(f"cell ind {cell_ind}: To EXPLORING")
            elif self.subspaces.get_cell(cell_ind).get_status() != CellStatus.COVERED and self.subspaces.get_cell(cell_ind).get_frontier_score() > 0:
                self.subspaces.get_cell(cell_ind).set_status(CellStatus.EXPLORING)
                print(f"cell ind {cell_ind}: To EXPLORING::Frontier Exists")
            elif self.subspaces.get_cell(cell_ind).get_status() == CellStatus.EXPLORING and candidate_count == 0:
                # first visit
                if self.subspaces.get_cell(cell_ind).get_visit_count() == 1 and len(self.subspaces.get_cell(cell_ind).get_graph_node_indices())==0:
                    self.subspaces.get_cell(cell_ind).set_status(CellStatus.COVERED)
                    print(f"cell ind {cell_ind}: To COVERED")
                else:
                    cell_position = self.subspaces.get_cell(cell_ind).get_position()
                    xy_dist_to_robot = np.linalg.norm(cell_position - self.robot_position)
                    z_dist_to_robot = np.abs(cell_position[2] - self.robot_position[2])
                    if xy_dist_to_robot < self.cell_size and z_dist_to_robot < self.cell_height * 0.8:
                        self.subspaces.get_cell(cell_ind).set_status(CellStatus.COVERED)
                        print(f"cell ind {cell_ind}: To COVERED")
            if self.subspaces.get_cell(cell_ind).get_status() == CellStatus.EXPLORING and candidate_count > 0:
                self.subspaces.get_cell(cell_ind).set_robot_position(self.robot_position)
                    
    def clear_frontiers_to_cell(self, cell_ind):
        self.subspaces.get_cell(cell_ind).clear_frontier()
        
    def add_frontier_to_cell(self, cell_ind:int, frontier_ind):
        assert self.subspaces.in_range(cell_ind)
        self.subspaces.get_cell(cell_ind).add_frontier(frontier_ind)
        
    def add_viewpoint_to_cell(self, cell_ind:int, viewpoint_ind:int):
        assert self.subspaces.in_range(cell_ind)
        self.subspaces.get_cell(cell_ind).add_viewpoint(viewpoint_ind)

    def astar_planning_region_mask(self, start_p, goal_p, region_key_lst, occupancy_map):
        goal_ind = occupancy_map.Pos2Sub(goal_p)
        start_ind = occupancy_map.Pos2Sub(start_p)
        
        lst_index_lb = [self.region_graph.vertices[id].lb_index for id in region_key_lst]
        lst_index_tr = [self.region_graph.vertices[id].tr_index for id in region_key_lst]
        
        idx_min = np.asarray(lst_index_lb).min(0)
        idx_max = np.asarray(lst_index_tr).max(0)

        idx_min = np.maximum(idx_min, np.array([0,0,0],dtype=int))
        idx_max = np.minimum(idx_max, np.array([self.region_graph.sub_bound[0],
                                                self.region_graph.sub_bound[1],
                                                self.region_graph.sub_bound[2]],dtype=int))

        lst_free_idx = []
        lst_occ_idx = []
        for key in region_key_lst:
            unocc = self.region_graph.vertices[key].inf_unocc_indices
            occ = self.region_graph.vertices[key].inf_occ_indices
            lst_occ_idx.append(occ)
            lst_free_idx.append(unocc)

        lst_free_idx = np.vstack(lst_free_idx)
        lst_occ_idx = np.vstack(lst_occ_idx)
        goal_ind[2] = start_ind[2]
        dist = lst_free_idx - np.array(goal_ind)
        dist_vals = np.array([np.linalg.norm(dist[q,:]) for q in range(dist.shape[0])])
        goal_ind = lst_free_idx[np.argmin(dist_vals)].astype(int).tolist()
        t1 = time.time()
        astar_path = restricted_astar_search(idx_min, idx_max, start_ind, goal_ind, lst_occ_idx, lst_free_idx, planner_type='bi-astar')
        print(f"astar region-mask time: {time.time()-t1}")
        is_path_found = len(astar_path) > 0
        return astar_path, is_path_found
        

    def solve_global_tsp_rev(self, 
                             robot_position,
                             occupancy_map):
        # get the node on keypose graph associated with the robot position
        
        region_graph = self.region_graph
        
        global_path_robot_position = robot_position
        # Get all the conneccted exploring cells
        global_path = []
        exploring_cell_positions = []
        exploring_region_keys = []
        exploring_cell_frontier_score = []
        exploring_cell_visit_score = []

        robot_idx = self.subspaces.Pos2Ind(global_path_robot_position)
        exploring_cell_positions.append(global_path_robot_position)
        exploring_region_keys.append(robot_idx)

        for i in region_graph.vertices.keys():
            if region_graph.get_region_status(i) == CellStatus.EXPLORING and region_graph.is_in_planning_bound(i): 
                if i in exploring_region_keys:
                    continue
                # if i is not inside the neighbor_cell_indices or viewpoint is empty             
                if not i in self.neighbor_cell_indices or (region_graph.get_frontier_score(i) > 0):
                    cell_pos = region_graph.vertices[i].target_pose[:3, 3] # 
                    exploring_cell_positions.append(np.array([cell_pos[0],cell_pos[1],cell_pos[2]])) # 
                    exploring_region_keys.append(i)
                    exploring_cell_frontier_score.append(region_graph.get_frontier_score(i))
                    exploring_cell_visit_score.append(region_graph.get_visit_count(i))

        

        if len(exploring_cell_positions) == 1:
            print("no exploring_cell_positions")
            global_path.append(robot_position)
            ordered_region_keys = []
            return global_path, ordered_region_keys, []
        
        # Construct the distance matrix
        len_cell = len(exploring_cell_positions)
        distance_matrix = np.zeros(shape=(len_cell, len_cell))        
        for i in range(len_cell-1):
            for j in range(i):
                # Check if two exporing cell has feasible path
                pos_i, pos_j = exploring_cell_positions[i], exploring_cell_positions[j]
                idx_i, idx_j = exploring_region_keys[i], exploring_region_keys[j]
                    
                is_connected, path_length = region_graph.is_connected(idx_i, idx_j)
                    
                if not is_connected:
                    distance_matrix[i][j] = 10000.0
                else:
                    distance_matrix[i][j] = 50.0 * path_length
                distance_matrix[i][j] += -5.0 * exploring_cell_frontier_score[j]
                distance_matrix[i][j] += 100.0 * exploring_cell_visit_score[j]

        for i in range(len_cell):
            for j in range(i+1, len_cell):
                distance_matrix[i][j] = distance_matrix[j][i]
        
        distance_matrix[:,0] = 0 # make it open-type tsp 
        # Solve the TSP
        # Solve with Lin-Kernighan heuristic
        # permutation, distance = solve_tsp_local_search(distance_matrix)
        permutation, distance = solve_tsp_lin_kernighan(distance_matrix=distance_matrix)
        # permutation, distance = solve_tsp_brute_force(distance_matrix=distance_matrix)
        
        # retrieve the solution and generate a global path
        node_index = copy.deepcopy(permutation)
        ordered_region_keys = []
        
        # make a loop
        if len(node_index)  > 0:
            node_index.append(node_index[0])
        
        for i in range(len(node_index)):
            cell_ind = node_index[i]
            position = exploring_cell_positions[cell_ind]
            global_path.append((position, True))
            ordered_region_keys.append(exploring_region_keys[cell_ind])
        
        
        # Post-process global path
        # 1. Find index of current position
        cur_idx = -1
        for i in range(len(global_path)):
            pos = global_path[i][0]
            if pos[0] == robot_position[0] and pos[1] == robot_position[1]:
                cur_idx = i
                break
        re_global_path = []
        only_coord_global_path = []
        for i in range(len(global_path)):
            node = global_path[(cur_idx+i)%(len(global_path))]
            re_global_path.append(node)
            only_coord_global_path.append(node[0])

        shortened_region_keys = []
        shortened_path = []
        only_coord_shortened_path = []

        start_idx = cur_idx+1
        end_idx = cur_idx-1 + len(global_path)

        while start_idx < end_idx:
            shortened_path.append(global_path[start_idx%len(global_path)])
            shortened_region_keys.append(ordered_region_keys[start_idx%len(global_path)])
            only_coord_shortened_path.append(global_path[start_idx%len(global_path)][0])
            cur_sub = occupancy_map.Pos2Sub(global_path[start_idx%len(global_path)][0]).tolist()
            tmp_idx = start_idx + 1
            while True:
                if tmp_idx == end_idx:
                    start_idx = tmp_idx
                    break
                next_sub = occupancy_map.Pos2Sub(global_path[tmp_idx%len(global_path)][0]).tolist()
                if not occupancy_map.check_los_free(cur_sub, next_sub, include_unknown=True):
                    start_idx = tmp_idx
                    break
                else:
                    tmp_idx += 1
        shortened_path.append(global_path[end_idx%len(global_path)])
        only_coord_shortened_path.append(global_path[end_idx%len(global_path)][0])
        shortened_region_keys.append(ordered_region_keys[end_idx%len(global_path)])
        return shortened_path, only_coord_shortened_path, shortened_region_keys