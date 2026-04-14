import numpy as np
import math
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any


from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
from pathfinding3d.finder.bi_a_star import BiAStarFinder
from pathfinding3d.finder.theta_star import ThetaStarFinder
from pathfinding3d.core.diagonal_movement import DiagonalMovement


def restricted_astar_search(index_min, index_max, start, goal, lst_occ_idx, lst_free_idx=None, planner_type='astar', diagonal=True):
    size = index_max - index_min
    start = start - index_min
    goal = goal - index_min
    
    if lst_free_idx is None:
        matrix = np.ones((size[0]+1, size[1]+1, size[2]+1), dtype=np.int8)
        lst_occ_idx = lst_occ_idx[lst_occ_idx[:,2]<=index_max[2]]
        matrix[np.minimum(size[0],np.maximum(lst_occ_idx[:,0]-index_min[0],0)), 
               np.minimum(size[1],np.maximum(lst_occ_idx[:,1]-index_min[1],0)), 
               np.minimum(size[2],np.maximum(lst_occ_idx[:,2]-index_min[2],0))] = 0
    else:
        matrix = np.zeros((size[0]+1, size[1]+1, size[2]+1), dtype=np.int8) # initialize to occupied
        idx1 = lst_free_idx[:,2] - index_min[2] < index_max[2]
        matrix[lst_free_idx[idx1,0]-index_min[0], lst_free_idx[idx1,1]-index_min[1], lst_free_idx[idx1,2]-index_min[2]] = 1 # only interested regions, mask it as free
        idx2 = lst_occ_idx[:,2] - index_min[2] < index_max[2]
        matrix[lst_occ_idx[idx2,0]-index_min[0], lst_occ_idx[idx2,1]-index_min[1], lst_occ_idx[idx2,2]-index_min[2]] = 0 # in interested regions, mask occupied index
        
    grid = Grid(matrix=matrix)
    start_node = grid.node(start[0], start[1], start[2])
    goal_node = grid.node(goal[0], goal[1], goal[2])
    
    if planner_type == 'astar':
        if diagonal:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        else:
            finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    elif planner_type == 'bi-astar':
        if diagonal:
            finder = BiAStarFinder(diagonal_movement=DiagonalMovement.always)
        else:
            finder = BiAStarFinder(diagonal_movement=DiagonalMovement.never)
    elif planner_type == 'theta-star':
        if diagonal:
            finder = ThetaStarFinder(diagonal_movement=DiagonalMovement.always)
        else:
            finder = ThetaStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start_node, goal_node, grid)
    path = [p.identifier for p in path]
    if len(path) > 0:
        path = np.array(path) + index_min
        # print("astar found")
    return path

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

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
            # if not self.lst_connected[4][i]: # 
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