import copy
import numpy as np
from multipledispatch import dispatch
from .grid import Grid

class RollingGrid(object):
    def __init__(self, size, bound):
        self.size = size
        self.which_grid = True
        self.array_ind_to_ind = [-1] * (size[0]*size[1]*size[2])
        
        self.grid0 = Grid(size=self.size, bound=bound, dimension=3)
        self.grid1 = Grid(size=self.size, bound=bound, dimension=3)
        # initialize array indices
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    ind = self.grid0.Sub2Ind(x,y,z)
                    self.grid0.set_cell(x,y,z,ind)
                    self.grid1.set_cell(x,y,z,ind)
                    self.array_ind_to_ind[ind] = ind
    
    def roll(self, roll_dir:np.ndarray):
        if roll_dir[0] == 0 and roll_dir[1] == 0 and roll_dir[2] == 0:
            return
        
        if self.which_grid:
            self.roll_helper(self.grid1, self.grid0, roll_dir)
        else:
            self.roll_helper(self.grid0, self.grid1, roll_dir)
        self.get_rolled_in_indices(roll_dir)
        
        self.which_grid = not self.which_grid
        
        # updae array ind to ind mapping
        cell_num = self.size[0] * self.size[1] * self.size[2]
        for ind in range(cell_num):
            if self.which_grid:
                array_ind = self.grid1.get_cell_value(ind)
            else:
                array_ind = self.grid0.get_cell_value(ind)
            self.array_ind_to_ind[array_ind] = ind
    
    def roll_helper(self, grid_in:Grid, grid_out:Grid, roll_dir):
        grid_in_size = grid_in.get_size()
        grid_out_size = grid_out.get_size()
        
        assert grid_in_size[0] == grid_out_size[0] and grid_in_size[0] == self.size[0]
        assert grid_in_size[1] == grid_out_size[1] and grid_in_size[1] == self.size[1]
        assert grid_in_size[2] == grid_out_size[2] and grid_in_size[2] == self.size[2]
        
        roll_dir[0] %= self.size[0]
        roll_dir[1] %= self.size[1]
        roll_dir[2] %= self.size[2]
        
        dir = copy.deepcopy(roll_dir)
        if roll_dir[0] < 0:
            dir[0] += self.size[0]
        if roll_dir[1] < 0:
            dir[1] += self.size[1]
        if roll_dir[2] < 0:
            dir[2] += self.size[2]
        
        cell_num = self.size[0] * self.size[1] * self.size[2]
        for ind in range(cell_num):
            sub = grid_out.Ind2Sub(ind)
            from_x = self.get_from_idx(sub[0], dir[0], self.size[0])
            from_y = self.get_from_idx(sub[1], dir[1], self.size[1])
            from_z = self.get_from_idx(sub[2], dir[2], self.size[2])
            grid_out.set_cell_with_ind(ind, grid_in.get_cell_value(from_x, from_y, from_z))
            
    def get_from_idx(self, cur_idx, roll_step, max_idx)->int:
        if cur_idx <= roll_step - 1:
            idx = max_idx - roll_step + cur_idx        
        else:
            idx = cur_idx - roll_step
        if type(idx) == int:
            return idx
        else:
            return idx.item()
        
    def get_rolled_in_indices(self, roll_dir):
        start_idx = np.zeros((3,), dtype=np.int64)
        end_idx = np.zeros((3,), dtype=np.int64)
        if roll_dir[0] < 0:
            start_idx[0] = self.size[0] + roll_dir[0]
        if roll_dir[1] < 0:
            start_idx[1] = self.size[1] + roll_dir[1]
        if roll_dir[2] < 0:
            start_idx[2] = self.size[2] + roll_dir[2]
        
        if roll_dir[0] >= 0:
            end_idx[0] = roll_dir[0] - 1
        else:
            end_idx[0] = self.size[0] - 1
        if roll_dir[1] >= 0:
            end_idx[1] = roll_dir[1] - 1
        else:
            end_idx[1] = self.size[1] - 1
        if roll_dir[2] >= 0:
            end_idx[2] = roll_dir[2] - 1
        else:
            end_idx[2] = self.size[2] - 1
        
        dir = np.zeros((3,), dtype=np.int64)
        if roll_dir[0] >= 0:
            dir[0] = roll_dir[0]
        else:
            dir[0] = self.size[0] + roll_dir[0]
        if roll_dir[1] >= 0:
            dir[1] = roll_dir[1]
        else:
            dir[1] = self.size[1] + roll_dir[1]
        if roll_dir[2] >= 0:
            dir[2] = roll_dir[2]
        else:
            dir[2] = self.size[2] + roll_dir[2]
        
        self.updated_indices = []
        if dir[0] > 0:
            self.updated_indices = self.get_indices(self.updated_indices, 
                                                    np.array([start_idx[0], 0, 0],dtype=np.int64), 
                                                    np.array([end_idx[0], self.size[1]-1, self.size[2]-1],dtype=np.int64))
        if dir[1] > 0:
            x_start = 0
            x_end = self.size[0] - 1
            if start_idx[0] == 0:
                x_start = end_idx[0] + 1
            else:
                x_end = start_idx[0] - 1
            self.updated_indices = self.get_indices(self.updated_indices,
                                                    np.array([x_start, start_idx[1], 0],dtype=np.int64), 
                                                    np.array([x_end, end_idx[1], self.size[2]-1],dtype=np.int64))
        if dir[2] > 0:
            x_start = 0
            x_end = self.size[0] - 1
            y_start = 0
            y_end = self.size[1] - 1
            if start_idx[0] == 0:
                x_start = end_idx[0] + 1
            else:
                x_end = start_idx[0] -1
            if start_idx[1] == 0:
                y_start = end_idx[1] + 1
            else:
                y_end = start_idx[1] -1        
            self.updated_indices = self.get_indices(self.updated_indices,
                                                    np.array([x_start, y_start, start_idx[2]],dtype=np.int64), 
                                                    np.array([x_end, y_end, end_idx[2]],dtype=np.int64))

    
    def get_indices(self, indices, start_idx, end_idx):
        start_idx[0] = start_idx[0] % self.size[0]
        start_idx[1] = start_idx[1] % self.size[1]
        start_idx[2] = start_idx[2] % self.size[2]

        end_idx[0] = end_idx[0] % self.size[0]
        end_idx[1] = end_idx[1] % self.size[1]
        end_idx[2] = end_idx[2] % self.size[2]
        
        start_idx[0] = min(start_idx[0], end_idx[0])
        start_idx[1] = min(start_idx[1], end_idx[1])
        start_idx[2] = min(start_idx[2], end_idx[2])

        end_idx[0] = max(start_idx[0], end_idx[0])
        end_idx[1] = max(start_idx[1], end_idx[1])
        end_idx[2] = max(start_idx[2], end_idx[2])
        
        for x in range(start_idx[0], end_idx[0]+1):
            for y in range(start_idx[1], end_idx[1]+1):
                for z in range(start_idx[2], end_idx[2]+1):
                    indices.append(self.grid0.Sub2Ind(x, y, z))
        return indices
    
    def get_updated_indices(self):
        return self.updated_indices
        
    def Sub2Ind(self, sub):
        return self.grid0.Sub2Ind(sub)
    
    def Ind2Sub(self, ind):
        return self.grid0.Ind2Sub(ind)
    
    @dispatch(np.ndarray)
    def in_range(self, sub):
        return self.grid0.in_range(sub)

    @dispatch(list)
    def in_range(self, sub):
        return self.grid0.in_range(sub)

    @dispatch(int)
    def in_range(self, ind):
        return self.grid0.in_range(ind)
                
    @dispatch(list)
    def get_array_ind(self, sub:list):
        assert type(sub) == list
        assert self.in_range(sub)
        if self.which_grid:
            return self.grid1.get_cell_value(sub[0], sub[1], sub[2])
        else:
            return self.grid0.get_cell_value(sub[0], sub[1], sub[2])
    
    @dispatch(np.ndarray)
    def get_array_ind(self, sub:np.ndarray):
        assert type(sub) == np.ndarray
        assert self.in_range(sub)
        if self.which_grid:
            return self.grid1.get_cell_value(sub[0].item(), sub[1].item(), sub[2].item())
        else:
            return self.grid0.get_cell_value(sub[0].item(), sub[1].item(), sub[2].item())
    
    @dispatch(int)
    def get_array_ind(self, ind:int):
        assert self.in_range(ind)
        sub = self.grid0.Ind2Sub(ind)
        return self.get_array_ind(sub)