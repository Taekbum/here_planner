import numpy as np

class ViewPoint(object):
    '''
    ViewPoint is pose of a sensor,
    In this case, we only consider position (without consideration of the orientation, similar to 360 degree of fov)
    only used for global planning
    '''
    def __init__(self, p:np.ndarray=np.zeros(3)):
        self.position = p
        self.in_collision = False
        self.in_los = False
        self.connected = False
        self.visited = False
        self.selected = False
        self.candidate = False
        self.has_terrain_height = False
        self.has_terrain_neighbor = False
        self.in_exploring_cell = False
        self.cell_ind = -1
        self.collision_frame_count = 0
        self.terrain_height = 0.0
        self.covered_point_list = []
        self.covered_frontier_point_list = []
        
    
    def reset(self):
        self.in_collision = False
        self.in_los = False
        self.connected = False
        self.visited = False
        self.selected = False
        self.candidate = False
        self.has_terrain_height = False
        self.has_terrain_neighbor = False
        self.in_exploring_cell = False
        self.cell_ind = -1
        self.collision_frame_count = 0
        self.terrain_height = 0.0
        self.covered_point_list = []
        self.covered_frontier_point_list = []
    
    def set_in_exploring_cell(self, in_exploring_cell):
        self.in_exploring_cell = in_exploring_cell
        
    def get_position(self):
        return self.position
        
    def is_candidate(self):
        return self.candidate
        
    def set_in_collision(self, in_collision):
        self.in_collision = in_collision
    
    def is_in_collision(self):
        return self.in_collision
    
    def set_in_los(self, in_los):
        self.in_los = in_los
    
    def set_position(self, position):
        self.position = position
        
    def is_connected(self):
        return self.connected
    
    def set_visited(self, visited):
        self.visited = visited
        
    def is_visited(self):
        return self.visited
    
    def is_selected(self):
        return self.selected
        
    def get_covered_frontier_point_num(self):
        return len(self.covered_frontier_point_list)
            
    def get_covered_point_num(self)->int:
        return len(self.covered_point_list)
    
    def get_cell_ind(self):
        return self.cell_ind
    
    def set_cell_ind(self, ind):
        self.cell_ind = ind