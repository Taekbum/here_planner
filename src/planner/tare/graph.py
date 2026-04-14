
class Graph(object):
    def __init__(self):
        self.connection = []
        self.distance = []
        self.positions = []
            
    def node_index_in_range(self, node_idx:int):
        return node_idx >= 0 and node_idx < len(self.connection)
    
        