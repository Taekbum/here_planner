import numpy as np
import copy
from scipy.spatial.transform import Rotation

from ..utils.kinematics import SE3Kinematics, angular_difference
from ..tare.grid import OccupancyGrid
from ..utils.utils import is_collision_free, MAX_VEL, MAX_ROT

from .python_tsp.heuristics.lin_kernighan import solve_tsp_lin_kernighan
from .python_tsp.heuristics.local_search import solve_tsp_local_search
from ..rotation_planning import rotation_planning

def set_path_between_pose_mat(robot, from_pose, to_pose):
    path = []
    pos_dist = np.linalg.norm(from_pose[:3,3] - to_pose[:3,3])
    ang_dist = angular_difference(Rotation.from_matrix(from_pose[:3,:3]), Rotation.from_matrix(to_pose[:3,:3]))
    max_t = max(pos_dist / (MAX_VEL), ang_dist / (MAX_ROT))
    
    lin_vel = (to_pose[:3,3] - from_pose[:3,3]) / max_t
    if ang_dist < 1e-3:
        ang_vel = np.zeros(3)
    else:
        ang_vel = robot.relative_angular_velocity_v3(Rotation.from_matrix(from_pose[:3,:3]), Rotation.from_matrix(to_pose[:3,:3]), dt=1, ang_max=ang_dist / max_t) 

    step = 0.0
    while step < max_t:
        tmp_pos, tmp_rot = robot.transition((from_pose[:3,3],Rotation.from_matrix(from_pose[:3,:3])), (lin_vel,ang_vel), step+1)
        tmp_pose = np.vstack((np.hstack((tmp_rot.as_matrix(),tmp_pos.reshape((3,1)))), np.array([[0.,0.,0.,1.]])))
        path.append(tmp_pose)
        if np.any(np.isnan(tmp_rot.as_matrix())):
            print('nan value occurs')
        step += 1
    return path


def set_path_between_pose(robot, from_pose, to_pose):
    path = []
    pos_dist = np.linalg.norm(from_pose[0] - to_pose[0])
    ang_dist = angular_difference(from_pose[1], to_pose[1])
    max_t = max(pos_dist / (MAX_VEL), ang_dist / (MAX_ROT))
    
    lin_vel = (to_pose[0] - from_pose[0]) / max_t
    if ang_dist < 1e-3:
        ang_vel = np.zeros(3)
    else:
        ang_vel = robot.relative_angular_velocity_v3(from_pose[1], to_pose[1], dt=1, ang_max=ang_dist / max_t) 

    step = 0.0
    while step < max_t:
        tmp_state = robot.transition(from_pose, (lin_vel,ang_vel), step+1)
        path.append(copy.deepcopy(tmp_state))
        if np.any(np.isnan(tmp_state[1].as_matrix())):
            print('nan value occurs')
        step += 1
    return path
    
class LocalTSPPlanner(object):
    def __init__(self,
                 robot:SE3Kinematics,
                 interval=0.5,
                 ):
        self.robot = robot
        self.interval = interval
            
    def set_occupancy_map(self, map:OccupancyGrid):
        self.occupancy_map = map
             
    def rotate_sequenece(self, pose, target_rots, max_rot_deg):
        '''
        rotation at same position
        '''
        rotate_path = rotation_planning(pose[1].as_matrix(), target_rots, max_rot_deg)
        pose_path = [(pose[0], Rotation.from_matrix(rotate_path[i])) for i in range(1, len(rotate_path))] 
        
        return pose_path
    
    # Local TSP planning with goal pose given 
    def solve_local_tsp_with_global_goal(self, cur_state, target_poses, goal_pose):
        cur_position = cur_state[:3,3]
        cur_rotation = cur_state[:3,:3]
        goal_position = goal_pose[:3,3]
        
        cur_pose = (cur_position, Rotation.from_matrix(cur_rotation))
        
        candidate_position = []
        candidate_pose = []
        candidate_pose_indices = []
        for i in range(len(target_poses)):
            pos = target_poses[i][0,:3,3]
            candidate_position.append(pos)
            candidate_pose.append(target_poses[i][0])
            candidate_pose_indices.append(i)
        # Add start node
        candidate_position.append(cur_position)
        candidate_pose.append(cur_state)
        candidate_pose_indices.append(len(target_poses))
        # Add goal node
        candidate_position.append(goal_position)
        candidate_pose.append(goal_pose)
        candidate_pose_indices.append(len(target_poses)+1)
        # Add dummy node
        candidate_position.append(np.zeros(3))
        candidate_pose.append(np.eye(4))
        candidate_pose_indices.append(len(target_poses)+2)

        len_poses = len(candidate_pose_indices)
        distance_matrix = np.zeros(shape=(len_poses, len_poses))
        for i in range(len(target_poses)+1):
            for j in range(i):
                pos_i, pos_j = candidate_position[i], candidate_position[j]
                lin_vel = (pos_j - pos_i)
                _, is_free = is_collision_free(pos_i, lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
                if not is_free:
                    distance_matrix[i][j] = 1000.0
                    distance_matrix[j][i] = 1000.0
                else:
                    pos_dist = np.linalg.norm(pos_i - pos_j)
                    max_t = pos_dist / MAX_VEL
                    distance_matrix[i][j] = max_t
                    distance_matrix[j][i] = max_t

        # For dummy node, zero value to start and goal node, and infinity for other nodes
        distance_matrix[:,len(target_poses)+2] = 100000000.0
        distance_matrix[len(target_poses)+2,:] = 100000000.0
        distance_matrix[len(target_poses)+2,len(target_poses)+2] = 0.0
        distance_matrix[len(target_poses)+2,len(target_poses)+1] = 0.0
        distance_matrix[len(target_poses)+2,len(target_poses)] = 0.0
        distance_matrix[len(target_poses),len(target_poses)+2] = 0.0
        distance_matrix[len(target_poses)+1,len(target_poses)+2] = 0.0
        #  
        permutation, distance = solve_tsp_local_search(distance_matrix)    
        # permutation, distance = solve_tsp_lin_kernighan(distance_matrix)
        # retrieve the solution and generate a global path
        node_index = copy.deepcopy(permutation)
        start_idx =  node_index.index(len(target_poses)) #  find a index of start pose 
        dummy_idx = node_index.index(len(target_poses)+2) # find a index of dummy node
        goal_idx = node_index.index(len(target_poses)+1) # find a index of goal pose
        # If start - dummy - goal permutation, reverse it 
        if dummy_idx == (start_idx + 1)%len(node_index) and goal_idx == (dummy_idx+1)%len(node_index):
            is_reversed = True
            node_index.reverse() # Again, compute start, goal idx from node_index
            start_idx =  node_index.index(len(target_poses)) #  find a index of start pose 
            dummy_idx = node_index.index(len(target_poses)+2) # find a index of dummy node
            goal_idx = node_index.index(len(target_poses)+1) # find a index of goal pose
        else:
            is_reversed = False
        ordered_pose_indices = []
        ordered_poses = []
        for i in range(len(node_index)):
            ordered_poses.append(candidate_position[node_index[i]])
        
        
        path = []
        tmp_pose = (cur_position, Rotation.from_matrix(cur_rotation))
        for i in range(1, len(node_index)-1): #  -> to include path to goal
            idx = (start_idx+i)%len(candidate_position) # next pose from TSP path
            if idx == goal_idx:
                # Last goal node, only single rotation 
                next_pose_math = goal_pose
            else:
                # next_pose_math = candidate_pose[node_index[idx]] # first pose for each target pose (position,rotation) pairs + current pose at last
                angular_differences = [angular_difference(tmp_pose[1], Rotation.from_matrix(target_poses[node_index[idx]][j, :3, :3])) for j in range(len(target_poses[node_index[idx]]))]
                nearest_rot_ind = np.argmin(angular_differences)
                next_pose_math = target_poses[node_index[idx]][nearest_rot_ind] 
                
            # Check collision-free
            lin_vel = next_pose_math[:3,3] - tmp_pose[0]
            _, is_free = is_collision_free(tmp_pose[0], lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
            if not is_free:
                continue  # break

            next_pose = (next_pose_math[:3,3], Rotation.from_matrix(next_pose_math[:3,:3]))
            tmp_path = set_path_between_pose(self.robot, tmp_pose, next_pose)
            path.extend(tmp_path)
                
            if idx is not goal_idx:
                # rotation sequence
                
                # rot_path = self.rotate_sequenece(next_pose, target_poses[node_index[idx]][:min(4,len(target_poses[node_index[idx]])),:3,:3], np.rad2deg(MAX_ROT))
                target_poses_tmp = [target_poses[node_index[idx]][j] for j in range(len(target_poses[node_index[idx]]))]
                target_poses_tmp.pop(nearest_rot_ind)
                if len(target_poses_tmp) > 0:
                    target_poses_tmp = np.stack(target_poses_tmp)
                    rot_path = self.rotate_sequenece(path[-1], target_poses_tmp[:min(4,len(target_poses_tmp)),:3,:3], np.rad2deg(MAX_ROT)) 
                else:
                    rot_path = []
                path.extend(rot_path)
                tmp_pose = path[-1] 
            
        return path

    def solve_local_tsp(self, cur_state, target_poses):
        
        cur_position = cur_state[:3,3]
        cur_rotation = cur_state[:3,:3]
        
        cur_pose = (cur_position, Rotation.from_matrix(cur_rotation))
        
        candidate_position = []
        candidate_pose = []
        candidate_pose_indices = []
        for i in range(len(target_poses)):
            pos = target_poses[i][0,:3,3]
            # rot = target_poses[i][:3,:3]
            candidate_position.append(pos)
            candidate_pose.append(target_poses[i][0])
            candidate_pose_indices.append(i)
        candidate_position.append(cur_position)
        candidate_pose.append(cur_state)
        candidate_pose_indices.append(len(target_poses))

        len_poses = len(candidate_pose_indices)
        distance_matrix = np.zeros(shape=(len_poses, len_poses))
        for i in range(len_poses):
            for j in range(i):
                pos_i, pos_j = candidate_position[i], candidate_position[j]
                lin_vel = (pos_j - pos_i)
                _, is_free = is_collision_free(pos_i, lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
                if not is_free:
                    distance_matrix[i][j] = 1000.0
                    distance_matrix[j][i] = 1000.0
                else:
                    pos_dist = np.linalg.norm(pos_i - pos_j)
                    max_t = pos_dist / MAX_VEL
                    distance_matrix[i][j] = max_t
                    distance_matrix[j][i] = max_t
        
        # permutation, distance = solve_tsp_local_search(distance_matrix)
        permutation, distance = solve_tsp_lin_kernighan(distance_matrix=distance_matrix)
        # permutation, distance = solve_tsp_brute_force(distance_matrix=distance_matrix)
        
        # retrieve the solution and generate a global path
        node_index = copy.deepcopy(permutation)
        ordered_poses = []
        for i in range(len(node_index)):
            ordered_poses.append(candidate_position[node_index[i]])
        start_idx =  node_index.index(len(target_poses)) #  find a index of start pose
        
        path = []
        tmp_pose = (cur_position, Rotation.from_matrix(cur_rotation))
        for i in range(1, len(candidate_pose)):
            idx = (start_idx+i)%len(candidate_position) # next pose from TSP path
            # next_pose_math = candidate_pose[node_index[idx]] # first pose for each target pose (position,rotation) pairs + current pose at last
            angular_differences = [angular_difference(tmp_pose[1], Rotation.from_matrix(target_poses[node_index[idx]][j, :3, :3])) for j in range(len(target_poses[node_index[idx]]))]
            nearest_rot_ind = np.argmin(angular_differences)
            next_pose_math = target_poses[node_index[idx]][nearest_rot_ind] 
            
            # Check collision-free
            lin_vel = next_pose_math[:3,3] - tmp_pose[0]
            _, is_free = is_collision_free(tmp_pose[0], lin_vel, 1.0, self.occupancy_map, include_unknown=True, include_collision_threshold=True)
            if not is_free:
                continue  # break

            next_pose = (next_pose_math[:3,3], Rotation.from_matrix(next_pose_math[:3,:3]))
            tmp_path = set_path_between_pose(self.robot, tmp_pose, next_pose)
            path.extend(tmp_path)
            
            # rotation sequence
            
            target_poses_tmp = [target_poses[node_index[idx]][j] for j in range(len(target_poses[node_index[idx]]))]
            target_poses_tmp.pop(nearest_rot_ind)
            if len(target_poses_tmp) > 0:
                target_poses_tmp = np.stack(target_poses_tmp)
                rot_path = self.rotate_sequenece(path[-1], target_poses_tmp[:min(4,len(target_poses_tmp)),:3,:3], np.rad2deg(MAX_ROT)) 
            else:
                rot_path = []
            path.extend(rot_path)
            tmp_pose = path[-1] 
            
        return path
    