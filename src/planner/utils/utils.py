
import numpy as np
import math
import copy
from numpy import linalg as LA
from typing import Dict, List, Tuple

MAX_VEL=0.1
MAX_ROT=0.17453292519 # 10 deg

def trimesh_to_open3d(src):
    import open3d as o3d
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = 0.5*np.ones_like(src.vertices)#src.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst

def as_mesh(scene_or_mesh):
    import trimesh
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

'''
    Sample utils
    Uniform Sampling within N-dimensional hypersphere, hyperellipsoid.
    InformedSampler for RRT*, BIT* to get random node given goal, start and current minimum cost
    author: Bharath Chandra
    email: iambharathchandra@gmail.com
'''
def spherical_fibonacci_sampling(n):
    
    """
    Spherical Fibonacci Sampling for generating n points on a sphere.
    
    Parameters:
        n (int): Number of points to generate.
        
    Returns:
        np.ndarray: Array of shape (n, 3) with points on the unit sphere.
    """
    indices = np.arange(0, n, dtype=float) + 0.5  # Half-index for uniform spacing
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio
    
    theta = 2 * np.pi * indices / phi  # Azimuthal angle
    z = 1 - 2 * indices / n  # Z-coordinates uniformly spaced between -1 and 1
    radius = np.sqrt(1 - z ** 2)  # Radius at each z-level (unit sphere)
    
    x = radius * np.cos(theta)  # X-coordinates
    y = radius * np.sin(theta)  # Y-coordinates
    
    return np.column_stack((x, y, z))  # Combine x, y, z into shape (n, 3)

def SampleUnitNBall(dim = 3,num = 1):
    '''
    uniformly sample a N-dimensional unit UnitBall
    Reference:
      Efficiently sampling vectors and coordinates from the n-sphere and n-ball
      http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    Input:
        num - no. of samples
        dim - dimensions
    Output:
        uniformly sampled points within N-dimensional unit ball
    '''
    #Sample on a unit N+1 sphere
    u = np.random.normal(0, 1, (num, dim + 2))
    norm = LA.norm(u, axis = -1,keepdims = True)
    u = u/norm
    #The first N coordinates are uniform in a unit N ball
    if num == 1: return u[0,:dim]
    return u[:,:dim]


def SphereSampler(center, radius, num = 1):
    '''
    uniformly sample inside N-dimensional hypersphere
    Input:
        center - Center of sphere
        radius - Radius of Sphere
        num - no. of samples
    Output:
       uniformly sampled points inside the hypersphere
    '''
    dim = center.shape[0]
    xball = SampleUnitNBall(dim,num)
    return radius*xball + center

def mod(value, modulus):
    return math.fmod(math.fmod(value, modulus) + modulus, modulus)

def intbound(s, ds):
    if ds<0:
        return intbound(-s,-ds)
    else:
        s = mod(s, 1)
        return (1-s)/ds

def raycast(start_sub, end_sub, max_sub=None, min_sub=None):
    raycast_cells = []
    
    if (start_sub == end_sub):
        raycast_cells.append(start_sub)
        return raycast_cells
    
    diff_sub = [end_sub[i] - start_sub[i] for i in range(len(start_sub))]
    max_dist = np.linalg.norm(diff_sub)
    step_x = np.sign(diff_sub[0])
    step_y = np.sign(diff_sub[1])
    step_z = np.sign(diff_sub[2])
    
    t_max_x, t_max_y, t_max_z = np.inf, np.inf, np.inf
    t_delta_x, t_delta_y, t_delta_z = np.inf, np.inf, np.inf
    if not step_x == 0:
        t_max_x = intbound(start_sub[0], diff_sub[0])
    if not step_y == 0:
        t_max_y = intbound(start_sub[1], diff_sub[1])
    if not step_z == 0:
        t_max_z = intbound(start_sub[2], diff_sub[2])
    
    if not step_x == 0:
        t_delta_x = float(step_x) / float(diff_sub[0])
    if not step_y == 0:
        t_delta_y = float(step_y) / float(diff_sub[1])
    if not step_z == 0:
        t_delta_z = float(step_z) / float(diff_sub[2])
    
    dist = 0
    cur_sub = copy.deepcopy(start_sub)
    while True:
        raycast_cells.append(copy.deepcopy(cur_sub))
        dist = np.linalg.norm([cur_sub[i]-start_sub[i] for i in range(len(cur_sub))])
        if cur_sub == end_sub or dist > max_dist:
            break
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                cur_sub[0] += step_x
                t_max_x += t_delta_x
            else:
                cur_sub[2] += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                cur_sub[1] += step_y
                t_max_y += t_delta_y
            else:
                cur_sub[2] += step_z
                t_max_z += t_delta_z

    return raycast_cells

def check_occupancy(start_sub, end_sub, occupancy_map, include_unknown=False, include_collision_threshold=True, use_raw_data=False):
    # for batch subs (n_sub, 3)
    t_values = np.linspace(0, 1, 30)
    view_vec = end_sub - start_sub
    cells = start_sub[..., None] + t_values * view_vec[..., None]
    cells = cells.astype(np.int64).transpose(0, 2, 1).reshape(-1, 3)
    sh = cells.shape
    cells = cells.reshape(-1, 3)
    
    if use_raw_data:
        is_occupied = occupancy_map.raw_data[cells[..., 0], cells[..., 1], cells[..., 2]] <= occupancy_map.collision_threshold
    elif include_unknown:
        if include_collision_threshold:
            is_occupied = occupancy_map.occupancy_data[cells[..., 0], cells[..., 1], cells[..., 2]] == occupancy_map.occupied
        else:
            is_occupied = occupancy_map.raw_data[cells[..., 0], cells[..., 1], cells[..., 2]] < 0.0
            is_certain = occupancy_map.uncertainty_data[cells[..., 0], cells[..., 1], cells[..., 2]] < 3.0
            is_occupied = np.logical_and(is_occupied, is_certain)                    
    else:
        is_occupied = occupancy_map.data[cells[..., 0], cells[..., 1], cells[..., 2]] == occupancy_map.occupied

    return is_occupied.reshape(sh[:-1]) # (n_sub, 30)

def is_collision_free(start_pos, lin_vel, dt, occupancy_map, include_unknown=True, include_collision_threshold=True):
    num = int((np.linalg.norm(lin_vel) * dt) / 0.02) + 1 # check every 2cm
    points = start_pos[None, :] + lin_vel[None, :] * np.linspace(0, dt, num=num)[:,None]
    _is_occupied = np.ones(points.shape[0]).astype(np.bool_)
    in_range = np.zeros(points.shape[0]).astype(np.bool_)
    subs = []
    for i, point in enumerate(points):
        sub = occupancy_map.Pos2Sub(point)
        if occupancy_map.in_range(sub):
            in_range[i] = True
            subs.append(sub)
    subs = np.stack(subs)
    
    if include_unknown:
        if include_collision_threshold:
            is_occupied = occupancy_map.occupancy_data[subs[:, 0], subs[:, 1], subs[:, 2]] == occupancy_map.occupied
        else:
            is_occupied = occupancy_map.raw_data[subs[:, 0], subs[:, 1], subs[:, 2]] < 0.0
            is_certain = occupancy_map.uncertainty_data[subs[:, 0], subs[:, 1], subs[:, 2]] < 3.0
            is_occupied = np.logical_and(is_occupied, is_certain)                    
    else:
        is_occupied = occupancy_map.data[subs[:, 0], subs[:, 1], subs[:, 2]] == occupancy_map.occupied

    _is_occupied[in_range] = is_occupied
    if np.logical_not(_is_occupied).sum() == len(points):
        num_collision_free = max((len(points) - 1) // 5, 1) # 10
        complete_free = True
    else:
        num_collision_free = (np.argmax(_is_occupied) - 1) // 5
        complete_free = False
    
    return num_collision_free, complete_free


import cc3d
import time

def connected_component_labelling(labels_in):
    connectivity = 8 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out, N = cc3d.connected_components(labels_in, connectivity=connectivity, binary_image=True, return_N=True)
    if N <=1:
        return False, None, None
    stats = cc3d.statistics(labels_out)

    voxel_counts = stats['voxel_counts']
    bboxes = stats['bounding_boxes']
    
    if voxel_counts[0] / voxel_counts[1] > 4.0:
        return False, None, None
    first_region_indices = np.argwhere(labels_out == 1)
    second_region_indices = np.argwhere(labels_out == 2)
    return True, first_region_indices, second_region_indices


def connected_component_3d_labelling(occupancy_indices):
    s_time = time.time()
    connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out, N = cc3d.connected_components(occupancy_indices, connectivity=connectivity, binary_image=True, return_N=True)
    print(f"connected component time: {(time.time()-s_time)*1000.0:.3f} [ms]")
    if N <=1:
        return False, None, None
    stats = cc3d.statistics(labels_out)

    voxel_counts = stats['voxel_counts']
    bboxes = stats['bounding_boxes']
    
    if voxel_counts[0] / voxel_counts[1] > 10.0:
        return False, None, None
    
    print(stats)
    print()
    first_region_indices = np.argwhere(labels_out == 0)
    second_region_indices = np.argwhere(labels_out != 0)
    return True, first_region_indices, second_region_indices


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


def trilinear_interpolation(voxel_grid: np.ndarray, point: np.ndarray) -> np.ndarray:
    """ trilinear interpolation

    Args:
        voxel_grid (np.ndarray, [H,W,D]): voxel grid
        point (np.ndarray, [3])         : query point

    Returns
        interpolated_value (float): interpolated value
    """
    # Get the dimensions of the voxel grid
    H, W, D = voxel_grid.shape
    
    # Extract the coordinates of the point
    x, y, z = point
    
    # Check if the point is within the valid range
    if (x < 0 or x >= H - 1 or
        y < 0 or y >= W - 1 or
        z < 0 or z >= D - 1):
        return None  # Point is outside the grid
    
    # Calculate the integer and fractional parts of the coordinates
    x0, x1 = int(x), int(x) + 1
    y0, y1 = int(y), int(y) + 1
    z0, z1 = int(z), int(z) + 1
    
    dx, dy, dz = x - x0, y - y0, z - z0
    
    # Perform trilinear interpolation
    c000 = voxel_grid[x0, y0, z0]
    c001 = voxel_grid[x0, y0, z1]
    c010 = voxel_grid[x0, y1, z0]
    c011 = voxel_grid[x0, y1, z1]
    c100 = voxel_grid[x1, y0, z0]
    c101 = voxel_grid[x1, y0, z1]
    c110 = voxel_grid[x1, y1, z0]
    c111 = voxel_grid[x1, y1, z1]
    
    interpolated_value = (1 - dx) * (1 - dy) * (1 - dz) * c000 + \
                        (1 - dx) * (1 - dy) * dz * c001 + \
                        (1 - dx) * dy * (1 - dz) * c010 + \
                        (1 - dx) * dy * dz * c011 + \
                        dx * (1 - dy) * (1 - dz) * c100 + \
                        dx * (1 - dy) * dz * c101 + \
                        dx * dy * (1 - dz) * c110 + \
                        dx * dy * dz * c111
    
    return interpolated_value

def query_sdf_np(sdf_grid, points):
    """ Query sdf values (numpy implementation)

    Args:
        sdf_grid (np.ndarray, [H,W,D]): sdf grid
        points (np.ndarray, [N, 3])   : query points

    Returns
        sdf (np.ndarray, [N]): queried SDF values
    """
    sdf = np.array([trilinear_interpolation(sdf_grid, point) for point in points])
    return sdf

def project_point_to_cuboid(px, py, pz, min_x, max_x, min_y, max_y, min_z, max_z):
    """
    Projects a point (px, py, pz) onto the surface of a cuboid defined by
    min_x, max_x, min_y, max_y, min_z, and max_z.

    Returns the coordinates of the projected point.
    """
    # Clamp the point's coordinates to the cuboid's boundaries
    proj_x = min(max(px, min_x), max_x)
    proj_y = min(max(py, min_y), max_y)
    proj_z = min(max(pz, min_z), max_z)

    return np.array([proj_x, proj_y, proj_z])


def set_path_between_pose(robot, from_pose,to_pose):
    path = []
    pos_dist = np.linalg.norm(from_pose[0] - to_pose[0])
    ang_dist = angular_difference(from_pose[1], to_pose[1])
    max_t = max(pos_dist / (MAX_VEL), ang_dist / (MAX_ROT))
    
    lin_vel = (to_pose[0] - from_pose[0]) / max_t
    if ang_dist < 1e-4:
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