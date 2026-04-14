"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import cv2
import mmengine
import numpy as np
import torch
import math
import copy
from typing import Union
from matplotlib import pyplot as plt

from src.utils.general_utils import InfoPrinter

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


#### SimpleRecon ####
def colormap_image(
        image_1hw,
        mask_1hw=None,
        invalid_color=(0.0, 0, 0.0),
        flip=True,
        vmin=None,
        vmax=None,
        return_vminvmax=False,
        colormap="turbo",
):
    """
    Colormaps a one channel tensor using a matplotlib colormap.
    Args:
        image_1hw: the tensor to colomap.
        mask_1hw: an optional float mask where 1.0 donates valid pixels.
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the
            tensor.
        return_vminvmax: when true, returns vmin and vmax.
    Returns:
        image_cm_3hw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.
    """
    valid_vals = image_1hw if mask_1hw is None else image_1hw[mask_1hw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(
        plt.cm.get_cmap(colormap)(
            torch.linspace(0, 1, 256)
        )[:, :3]
    ).to(image_1hw.device)
    if flip:
        cmap = torch.flip(cmap, (0,))

    h, w = image_1hw.shape[1:]

    image_norm_1hw = (image_1hw - vmin) / (vmax - vmin)
    image_int_1hw = (torch.clamp(image_norm_1hw * 255, 0, 255)).byte().long()

    image_cm_3hw = cmap[image_int_1hw.flatten(start_dim=1)
    ].permute([0, 2, 1]).view([-1, h, w])

    if mask_1hw is not None:
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1).to(image_1hw.device)
        image_cm_3hw = image_cm_3hw * mask_1hw + invalid_color * (1 - mask_1hw)

    if return_vminvmax:
        return image_cm_3hw, vmin, vmax
    else:
        return image_cm_3hw


class Visualizer():
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
    
        Attributes:
            main_cfg (mmengine.Config): configurations
            vis_cfg (mmengine.Config) : visualizer model configurations
            info_printer (InfoPrinter): information printer
            
        """
        self.main_cfg = main_cfg
        self.vis_cfg = main_cfg.visualizer
        self.info_printer = info_printer

    def update_step(self, step):
        """ update step information
    
        Args:
            step (int): step size
    
        """
        self.step = step

    def visualize_occupancy_grid(self, planner, pose, uncertain_threshold):
        position = pose[:3,3]
        occupancy_map = planner.occupancy_map
        
        # occupancy_map
        position_tb = np.array([position[0], position[1], occupancy_map.origin[2] + 0.1*(occupancy_map.size[2]//2)]) 
        idx = occupancy_map.Pos2Sub(position_tb)
        bool_array = 1 - np.array(occupancy_map.data[:,:,idx[2]])
        # image_array = ( * 255).astype(np.uint8)
        image_array = np.stack([(bool_array * 255).astype(np.uint8)] * 3, axis=-1)
        
        # Show first global path in voxel
        if len(planner.voxel_global_path) > 0:
            image_array[planner.voxel_global_path[:,0],planner.voxel_global_path[:,1]] = np.array([100., 100., 255.]).astype(np.uint8)

        global_path = planner.get_viz_global_path()
        cur_global_path = planner.global_path
        
        length = min(len(global_path)-1,100)
        if length > 1:
            for i in range(length):
                idx1 = occupancy_map.Pos2Sub(global_path[i])
                idx2 = occupancy_map.Pos2Sub(global_path[i+1])
                cv2.line(image_array,[idx1[1],idx1[0]],[idx2[1],idx2[0]],color=(0,0,255),thickness=1)

        idx = occupancy_map.Pos2Sub(position)
        image_array[idx[0], idx[1]] = np.array([255., 0., 0.]).astype(np.uint8)
        
        subspace_world = planner.subspace_world
        subspaces = subspace_world.subspaces        
        for i in range(subspaces.get_cell_number()):
            cell = subspaces.cells[i]
            center = cell.get_position()
            idx2 = occupancy_map.Pos2Sub(center)
            if occupancy_map.in_range(idx2):
                if cell.status.value == 0: # UNSEEN
                    image_array[idx2[0], idx2[1]] = np.array([0., 0., 255.]).astype(np.uint8)
                elif cell.status.value == 1: # EXPLORING
                    image_array[idx2[0], idx2[1]] = np.array([255., 0., 0.]).astype(np.uint8)
                elif cell.status.value == 2: # COVERED
                    image_array[idx2[0], idx2[1]] = np.array([0., 0., 0.]).astype(np.uint8)

        path = planner.path
        for i in range(len(path)-1):
            idx1 = occupancy_map.Pos2Sub(path[i][:3,3])
            idx2 = occupancy_map.Pos2Sub(path[i+1][:3,3])
            cv2.line(image_array,[idx1[1],idx1[0]],[idx2[1],idx2[0]],color=(0,255,0),thickness=1)
            
        # if planner.gp_planner_goal:
        goal = planner.goal
        idx2 = occupancy_map.Pos2Sub(goal[0])
        image_array[idx2[0], idx2[1]] = np.array([255., 0., 255.]).astype(np.uint8)

        # sdf map
        # sdf_array = sdf_array /  # normalize
        # sdf_array = np.minimum(sdf_array, 1.0)
        # sdf_array = np.stack([(sdf_array * 255).astype(np.uint8)] * 3, axis=-1)
        # sdf_array[idx[0], idx[1]] = np.array([255., 0., 0.]).astype(np.uint8)
        
        # uncertainty map
        u_array = occupancy_map.get_uncertainty_image(uncertain_threshold, idx[2])
        # u_array = occupancy_map.uncertainty_data[:,:,idx[2]]
        # u_array = np.array(u_array) / uncertain_threshold
        # u_array = 1 - np.minimum(u_array, 1)
        # u_array = np.stack([(u_array * 255).astype(np.uint8)] * 3, axis=-1)
        # idx = occupancy_map.Pos2Sub(position)

        # Show current position on uncertainty image
        u_array[idx[0], idx[1]] = np.array([255., 0., 0.]).astype(np.uint8)
        
        # Show connected subgrids
        grid_graph = planner.subspace_world.subspace_graph
        ind = planner.subspace_world.subspaces.Pos2Ind(position)
        goal_ind = planner.subspace_world.subspaces.Pos2Ind(goal[0])        
        # print()
        # print("pos: ", position, "ind", ind)
        # print()
        
        # Draw disconnected path
        # for i in range(grid_graph.cell_number):
        #     pos_i = grid_graph.positions[i]
        #     sub_i = occupancy_map.Pos2Sub(pos_i).tolist()
        #     if not occupancy_map.in_range(sub_i):
        #         continue
        #     for j in range(len(grid_graph.lst_connected[i])):
        #         if not grid_graph.lst_connected[i][j]:
        #             neighbor_idx = grid_graph.neighbor_idx[i][j]
        #             pos_j = grid_graph.positions[neighbor_idx]
        #             sub_j = occupancy_map.Pos2Sub(pos_j).tolist()
        #             if not occupancy_map.in_range(sub_j):
        #                 continue
        #             raycast_cells = raycast(sub_i, sub_j)
        #             for ray_sub in raycast_cells:
        #                 u_array[ray_sub[0], ray_sub[1]] = np.array([255., 255., 0.]).astype(np.uint8)
            # grid_graph.lst_connected[i]
        
        for i in range(len(grid_graph.neighbor_idx[ind])):
            if grid_graph.lst_connected[ind][i]:
                # center = subspaces.cells[i].center
                center = grid_graph.positions[grid_graph.neighbor_idx[ind][i]]
                sub = occupancy_map.Pos2Sub(center)
                if not occupancy_map.in_range(sub):
                    continue
                # if (not sub[2] == idx[2]):
                #     continue
                u_array[sub[0], sub[1]] = np.array([0., 255., 0.]).astype(np.uint8)
        
        _, astar_path = grid_graph.astar_search(ind, goal_ind)
        for node in astar_path:
            sub = occupancy_map.Pos2Sub(grid_graph.positions[node])
            try:
                u_array[sub[0], sub[1]] = np.array([0., 0., 255.]).astype(np.uint8)
            except:
                print()
        # Frontier map
        f_cluster_pos = planner.cluster_frontiers
        f_subs = occupancy_map.frontier_indices
        f_array = np.stack([(np.ones(shape=bool_array.shape) * 255).astype(np.uint8)] * 3, axis=-1)
        if len(f_subs) > 0:
            # print("Frontier Exists")
            for sub in f_subs:
                # print("[Frontier]: ", sub[0], sub[1])
                f_array[sub[0], sub[1]] = np.array([255., 0., 0.]).astype(np.uint8)
        for cluster_pos in f_cluster_pos:
            sub = occupancy_map.Pos2Sub(cluster_pos)
            f_array[sub[0], sub[1]] = np.array([0., 0., 255.]).astype(np.uint8)

        r_array = np.stack([(np.ones(shape=bool_array.shape) * 255).astype(np.uint8)] * 3, axis=-1)
        for region in planner.subspace_world.region_graph.vertices.values():
            # region_indices = region.unocc_indices
            region_indices = region.unocc_indices[region.unocc_indices[:,2] == region.z_ref_idx]
            if len(region_indices) > 0:
                r_array[region_indices[:,0], region_indices[:,1]] = region.color.astype(np.uint8)
            # r_occ_indices = region.occ_indices[region.occ_indices[:,2] == idx[2]]
            r_occ_indices = region.occ_indices[region.occ_indices[:,2] == region.z_ref_idx]
            if len(r_occ_indices) > 0:
                r_array[r_occ_indices[:,0], r_occ_indices[:,1]] = np.array([0., 0., 0.]).astype(np.uint8)
            
            center = region.center
            sub = occupancy_map.Pos2Sub(center)
            r_array[sub[0], sub[1]] = np.array([255., 255., 255.]).astype(np.uint8)

        # Draw connected path
        region_graph = planner.subspace_world.region_graph
        for i in region_graph.vertices.keys():
            region = region_graph.vertices[i]
            pos_i = region.center
            sub_i = occupancy_map.Pos2Sub(pos_i).tolist()
            for j in range(len(region_graph.dct_neighbor_idx[i])):
                if region_graph.dct_connected[i][j]:
                    neighbor_path = region_graph.dct_neighbor_path[i][j]
                    # neighbor_idx = region_graph.dct_neighbor_idx[i][j]
                    # pos_j = region_graph.vertices[neighbor_idx].center
                    # sub_j = occupancy_map.Pos2Sub(pos_j).tolist()
                    # raycast_cells = raycast(sub_i, sub_j)
                    for ray_sub in neighbor_path:
                        r_array[ray_sub[0], ray_sub[1]] = np.array([255., 255., 0.]).astype(np.uint8)
            # grid_graph.lst_connected[i]            
        
        image_array = np.hstack((image_array, u_array, f_array, r_array))
        return image_array

    def visualize_rgbd(self,
                       rgb       : torch.Tensor,
                       depth     : torch.Tensor,
                       max_depth : float = 100.,
                       vis_size  : int = 320,
                       return_vis: bool = False
                       ) -> Union[None, np.ndarray]:
        """ visualiz RGB-D 
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
            max_depth (float)          : maximum depth value
            vis_size (int)             : image size used for visualization
            return_vis (bool)          : return visualization (OpenCV format) if True

        Returns:
            Union: 
                - image (np.ndarray, [H,W,3]): RGB-D visualization if return_vis
        """
        ## process RGB ##
        rgb = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (vis_size, vis_size))

        ### process Depth map ###
        depth = depth.unsqueeze(0)
        mask = (depth < max_depth) * 1.0
        depth_colormap = colormap_image(depth, mask)
        depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
        depth_colormap = cv2.resize(depth_colormap, (vis_size, vis_size))

        ### display RGB-D ###
        image = np.hstack((rgb, depth_colormap))

        ### return visualization ###
        if return_vis:
            return image
        else:
            cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB-D', image)
            key = cv2.waitKey(1)
