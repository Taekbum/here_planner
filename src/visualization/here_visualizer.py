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
import os
import torch
from typing import List

from src.slam.coslam.coslam import CoSLAMHere as CoSLAM

from .visualizer import Visualizer
from src.utils.general_utils import InfoPrinter
import matplotlib.pyplot as plt

class HEREVisualizer(Visualizer):
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
        super(HEREVisualizer, self).__init__(main_cfg, info_printer)

        keys = ['rgbd', 'pose', 'state', 'occ_map', 'goal', 'color_mesh', 'uncert_mesh']
        ### create directory ###
        for key in keys:
            if self.vis_cfg.get(f"save_{key}", False):
                vis_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", key)
                os.makedirs(vis_dir, exist_ok=True)

        ### write remark ###
        with open(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "README.md"), 'w') as f:
            f.writelines("rgbd: RGB-D visualization\n")
            f.writelines("poses: (np.ndarray, [4,4]). Camera-to-world. RUB system. \n")
            f.writelines("state: (str), planner state \n")
            f.writelines("occ_map: occupancy map visualization \n")
            f.writelines("goal: (np.ndarray), intermediate goals \n")
            f.writelines("color_mesh: texture mesh \n")
            f.writelines("uncert_mesh: uncertainty mesh \n")
            

    def main(self, 
             slam           : CoSLAM,
             planner,
             color          : torch.Tensor,
             depth          : torch.Tensor,
             pose            : np.ndarray,
             ) -> None:
        """ save data for visualization purpose
    
        Args:
            slam           : SLAM module
            planner        : Planner module
            color          : [H,W,3], color image. Range  : 0-1
            depth          : [H,W,3], depth image.
            pose           : [4,4],   current pose. Format: camera-to-world, RUB system
            
        Returns:
            
    
        Attributes:
            
        """
        ### RGB-D ###
        if self.vis_cfg.save_rgbd:
            self.info_printer("Saving RGBD for visualization", self.step, self.__class__.__name__)
            self.save_rgbd(color, depth)
        
        ### pose ###
        if self.vis_cfg.save_pose:
            self.info_printer("Saving pose for visualization", self.step, self.__class__.__name__)
            self.save_pose(pose)
            
        ### state ###
        if self.vis_cfg.save_state:
            self.info_printer("Saving state for visualization", self.step, self.__class__.__name__)
            self.save_state(planner)
            
        ### Occupancy Map ###
        if self.vis_cfg.save_occ_map:
            self.info_printer("Saving 2D occupancy map for visualization", self.step, self.__class__.__name__)
            self.save_2d_occupancy_map(planner, pose)

        ### Goal ###
        if self.vis_cfg.save_goal:
            self.info_printer("Saving goal for visualization", self.step, self.__class__.__name__)
            self.save_goal(planner)
        
        ### color_mesh ###
        if self.vis_cfg.save_color_mesh:
            self.info_printer("Saving color mesh for visualization", self.step, self.__class__.__name__)
            self.save_color_mesh(slam)
        
        ### uncert_mesh ###
        if self.vis_cfg.save_uncert_mesh:
            self.info_printer("Saving uncertainty mesh for visualization", self.step, self.__class__.__name__)
            self.save_uncert_mesh(slam, planner)
        return
    
    def save_2d_occupancy_map(self, planner, pose):
        if planner.binary_occupancy_map is not None:
            filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "occ_map", f"{self.step:04}.png")
            image_array = self.visualize_occupancy_grid(planner, pose, self.main_cfg.tare.uncertain_threshold)
            image_array = image_array.repeat(5, axis=0).repeat(5,axis=1)
            cv2.imwrite(filepath, image_array)
    
    def save_goal(self, planner):
        goal = planner.goal
            
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "goal", f"{self.step:04}.npy")
        if type(goal) == tuple:
            goal = goal[0]
        np.save(filepath, goal)
        
    def save_rgbd(self, 
                  color: torch.Tensor, 
                  depth: torch.Tensor
                  ) -> None:
        """save RGB-D visualization
    
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
    
        """
        rgbd_vis = self.visualize_rgbd(color, depth, return_vis=True)
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "rgbd", f"{self.step:04}.png")
        rgbd_vis = (rgbd_vis * 255).astype(np.uint8)
        cv2.imwrite(filepath, rgbd_vis)

    def save_pose(self, pose: np.ndarray) -> None:
        """ save pose
    
        Args:
            pose: [4,4], current pose. Format: camera-to-world, RUB system
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "pose", f"{self.step:04}.npy")
        np.save(filepath, pose)
    
    def save_state(self, planner) -> None:
        """ save planner state
    
        Args:
            planner: planner module
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "state", f"{self.step:04}.txt")
        with open(filepath, 'w') as f:
            f.writelines(f"{planner.state}")
        
    def save_color_mesh(self, slam: CoSLAM) -> None:
        """ save colored mesh
    
        Args:
            slam: SLAM module
        """
        if self.step % self.vis_cfg.save_mesh_freq == 0:
            mesh_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "color_mesh")
            slam.save_mesh(self.step, voxel_size=self.vis_cfg.save_mesh_voxel_size, suffix='', mesh_savedir=mesh_dir)
        else:
            return
    
    def save_uncert_mesh(self, slam: CoSLAM, planner) -> None:
        """ save uncertainty mesh
    
        Args:
            slam: SLAM module
        """
        if self.step % self.vis_cfg.save_mesh_freq == 0:
            mesh_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "uncert_mesh")
            slam.save_uncert_mesh(self.step, voxel_size=self.vis_cfg.save_mesh_voxel_size, suffix='', mesh_savedir=mesh_dir)
        else:
            return