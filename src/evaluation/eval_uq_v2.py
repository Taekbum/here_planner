import argparse
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import open3d as o3d
import sys, os
sys.path.insert(0, os.getcwd())

from src.here.cfg_loader import load_cfg
from src.slam.coslam.coslam import CoSLAMHere as CoSLAM
from src.utils.general_utils import InfoPrinter, update_results_file


def as_mesh(scene_or_mesh):
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

##################################################
### argparse
##################################################
def parse_args():
    parser = argparse.ArgumentParser(
            description="Arguments to the active sim slam."
        )
    parser.add_argument("--cfg", type=str,
                        help="experiement config file path")
    parser.add_argument("--ckpt_dir", type=str,
                        help="directory which contains ckpt files")
    parser.add_argument("--result_dir", type=str,
                        help="result directory")
    parser.add_argument("--result_txt", type=str,
                        help="result txt to record mean of AUSE")
    parser.add_argument("--gt_mesh", type=str)  
    parser.add_argument("--visualize", action="store_true") 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    result_dir = os.path.join(args.result_dir, 'eval_uq')
    os.makedirs(result_dir, exist_ok=True)
    info_printer = InfoPrinter("HERE", 0)
    main_cfg = load_cfg(args)
    num_iter = main_cfg.general.num_iter
    mesh_savedir = os.path.join(main_cfg.dirs.result_dir, 'coslam', "mesh")
    scene = main_cfg.dirs.result_dir.split("/")[-3]
    
    gt_mesh = trimesh.load(args.gt_mesh, process=False)
    gt_mesh = as_mesh(gt_mesh)

    slam = CoSLAM(main_cfg, info_printer)
    vis_iter = slam.config['mesh']['vis']

    if args.visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name = 'eval_uq_3d', 
            width=1024, height=1024
        )
    else:
        vis = None

    for step in range(vis_iter, num_iter + vis_iter, vis_iter): # self.config['mesh']['vis']: visualization frequency. Now: 500
        suffix = '_final' if step == num_iter else '' #if step == num_iter else ''
        ckpt_file = os.path.join(args.ckpt_dir, f'ckpt_{step:04}{suffix}.pt')
        slam.load_ckpt(ckpt_file)

        mesh_name = f"mesh_{step:04}{suffix}.ply" if step is not None else f"mesh_{step:04}{suffix}.ply"
        mesh_savepath = os.path.join(mesh_savedir, mesh_name)
        pred_mesh = trimesh.load(mesh_savepath, process=False)
        
        slam.eval_uq_v2(step, gt_mesh, pred_mesh, result_dir, vis=vis)
    
    if args.visualize:
        vis.destroy_window()
    
    np.save(os.path.join(result_dir, "ause_3d.npy"), np.array(slam.metric_ause_3d))
    ause_3d_mean = np.array(slam.metric_ause_3d).mean()
    print(f"AUSE_3d: {ause_3d_mean}")
    result = {"AUSE_3d": ause_3d_mean}
    if os.path.exists(args.result_txt):
        update_results_file(result, args.result_txt)
    else:
        print("failed to record AUSE mean!")
    
    ause_3d_png = os.path.join(result_dir, "ause_3d.png")
    
    x = np.arange(vis_iter, num_iter + vis_iter, vis_iter)
    plt.plot(x, np.array(slam.metric_ause_3d))
    plt.title("AUSE_3d")
    plt.savefig(ause_3d_png)
    plt.clf()
    
    