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


import marching_cubes as mcubes
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import trimesh
from third_parties.coslam.utils import getVoxels, get_batch_query_fn
from third_parties.coslam.model.utils import get_masks
import torch.nn.functional as F

@torch.no_grad()
def query_point_sdf(query_fn, query_pts, config, bounding_box):
    """
    Args:
        query_fn (): sdf query function
        query_pts (torch.Tensor, [N,K,3])
        config (): 
        bounding_box ():
    Returns:
        sdf (torch.Tensor, [N,K,1])
    """
    query_pts = (query_pts - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])
    embed = query_fn(query_pts, embed=True)
    embed = torch.sum(torch.abs(embed), dim=2)
    
    if config['decoder']['pred_uncert'] or config['decoder']['uncert_grid']:
        sdf = query_fn(query_pts, embed=False, return_uncert=True)
        sdf, uncert = sdf[..., 0], sdf[..., 1]
    else:
        sdf = query_fn(query_pts, embed=False)
    
    return sdf

@torch.no_grad()
def get_map_volumes(query_fn, bounding_box, voxel_size, grid_type='uncert'):
    """ get map data (Uncertainty Volume and SDF volume)

    Args:
        query_fn                           : query function
        bounding_box (torch.Tensor, [3, 2]): bounding box
        voxel_size (float)                 : voxel size

    Returns:
        map_vols (List):
        - uncert_vol (torch.Tensor, [X,Y,Z]): Uncertainty Volume
        - sdf_vol (torch.Tensor, [X,Y,Z])   : SDF Volume
    """
    ##################################################
    ### get query pts
    ##################################################
    x_min, y_min, z_min = bounding_box[:, 0]
    x_max, y_max, z_max = bounding_box[:, 1]

    tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size)
    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32).to(bounding_box.device)

    query_pts = (query_pts - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])

    ##################################################
    ### query SDF and Uncertainty
    ##################################################
    embed = query_fn(query_pts, embed=True)
    embed = torch.sum(torch.abs(embed), dim=3)
    
    sdf = query_fn(query_pts, embed=False, return_uncert=True)
    if grid_type == 'uncert':
        sdf, uncert = sdf[..., 0], sdf[..., 1]
        uncert_map = torch.nn.functional.softplus(uncert) + 0.01
    elif grid_type == 'prior':
        sdf, alpha, beta = sdf[..., 0], sdf[..., 1], sdf[..., 2]
        uncert_map = entropy(alpha, beta, 2*alpha)
        uncert_map = F.softplus(uncert_map) + 0.01 

    return [uncert_map.cpu().numpy().copy(), sdf.cpu().numpy().copy()]


@torch.no_grad()
def extract_mesh(
        query_fn, 
        config, 
        bounding_box, 
        marching_cube_bound=None, 
        color_func = None, 
        voxel_size=None, 
        resolution=None, 
        isolevel=0.0, 
        scene_name='', 
        mesh_savepath='',
        render_uncert=True,
        ) -> trimesh.Trimesh:
    '''
    Extracts mesh from the scene model using marching cubes (Adapted from NeuralRGBD)
    '''
    # Query network on dense 3d grid of points
    if marching_cube_bound is None:
        marching_cube_bound = bounding_box

    x_min, y_min, z_min = marching_cube_bound[:, 0]
    x_max, y_max, z_max = marching_cube_bound[:, 1]

    tx, ty, tz = getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size, resolution)
    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)

    
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])
    bounding_box_cpu = bounding_box.cpu()

    if config['grid']['tcnn_encoding']:
        flat = (flat - bounding_box_cpu[:, 0]) / (bounding_box_cpu[:, 1] - bounding_box_cpu[:, 0])

    fn = get_batch_query_fn(query_fn, device=bounding_box.device)

    chunk = 1024 * 64
    raw = [fn(flat, i, i + chunk).cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    

    print('Running Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(raw.squeeze(), isolevel, truncation=3.0)
    print('done', vertices.shape, triangles.shape)

    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    tx = tx.cpu().data.numpy()
    ty = ty.cpu().data.numpy()
    tz = tz.cpu().data.numpy()
    
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / config['data']['sc_factor'] - config['data']['translation']


    if color_func is not None and not config['mesh']['render_color']:
        if config['grid']['tcnn_encoding']:
            vert_flat = (torch.from_numpy(vertices).to(bounding_box) - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])


        fn_color = get_batch_query_fn(color_func, 1)

        chunk = 1024 * 64
        raw = [fn_color(vert_flat,  i, i + chunk).cpu().data.numpy() for i in range(0, vert_flat.shape[0], chunk)]

        sh = vert_flat.shape
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        color = np.reshape(raw, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)
    
    elif color_func is not None and config['mesh']['render_color']:
        print('rendering surface color')
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        vertex_normals = torch.from_numpy(mesh.vertex_normals)
        fn_color = get_batch_query_fn(color_func, 2, device=bounding_box.device)
        raw = [fn_color(torch.from_numpy(vertices), vertex_normals,  i, i + chunk).cpu().data.numpy() for i in range(0, vertices.shape[0], chunk)]

        sh = vertex_normals.shape
        
        raw = np.concatenate(raw, 0).astype(np.float32)
        color = np.reshape(raw, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)
    
    elif render_uncert:
        # print('rendering surface uncertainty')
        if config['grid']['tcnn_encoding']:
            vert_flat = (torch.from_numpy(vertices).to(bounding_box) - bounding_box[:, 0]) / (bounding_box[:, 1] - bounding_box[:, 0])
        
        ### get query function for uncertainty ###
        ### get uncertainty ###
        if config['decoder']['pred_uncert'] or config['decoder']['uncert_grid']:
            fn_uncert = lambda f, i0, i1: query_fn(f[i0:i1, None, :].to(bounding_box.device), return_uncert=True)[:, 0, 1] # query function for uncertainty
            raw_uncert = [fn_uncert(vert_flat,  i, i + chunk).cpu().data.numpy() for i in range(0, vert_flat.shape[0], chunk)]
            raw_uncert = np.concatenate(raw_uncert, 0).astype(np.float32)
        elif config['decoder']['prior_grid']:
            fn_uncert = lambda f, i0, i1: query_fn(f[i0:i1, None, :].to(bounding_box.device), return_uncert=True)[:, 0, 1:] # query function for uncertainty
            alpha_beta = [fn_uncert(vert_flat,  i, i + chunk) for i in range(0, vert_flat.shape[0], chunk)]
            alpha_beta = torch.cat(alpha_beta, 0)
            alpha, beta = alpha_beta[:, 0], alpha_beta[:, 1]
            raw_uncert = entropy(alpha, beta, 2*alpha).detach().cpu().numpy()

        sh = vert_flat.shape

        ### colorize mesh with uncertainty ###
        ## relative uncertainty ##
        # uncert_normalized = (raw_uncert - raw_uncert.min()) / (raw_uncert.max() - raw_uncert.min())
        
        uncert_normalized = np.clip(raw_uncert - raw_uncert.min(), 1e-4, None) / np.clip(raw_uncert.max() - raw_uncert.min(), 1e-4, None)  # To avoid strange unc color in step 0
        ## absolute uncertainty ##
        # uncert_normalized = np.clip(raw_uncert, 0, 3) / 3
        colormap = plt.get_cmap('jet')
        uncert_colored = colormap(uncert_normalized.flatten())[:, :3]  # Discard alpha channel

        color = np.reshape(uncert_colored, list(sh[:-1]) + [-1])
        mesh = trimesh.Trimesh(vertices, triangles, process=False, vertex_colors=color)

    else:
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

    
    os.makedirs(os.path.split(mesh_savepath)[0], exist_ok=True)
    mesh.export(mesh_savepath)

    print('Mesh saved')
    return mesh

def get_sdf_evidential_loss(z_vals, target_d, predicted_sdf, truncation, alpha, beta):
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation)
    
    y = predicted_sdf * front_mask
    x = torch.ones_like(predicted_sdf) * front_mask
    fs_nll_loss = -(expected_log_likelihood(x, y, alpha, beta, 2*alpha)*front_mask).mean() * fs_weight
    fs_reg_loss = (entropy(alpha, beta, 2*alpha)*front_mask).mean() * fs_weight
        
    y = (z_vals + predicted_sdf * truncation) * sdf_mask
    x = target_d * sdf_mask
    sdf_nll_loss = -(expected_log_likelihood(x, y, alpha, beta, 2*alpha)*sdf_mask).mean() * sdf_weight
    sdf_reg_loss = (entropy(alpha, beta, 2*alpha)*sdf_mask).mean() * sdf_weight
        
    return fs_nll_loss, fs_reg_loss, sdf_nll_loss, sdf_reg_loss

def expected_log_likelihood(data, mu, alpha, beta, lambd):
    const = -torch.log(2 * np.pi * torch.ones_like(data))
    scaled_mse = alpha / (beta+1e-9) * (mu - data) ** 2
    diff = alpha.digamma() - (beta+1e-9).log() - (lambd+1e-9).reciprocal()
    return 0.5 * (const - scaled_mse + diff)

def entropy(alpha, beta, lambd):
    t1 = 0.5 + 0.5 * torch.log(2 * np.pi * torch.ones_like(alpha))
    t2_exact = alpha.lgamma() + alpha - (alpha + 1.5) * alpha.digamma()
    t2_approx = t1 - 2 * (alpha+1e-9).log()
    t2 = torch.where(alpha >= 10000, t2_approx, t2_exact)
    t3 = 1.5 * (beta+1e-9).log() - 0.5 * (lambd+1e-9).log()
    return t1 + t2 + t3

def clamp_preserve_gradients(x, lower, upper):
    return x + (x.clamp(min=lower, max=upper) - x).detach()

def calc_ause(unc_vec, err_vec, err_type='rmse'):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae' or err_type == 'mse':
            ause_err.append(err_slice.mean().cpu().numpy())
       

    ###########################################

    # Sort by variance
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []
    for r in ratio_removed:
        
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae'or err_type == 'mse':
            ause_err_by_var.append(err_slice.mean().cpu().numpy())
    
    #Normalize and append
    max_val = max(max(ause_err), max(ause_err_by_var))#max(ause_err)
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)
    
    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)
    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)
    return ratio_removed, ause_err, ause_err_by_var, ause

def plot_errors(ratio_removed, ause_err, ause_err_by_var, err_type, step, output_path): #AUSE plots, with oracle curve also visible
    plt.plot(ratio_removed, ause_err, '--')
    plt.plot(ratio_removed, ause_err_by_var, '-r')
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'plot_'+err_type+'_'+str(step)+'.png'))
    plt.clf()
    
def trimesh_to_open3d(src):
    import open3d as o3d
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst