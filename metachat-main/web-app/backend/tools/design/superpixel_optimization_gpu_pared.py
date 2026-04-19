import os
import torch
import matplotlib.pyplot as plt
import numpy as np
EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from stratton_chu_ceviche_rectangle_torch import strattonChu2D_batched

import consts_hr as consts
import argparse
from tqdm import tqdm
import sys, os, datetime
import os
import random
from models import ParallelHashModelWrapper

import multiprocessing
import json
import time
from datetime import datetime
import gc

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from phys import Hz_to_Ex, Hz_to_Ey

import yaml
import torch.nn.functional as F

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
cmap_blue = LinearSegmentedColormap.from_list("custom1", [(0,0,0), (0,0xf7/255,1)])
cmap_red = LinearSegmentedColormap.from_list("custom3", [(0,0,0), (1,0,0)])

inch_to_pt = 72.
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 8,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'text.usetex': False,
   'figure.figsize': [240 / inch_to_pt, 200 / inch_to_pt],
   'pdf.fonttype': 42,
#    'font.sans-serif': "Arial",
#    'font.family': "Arial",
   'lines.linewidth': 1
   }
plt.rcParams.update(params)

torch.set_default_dtype(torch.float64)

# Paths to models
checkpoint_directory_multisrc = os.getenv('CHECKPOINT_DIRECTORY_MULTISRC', '/media/tmp1/metachat/metachat_code/waveynet')
learners_directory = os.path.join(checkpoint_directory_multisrc, 'source_code')
sys.path.append(learners_directory)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.getenv('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:49000')

def get_default_args():
    parser = argparse.ArgumentParser(description="Topology optimization for photonic devices using PyTorch with hash table encoding")
    parser.add_argument("--config", nargs="?", default="config.json", help="JSON config for hash table")
    parser.add_argument("--feature_weight", nargs="?", type=float, default=1.5e3, help="Feature weight")
    parser.add_argument("--device", nargs="?", default="cuda", help="Device to execute")
    return parser.parse_args([])

def merge_args_and_config(args, config):
    merged = vars(args)
    merged.update(config)
    return argparse.Namespace(**merged)

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)


def buildDielectricDistribution(full_pattern, full_grid_shape, optimization_bounds):
    eps_r = torch.ones(full_grid_shape)
    x_lower, x_upper, y_lower, y_upper = optimization_bounds
    eps_r[x_lower : x_upper, y_lower : y_upper] = full_pattern
    eps_r[:, 0:y_lower] = consts.n_sub**2
    return eps_r

def buildDielectricDistributionParallel(full_pattern_batch, full_grid_shape, optimization_bounds):
    """
    Builds dielectric distribution for a batch of patterns in parallel
    
    Args:
        full_pattern_batch: Tensor of shape [batch_size, pattern_x, pattern_y]
        full_grid_shape: Tuple of (grid_x, grid_y)
        optimization_bounds: List [x_lower, x_upper, y_lower, y_upper]
    
    Returns:
        Tensor of shape [batch_size, grid_x, grid_y]
    """
    batch_size = full_pattern_batch.shape[0]
    device = full_pattern_batch.device

    eps_r = torch.ones((batch_size,) + full_grid_shape, device=device)
    x_lower, x_upper, y_lower, y_upper = optimization_bounds
    eps_r[:, x_lower:x_upper, y_lower:y_upper] = full_pattern_batch
    eps_r[:, :, 0:y_lower] = consts.n_sub**2
    
    return eps_r

def forward_sim_nn(model, scaling_factors, eps_r_batch, wavelength, incidence_angle_deg=0, source_phase_offset=0):
    device = eps_r_batch.device
    B, Nx, Ny = eps_r_batch.shape
    omega = 2 * np.pi * C_0 / wavelength
    k0 = 2 * np.pi / wavelength

    angle_rad = (90 - incidence_angle_deg) * np.pi / 180
    kx = k0 * np.cos(angle_rad)
    ky = k0 * np.sin(angle_rad)

    source_amp = 64e9 / consts.dL**2
    Lx = consts.Nx * consts.dL * 1e-9
    x_vec = torch.linspace(-Lx/2, Lx/2, Nx, device=device)
    source_amp_x = torch.exp(1j * kx * x_vec)
    source_amp_x = source_amp_x.unsqueeze(0).expand(B, Nx)

    fwd_src = torch.zeros((B, Nx, Ny), dtype=torch.cfloat, device=device)
    fwd_src[:, consts.pml_x:-consts.pml_x, consts.pml_y+consts.spacing-4] = source_amp * source_amp_x[:, consts.pml_x:-consts.pml_x]
    fwd_src[:, consts.pml_x:-consts.pml_x, consts.pml_y+consts.spacing-4] *= np.exp(1j*source_phase_offset)

    x_start, x_end = int(consts.Nx/2-consts.image_sizex/2), int(consts.Nx/2-consts.image_sizex/2)+consts.image_sizex
    y_start, y_end = consts.y_start, consts.y_end
    center_struct_x = int((x_start+x_end)/2)
    center_struct_y = int((y_start+y_end)/2)

    left_pix = center_struct_x-consts.nn_x_pix//2
    right_pix = center_struct_x+consts.nn_x_pix//2
    bottom_pix = center_struct_y-consts.nn_y_pix//2
    top_pix = center_struct_y+consts.nn_y_pix//2

    eps_r_nn = eps_r_batch[:, left_pix:right_pix, bottom_pix:top_pix].permute(0,2,1).unsqueeze(1).float()
    fwd_src_cropped = fwd_src[:, left_pix:right_pix, bottom_pix:top_pix].permute(0,2,1)
    src = torch.zeros((B, 2, consts.nn_y_pix, consts.nn_x_pix), dtype=torch.float32, device=device)
    src[:,0] = fwd_src_cropped.real / scaling_factors['src_data_scaling_factor']
    src[:,1] = fwd_src_cropped.imag / scaling_factors['src_data_scaling_factor']

    input_wavelengths = torch.tensor([(wavelength-scaling_factors['min_wavelength'])/(scaling_factors['max_wavelength']-scaling_factors['min_wavelength'])]*B).cuda()
    input_angles = torch.tensor([(incidence_angle_deg-scaling_factors['min_angle'])/(scaling_factors['max_angle']-scaling_factors['min_angle'])]*B).cuda()

    Hz_forward_nn = model(torch.cat((eps_r_nn, src), dim=1).cuda(), input_wavelengths.to(torch.float32), input_angles.to(torch.float32)) * scaling_factors['field_scaling_factor']

    x_shape = eps_r_nn.shape
    pattern = torch.cat((torch.ones([B,1,1,x_shape[-1]], dtype=torch.float32, device=eps_r_nn.device)*consts.n_sub**2, eps_r_nn), dim=2)

    omega = 2 * np.pi * C_0 / wavelength

    Ex_forward_nn = Hz_to_Ex(Hz_forward_nn[:,0], Hz_forward_nn[:,1], torch.zeros_like(Hz_forward_nn[:,0]), 
                             torch.zeros_like(Hz_forward_nn[:,0]), consts.dL * 1e-9, omega, pattern, consts.eps_0)
    Ey_forward_nn = Hz_to_Ey(Hz_forward_nn[:,0], Hz_forward_nn[:,1], torch.zeros_like(Hz_forward_nn[:,0]), 
                             torch.zeros_like(Hz_forward_nn[:,0]), consts.dL * 1e-9, omega, pattern, consts.eps_0)

    Hz_forward_nn = Hz_forward_nn[:,:,1:-1,1:-1]
    Hz_forward_nn_optim = Hz_forward_nn[:,0,:,:] + Hz_forward_nn[:,1,:,:]*1j
    
    Ex_forward_nn = Ex_forward_nn[:,:,1:,1:-1]
    Ex_forward_nn_optim = Ex_forward_nn[:,0,:,:] + Ex_forward_nn[:,1,:,:]*1j
    
    Ey_forward_nn = Ey_forward_nn[:,:,1:-1,1:-1]
    Ey_forward_nn_optim = Ey_forward_nn[:,0,:,:] + Ey_forward_nn[:,1,:,:]*1j

    return Hz_forward_nn_optim.transpose(1,2), -Ey_forward_nn_optim.transpose(1,2), -Ex_forward_nn_optim.transpose(1,2), fwd_src


def near_to_far(spacing, Ex, Ey, Hz, r_obs, eps_r_batch, fwd_src, wl):
    # Ex,Ey,Hz: [B, ny, nx]
    B = Ex.shape[0]
    Nx = consts.nn_x_pix - 2
    Ny = consts.nn_y_pix - 2

    xc = 0
    yc = 0
    n2f_Nx = Nx
    n2f_Ny = Ny

    sx = (n2f_Nx - 1) * consts.dL * 1e-9
    sy = (n2f_Ny - 1) * consts.dL * 1e-9

    Rx_pix = (consts.nn_x_pix - 2 * consts.nn_padding) // 2
    Ry_pix = (consts.nn_y_pix - 2 * consts.nn_padding) // 2

    Rx = Rx_pix * consts.dL * 1e-9
    Ry = Ry_pix * consts.dL * 1e-9

    theta_obs, Far_Ex_b, Far_Ey_b, Far_Hz_b = strattonChu2D_batched(
        dL=consts.dL * 1e-9,
        sx=sx,
        sy=sy,
        Nx=n2f_Nx,
        Ny=n2f_Ny,
        xc=xc,
        yc=yc,
        Rx=Rx,
        Ry=Ry,
        lambda_val=wl,
        r_obs=r_obs,
        Ex_OBJ=-Ey.transpose(-1,-2),
        Ey_OBJ=-Ex.transpose(-1,-2),
        Hz_OBJ=-Hz.transpose(-1,-2),
        N_theta=consts.N_theta,
        debug=0
    )

    return theta_obs, Far_Ex_b, Far_Ey_b, Far_Hz_b


def optimizeDualWavelengthsBestFromBatchDeflector(incidence_angle_deg, wvl1, wvl2, deflection_angle_deg1, deflection_angle_deg2, target_phase1, target_phase2, 
                                               optim_distance, batch_size, thickness, seed=None, debug=False, debug_dir_root="./debug/device", device_idx=0):
    # wvl in the unit of meter. Should be within 400~700 nm.
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    debug_dir = debug_dir_root + str(device_idx)
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    num_sections = batch_size

    args = get_default_args()
    config = load_config(args.config)
    args = merge_args_and_config(args, config)
    
    parallel_model = ParallelHashModelWrapper(args, batch_size)


    model_checkpoint = torch.load(os.path.join(checkpoint_directory_multisrc, 'best_model.pt'))
    model = model_checkpoint['model'].cuda()
    model.eval()

    with open(os.path.join(checkpoint_directory_multisrc, 'scaling_factors.yaml'), 'r') as f:
        scaling_factors = yaml.safe_load(f)
    field_scaling_factor = scaling_factors['field_scaling_factor']
    src_data_scaling_factor = scaling_factors['src_data_scaling_factor']
    min_angle = scaling_factors['min_angle']
    max_angle = scaling_factors['max_angle']
    min_wavelength = scaling_factors['min_wavelength']
    max_wavelength = scaling_factors['max_wavelength']

    scaling_factors = {
        'field_scaling_factor': field_scaling_factor,
        'src_data_scaling_factor': src_data_scaling_factor,
        'min_angle': min_angle,
        'max_angle': max_angle,
        'min_wavelength': min_wavelength,
        'max_wavelength': max_wavelength
    }
    
    FoM_history = []
    FoM_history_plot = []

    bestFoM = None
    idx_section = None
    bestPattern = None
    best_Hz_f1 = None
    best_Hz_f2 = None
    best_Far_Hz1 = None
    best_Far_Hz2 = None

    pixel_thickness = int(round(thickness * 1e9 /consts.dL))

    for i in range(consts.opt_iters):
        if i < 150:
            sdf_batch, x_batch = parallel_model.sample_points(consts.image_sizex, 0.5 ** (i/3))
        else:
            sdf_batch, x_batch = parallel_model.sample_points(consts.image_sizex, None)

        x_batch = x_batch.detach().requires_grad_(True)
        full_pattern_batch = torch.zeros((batch_size, consts.image_sizex, consts.image_sizey), device=x_batch.device)
        full_pattern_batch[:, :, 0:pixel_thickness] = x_batch.unsqueeze(-1)
        full_pattern_batch = full_pattern_batch * (consts.eps_mat - 1) + 1
        
        # Build dielectric distribution in parallel
        eps_r_batch = buildDielectricDistributionParallel(
            full_pattern_batch,
            (consts.Nx, consts.Ny),
            [consts.x_start, consts.x_end, consts.y_start, consts.y_end]
        ).cuda()

        # Forward simulations
        Hz_f1_b, Ex_f1_b, Ey_f1_b, fwd_src1_b = forward_sim_nn(model, scaling_factors, eps_r_batch, wvl1, incidence_angle_deg)
        Hz_f2_b, Ex_f2_b, Ey_f2_b, fwd_src2_b = forward_sim_nn(model, scaling_factors, eps_r_batch, wvl2, incidence_angle_deg)
        
        # Near to far
        theta_obs, Far_Ex1_b, Far_Ey1_b, Far_Hz1_b = near_to_far(consts.spacing, Ex_f1_b, Ey_f1_b, Hz_f1_b, optim_distance, eps_r_batch, fwd_src1_b, wvl1)
        theta_obs, Far_Ex2_b, Far_Ey2_b, Far_Hz2_b = near_to_far(consts.spacing, Ex_f2_b, Ey_f2_b, Hz_f2_b, optim_distance, eps_r_batch, fwd_src2_b, wvl2)

        target_angle_idx1 = round((90 - deflection_angle_deg1)/360*consts.N_theta)
        target_angle_idx2 = round((90 - deflection_angle_deg2)/360*consts.N_theta)
        
        FoM1 = torch.real(Far_Hz1_b[:, target_angle_idx1] * np.exp(-1j * target_phase1))
        FoM2 = torch.real(Far_Hz2_b[:, target_angle_idx2] * np.exp(-1j * target_phase2))

        fwhm = round(3/360*consts.N_theta)
        penalty = torch.zeros_like(FoM1)
        if target_angle_idx1 > fwhm:
            penalty += torch.sum(torch.abs(Far_Hz1_b[:, :target_angle_idx1-fwhm])) / consts.N_theta
        if target_angle_idx1 + fwhm < consts.N_theta // 2:
            penalty += torch.sum(torch.abs(Far_Hz1_b[:, target_angle_idx1+fwhm : consts.N_theta // 2])) / consts.N_theta
        if target_angle_idx2 > fwhm:
            penalty += wvl2/wvl1 * torch.sum(torch.abs(Far_Hz2_b[:, :target_angle_idx2-fwhm])) / consts.N_theta
        if target_angle_idx2 + fwhm < consts.N_theta // 2:
            penalty += wvl2/wvl1 * torch.sum(torch.abs(Far_Hz2_b[:, target_angle_idx2+fwhm : consts.N_theta // 2])) / consts.N_theta
        if abs(target_angle_idx1 - (consts.N_theta // 4)) >= fwhm:
            penalty += 0.2 * torch.abs(Far_Hz1_b[:, consts.N_theta // 4])
        if abs(target_angle_idx2 - (consts.N_theta // 4)) >= fwhm:
            penalty += wvl2/wvl1 * 0.2 * torch.abs(Far_Hz2_b[:, consts.N_theta // 4])
        
        FoM_batch = - (1. * FoM1 + wvl2/wvl1 * FoM2) + penalty
        FoM = torch.sum(FoM_batch)

        FoM_history.append(torch.min(FoM_batch).item())
        FoM_history_plot.append(FoM_batch.clone().detach().cpu().numpy())

        FoM.backward()
        gradients = x_batch.grad.clone().detach()

        for k in range(2):
            parallel_model.optimizer.zero_grad()
            sdf_batch, x_batch = parallel_model.sample_points(consts.image_sizex, None)
            loss = torch.sum(gradients * sdf_batch * 1.0 * 0.5 ** (i/10))
            loss.backward()
            parallel_model.optimizer.step()
            parallel_model.per_loop_call(i)


        # save the best device up to now
        if (i > consts.opt_iters * 0.90): # binarized enough
            if (torch.min(FoM_batch).item() < min(FoM_history[:-1])) or (bestPattern is None):
                bestFoM = torch.min(FoM_batch).item()
                idx_section = torch.argmin(FoM_batch)
                bestPattern = full_pattern_batch[idx_section].detach().cpu().numpy()
                bestFullEps = eps_r_batch[idx_section].detach().cpu().numpy()
                best_Hz_f1 = Hz_f1_b[idx_section].cpu().detach().numpy()
                best_Hz_f2 = Hz_f2_b[idx_section].cpu().detach().numpy()
                best_Far_Hz1 = Far_Hz1_b[idx_section, :].cpu().clone().detach().numpy()
                best_Far_Hz2 = Far_Hz2_b[idx_section, :].cpu().clone().detach().numpy()


        if(debug and (i+1)==consts.opt_iters):
            plt.figure(figsize=(8,4))
            plt.plot(FoM_history_plot, alpha=0.2, color='#1f77b4')
            plt.plot(np.mean(np.array(FoM_history_plot), axis=1), color='#c00200', label='Mean FoM')
            plt.xlabel('Iteration')
            plt.ylabel('FoM')
            plt.savefig(f"./{debug_dir_root}{device_idx}/FoM_history.png", bbox_inches='tight')
            plt.close()

            theta_obs = theta_obs.cpu().detach().numpy()
            
            # wavelength 1
            plt.figure(figsize=(8,4))

            x_start, x_end = int(consts.Nx/2-consts.image_sizex/2), int(consts.Nx/2-consts.image_sizex/2)+consts.image_sizex
            y_start, y_end = consts.y_start, consts.y_end
            center_struct_x = int((x_start+x_end)/2)
            center_struct_y = int((y_start+y_end)/2)

            left_pix = center_struct_x-consts.nn_x_pix//2+1
            right_pix = center_struct_x+consts.nn_x_pix//2-1
            bottom_pix = center_struct_y-consts.nn_y_pix//2+1
            top_pix = center_struct_y+consts.nn_y_pix//2-1

            # Left column: first two plots
            ax1 = plt.subplot(2,2,1)
            
            # Create a custom colormap from white to dark beige
            colors = [(1, 1, 1), (30/255, 30/255, 30/255)]  # White to dark gray (RGB 30, 30, 30)
            n_bins = 100  # Number of color gradations
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            # Set min and max values for the colormap
            min_val = 1  # Minimum epsilon (air)
            max_val = consts.eps_mat  # Maximum epsilon (material)

            im = ax1.imshow(bestFullEps[left_pix:right_pix, bottom_pix:top_pix].T, 
                            origin='lower', cmap=custom_cmap, vmin=min_val, vmax=max_val)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(r'$\varepsilon$', fontsize=16)

            # Add colorbar with custom colormap
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, ticks=[min_val, max_val])
            cbar.set_ticklabels([f'{min_val:.2f}', f'{max_val:.2f}'])
            cbar.ax.tick_params(labelsize=12)

            ax2 = plt.subplot(2,2,3)

            Hz_plot1 = np.real(best_Hz_f1.T)
            
            # Create a custom colormap with white at 0
            colors = ['blue', 'white', 'red']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
            
            vmin, vmax = np.min(Hz_plot1), np.max(Hz_plot1)
            vabs = max(abs(vmin), abs(vmax))

            im2 = ax2.imshow(Hz_plot1, cmap=custom_cmap, origin='lower', vmin=-vabs, vmax=vabs)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(r'$H_z$', fontsize=16)
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_ticks([-vabs, 0, vabs])
            cbar2.set_ticklabels([f'{-vabs:.2e}', '0', f'{vabs:.2e}'])
            cbar2.ax.tick_params(labelsize=12)
            

            ax = plt.subplot(1,2,2, projection='polar')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            plt.axvline(x=np.deg2rad(-deflection_angle_deg1), color='r')
            Hz_for_plot = np.abs(best_Far_Hz1)
            Hz_for_plot[theta_obs > np.pi] = 0
            plt.plot(theta_obs - np.pi/2, Hz_for_plot**2)
            ax.yaxis.set_ticklabels([])
            ax.set_title(f'Far field phase at the target angle: {np.angle(best_Far_Hz1[target_angle_idx1])}', fontsize=16)
            ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
            ax.tick_params(axis='both', which='major', labelsize=12)

            plt.tight_layout()
            plt.savefig(f"./{debug_dir_root}{device_idx}/section{idx_section}_wavelength1.png", dpi=200, bbox_inches='tight')
            plt.close()


            # wavelength 2
            plt.figure(figsize=(8,4))

            x_start, x_end = int(consts.Nx/2-consts.image_sizex/2), int(consts.Nx/2-consts.image_sizex/2)+consts.image_sizex
            y_start, y_end = consts.y_start, consts.y_end
            center_struct_x = int((x_start+x_end)/2)
            center_struct_y = int((y_start+y_end)/2)

            left_pix = center_struct_x-consts.nn_x_pix//2+1
            right_pix = center_struct_x+consts.nn_x_pix//2-1
            bottom_pix = center_struct_y-consts.nn_y_pix//2+1
            top_pix = center_struct_y+consts.nn_y_pix//2-1

            # Left column: first two plots
            ax1 = plt.subplot(2,2,1)
            
            # Create a custom colormap from white to dark beige
            colors = [(1, 1, 1), (30/255, 30/255, 30/255)]  # White to dark gray (RGB 30, 30, 30)
            n_bins = 100  # Number of color gradations
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            # Set min and max values for the colormap
            min_val = 1  # Minimum epsilon (air)
            max_val = consts.eps_mat  # Maximum epsilon (material)

            im = ax1.imshow(bestFullEps[left_pix:right_pix, bottom_pix:top_pix].T, 
                            origin='lower', cmap=custom_cmap, vmin=min_val, vmax=max_val)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(r'$\varepsilon$', fontsize=16)

            # Add colorbar with custom colormap
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, ticks=[min_val, max_val])
            cbar.set_ticklabels([f'{min_val:.2f}', f'{max_val:.2f}'])
            cbar.ax.tick_params(labelsize=12)

            ax2 = plt.subplot(2,2,3)

            Hz_plot2 = np.real(best_Hz_f2.T)
            
            # Create a custom colormap with white at 0
            colors = ['blue', 'white', 'red']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
            
            vmin, vmax = np.min(Hz_plot2), np.max(Hz_plot2)
            vabs = max(abs(vmin), abs(vmax))

            im2 = ax2.imshow(Hz_plot2, cmap=custom_cmap, origin='lower', vmin=-vabs, vmax=vabs)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(r'$H_z$', fontsize=16)
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_ticks([-vabs, 0, vabs])
            cbar2.set_ticklabels([f'{-vabs:.2e}', '0', f'{vabs:.2e}'])
            cbar2.ax.tick_params(labelsize=12)
            

            ax = plt.subplot(1,2,2, projection='polar')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            plt.axvline(x=np.deg2rad(-deflection_angle_deg2), color='r')
            Hz_for_plot = np.abs(best_Far_Hz2)
            Hz_for_plot[theta_obs > np.pi] = 0
            plt.plot(theta_obs - np.pi/2, Hz_for_plot**2)
            ax.yaxis.set_ticklabels([])
            ax.set_title(f'Far field phase at the target angle: {np.angle(best_Far_Hz2[target_angle_idx2])}', fontsize=16)
            ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
            ax.tick_params(axis='both', which='major', labelsize=12)

            plt.tight_layout()
            plt.savefig(f"./{debug_dir_root}{device_idx}/section{idx_section}_wavelength2.png", dpi=200, bbox_inches='tight')
            plt.close()

            np.savez(f"./{debug_dir_root}{device_idx}/data.npz", 
             Far_Hz1=best_Far_Hz1,
             Far_Hz2=best_Far_Hz2,
             pattern=bestPattern,
             FoM=bestFoM
            )

    return best_Far_Hz1, best_Far_Hz2, bestPattern, bestFoM, best_Hz_f1, best_Hz_f2


def optimizeSingleWavelengthBestFromBatchDeflector(incidence_angle_deg, wvl1, deflection_angle_deg1, target_phase1, 
                                               optim_distance, batch_size, thickness, seed=None, debug=False, debug_dir_root="./debug/device", device_idx=0):
    # wvl in the unit of meter. Should be within 400-700 nm for optimal waveynet performance--otherwise working out of distribution.
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    debug_dir = debug_dir_root + str(device_idx)
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    num_sections = batch_size

    args = get_default_args()
    config = load_config(args.config)
    args = merge_args_and_config(args, config)
    
    parallel_model = ParallelHashModelWrapper(args, batch_size)


    model_checkpoint = torch.load(os.path.join(checkpoint_directory_multisrc, 'best_model.pt'))
    model = model_checkpoint['model'].cuda()
    model.eval()
    
    with open(os.path.join(checkpoint_directory_multisrc, 'scaling_factors.yaml'), 'r') as f:
        scaling_factors = yaml.safe_load(f)
    field_scaling_factor = scaling_factors['field_scaling_factor']
    src_data_scaling_factor = scaling_factors['src_data_scaling_factor']
    min_angle = scaling_factors['min_angle']
    max_angle = scaling_factors['max_angle']
    min_wavelength = scaling_factors['min_wavelength']
    max_wavelength = scaling_factors['max_wavelength']

    scaling_factors = {
        'field_scaling_factor': field_scaling_factor,
        'src_data_scaling_factor': src_data_scaling_factor,
        'min_angle': min_angle,
        'max_angle': max_angle,
        'min_wavelength': min_wavelength,
        'max_wavelength': max_wavelength
    }
    
    FoM_history = []
    FoM_history_plot = []

    bestFoM = None
    idx_section = None
    bestPattern = None
    best_Hz_f1 = None
    best_Far_Hz1 = None

    pixel_thickness = int(round(thickness * 1e9 /consts.dL))

    for i in range(consts.opt_iters):
        if i < 150:
            sdf_batch, x_batch = parallel_model.sample_points(consts.image_sizex, 0.5 ** (i/3))
        else:
            sdf_batch, x_batch = parallel_model.sample_points(consts.image_sizex, None)

        x_batch = x_batch.detach().requires_grad_(True)
        full_pattern_batch = torch.zeros((batch_size, consts.image_sizex, consts.image_sizey), device=x_batch.device)
        full_pattern_batch[:, :, 0:pixel_thickness] = x_batch.unsqueeze(-1)
        full_pattern_batch = full_pattern_batch * (consts.eps_mat - 1) + 1
        
        # Build dielectric distribution in parallel
        eps_r_batch = buildDielectricDistributionParallel(
            full_pattern_batch,
            (consts.Nx, consts.Ny),
            [consts.x_start, consts.x_end, consts.y_start, consts.y_end]
        ).cuda()

        # Forward simulations
        Hz_f1_b, Ex_f1_b, Ey_f1_b, fwd_src1_b = forward_sim_nn(model, scaling_factors, eps_r_batch, wvl1, incidence_angle_deg)
        
        # Near to far
        theta_obs, Far_Ex1_b, Far_Ey1_b, Far_Hz1_b = near_to_far(consts.spacing, Ex_f1_b, Ey_f1_b, Hz_f1_b, optim_distance, eps_r_batch, fwd_src1_b, wvl1)

        target_angle_idx1 = round((90 - deflection_angle_deg1)/360*consts.N_theta)
        
        FoM1 = torch.real(Far_Hz1_b[:, target_angle_idx1] * np.exp(-1j * target_phase1))

        fwhm = round(3/360*consts.N_theta)
        penalty = torch.zeros_like(FoM1)
        if target_angle_idx1 > fwhm:
            penalty += torch.sum(torch.abs(Far_Hz1_b[:, :target_angle_idx1-fwhm])) / consts.N_theta
        if target_angle_idx1 + fwhm < consts.N_theta // 2:
            penalty += torch.sum(torch.abs(Far_Hz1_b[:, target_angle_idx1+fwhm : consts.N_theta // 2])) / consts.N_theta
        if abs(target_angle_idx1 - (consts.N_theta // 4)) >= fwhm:
            penalty += 0.2 * torch.abs(Far_Hz1_b[:, consts.N_theta // 4])
        
        FoM_batch = - FoM1 + penalty
        FoM = torch.sum(FoM_batch)

        FoM_history.append(torch.min(FoM_batch).item())
        FoM_history_plot.append(FoM_batch.clone().detach().cpu().numpy())

        FoM.backward()
        gradients = x_batch.grad.clone().detach()

        for k in range(2):
            parallel_model.optimizer.zero_grad()
            sdf_batch, x_batch = parallel_model.sample_points(consts.image_sizex, None)
            loss = torch.sum(gradients * sdf_batch * 1.0 * 0.5 ** (i/10))
            loss.backward()
            parallel_model.optimizer.step()
            parallel_model.per_loop_call(i)


        # save the best device up to now
        if (i > consts.opt_iters * 0.90): # binarized enough
            if (torch.min(FoM_batch).item() < min(FoM_history[:-1])) or (bestPattern is None):
                bestFoM = torch.min(FoM_batch).item()
                idx_section = torch.argmin(FoM_batch)
                bestPattern = full_pattern_batch[idx_section].detach().cpu().numpy()
                bestFullEps = eps_r_batch[idx_section].detach().cpu().numpy()
                best_Hz_f1 = Hz_f1_b[idx_section].cpu().detach().numpy()
                best_Far_Hz1 = Far_Hz1_b[idx_section, :].cpu().clone().detach().numpy()


        if(debug and (i+1)==consts.opt_iters):
            plt.figure(figsize=(8,4))
            plt.plot(FoM_history_plot, alpha=0.2, color='#1f77b4')
            plt.plot(np.mean(np.array(FoM_history_plot), axis=1), color='#c00200', label='Mean FoM')
            plt.xlabel('Iteration')
            plt.ylabel('FoM')
            plt.savefig(f"./{debug_dir_root}{device_idx}/FoM_history.png", bbox_inches='tight')
            plt.close()

            theta_obs = theta_obs.cpu().detach().numpy()
            
            # wavelength 1
            plt.figure(figsize=(8,4))

            x_start, x_end = int(consts.Nx/2-consts.image_sizex/2), int(consts.Nx/2-consts.image_sizex/2)+consts.image_sizex
            y_start, y_end = consts.y_start, consts.y_end
            center_struct_x = int((x_start+x_end)/2)
            center_struct_y = int((y_start+y_end)/2)

            left_pix = center_struct_x-consts.nn_x_pix//2+1
            right_pix = center_struct_x+consts.nn_x_pix//2-1
            bottom_pix = center_struct_y-consts.nn_y_pix//2+1
            top_pix = center_struct_y+consts.nn_y_pix//2-1

            # Left column: first two plots
            ax1 = plt.subplot(2,2,1)
            
            # Create a custom colormap from white to dark beige
            colors = [(1, 1, 1), (30/255, 30/255, 30/255)]  # White to dark gray (RGB 30, 30, 30)
            n_bins = 100  # Number of color gradations
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            # Set min and max values for the colormap
            min_val = 1  # Minimum epsilon (air)
            max_val = consts.eps_mat  # Maximum epsilon (material)

            im = ax1.imshow(bestFullEps[left_pix:right_pix, bottom_pix:top_pix].T, 
                            origin='lower', cmap=custom_cmap, vmin=min_val, vmax=max_val)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(r'$\varepsilon$', fontsize=16)

            # Add colorbar with custom colormap
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, ticks=[min_val, max_val])
            cbar.set_ticklabels([f'{min_val:.2f}', f'{max_val:.2f}'])
            cbar.ax.tick_params(labelsize=12)

            ax2 = plt.subplot(2,2,3)

            Hz_plot1 = np.real(best_Hz_f1.T)
            
            # Create a custom colormap with white at 0
            colors = ['blue', 'white', 'red']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
            
            vmin, vmax = np.min(Hz_plot1), np.max(Hz_plot1)
            vabs = max(abs(vmin), abs(vmax))

            im2 = ax2.imshow(Hz_plot1, cmap=custom_cmap, origin='lower', vmin=-vabs, vmax=vabs)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(r'$H_z$', fontsize=16)
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_ticks([-vabs, 0, vabs])
            cbar2.set_ticklabels([f'{-vabs:.2e}', '0', f'{vabs:.2e}'])
            cbar2.ax.tick_params(labelsize=12)
            

            ax = plt.subplot(1,2,2, projection='polar')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            plt.axvline(x=np.deg2rad(-deflection_angle_deg1), color='r')
            Hz_for_plot = np.abs(best_Far_Hz1)
            Hz_for_plot[theta_obs > np.pi] = 0
            plt.plot(theta_obs - np.pi/2, Hz_for_plot**2)
            ax.yaxis.set_ticklabels([])
            ax.set_title(f'Far field phase at the target angle: {np.angle(best_Far_Hz1[target_angle_idx1])}', fontsize=16)
            ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
            ax.tick_params(axis='both', which='major', labelsize=12)

            plt.tight_layout()
            plt.savefig(f"./{debug_dir_root}{device_idx}/section{idx_section}_wavelength1.png", dpi=200, bbox_inches='tight')
            plt.close()


            np.savez(f"./{debug_dir_root}{device_idx}/data.npz", 
             Far_Hz1=best_Far_Hz1,
             pattern=bestPattern,
             FoM=bestFoM
            )

        # pbar.set_description(f"step: {i}, FOM: {FoM_history[-1]:.4f}")
        # pbar.update()

    return best_Far_Hz1, bestPattern, bestFoM, best_Hz_f1


def setFeatureSize(feature_size_in_pixel):
    relative_feature_size = 2 * feature_size_in_pixel / consts.image_sizex

    # Load the JSON file
    with open('config.json', 'r') as file:
        config = json.load(file)
    
    # Update the values of min_gap and min_post
    config['model']['min_gap'] = relative_feature_size
    config['model']['min_post'] = relative_feature_size
    
    # Save the updated JSON back to the file
    with open('config.json', 'w') as file:
        json.dump(config, file, indent=4)

    return


def postOptimizationFeatureSizeCorrection(full_pattern, feature_size_in_pixel: int, thickness: float):
    # remove all small features

    pixel_thickness = int(round(thickness * 1e9 /consts.dL))
    
    pattern1d = full_pattern[:, consts.y_start]
    mean_value = np.mean(pattern1d)
    binarized = (pattern1d > mean_value).astype(int)
    
    result = binarized.copy()
    n = len(binarized)
    start = 0
    
    while start < n:
        # Find the end of current consecutive segment
        end = start
        while end < n and binarized[end] == binarized[start]:
            end += 1
            
        # Calculate length of current segment
        segment_length = end - start
        
        # Modify values based on conditions
        if segment_length < feature_size_in_pixel:
            if start == 0 or end == n:  # Segment is at boundary
                result[start:end] = 0
            else:  # Segment is not at boundary
                result[start:end] = 1 - binarized[start]
                
        # Move to next segment
        start = end

    full_pattern[:, consts.y_start:consts.y_start+pixel_thickness] = result[:, np.newaxis] * (consts.eps_mat - 1) + 1
    return full_pattern


def stitchSuperPixels(all_best_patterns, neighbor_section_distance_in_pixel: int):
    pattern = np.stack(all_best_patterns)
    spacing = neighbor_section_distance_in_pixel - pattern.shape[1]
    full_pattern = np.ones((neighbor_section_distance_in_pixel * pattern.shape[0], consts.Ny))
    full_pattern[:,:consts.y_start] = consts.n_sub ** 2
    for i in range(pattern.shape[0]):
        full_pattern[i*neighbor_section_distance_in_pixel + spacing//2:\
                     i*neighbor_section_distance_in_pixel + spacing//2 + pattern.shape[1], \
                    consts.y_start:consts.y_end] = pattern[i, :, :]
    
    return full_pattern


def plotDualWavelengthsFarField(eps_r_batch, incidence_angle_deg, wvl1, wvl2, deflection_angle_deg1, deflection_angle_deg2, optim_distance, 
                 neighbor_section_distance_in_pixel, plotDirectory):
    num_sections = eps_r_batch.shape[0]

    model_checkpoint = torch.load(os.path.join(checkpoint_directory_multisrc, 'best_model.pt'))
    model = model_checkpoint['model'].cuda()
    model.eval()
    
    with open(os.path.join(checkpoint_directory_multisrc, 'scaling_factors.yaml'), 'r') as f:
        scaling_factors = yaml.safe_load(f)
    field_scaling_factor = scaling_factors['field_scaling_factor']
    src_data_scaling_factor = scaling_factors['src_data_scaling_factor']
    min_angle = scaling_factors['min_angle']
    max_angle = scaling_factors['max_angle']
    min_wavelength = scaling_factors['min_wavelength']
    max_wavelength = scaling_factors['max_wavelength']
    
    scaling_factors = {
        'field_scaling_factor': field_scaling_factor,
        'src_data_scaling_factor': src_data_scaling_factor,
        'min_angle': min_angle,
        'max_angle': max_angle,
        'min_wavelength': min_wavelength,
        'max_wavelength': max_wavelength
    }
    
    # Forward simulations
    Hz_f1_b, Ex_f1_b, Ey_f1_b, fwd_src1_b = forward_sim_nn(model, scaling_factors, eps_r_batch, wvl1, incidence_angle_deg)
    Hz_f2_b, Ex_f2_b, Ey_f2_b, fwd_src2_b = forward_sim_nn(model, scaling_factors, eps_r_batch, wvl2, incidence_angle_deg)
    
    # Near to far
    theta_obs, Far_Ex1_b, Far_Ey1_b, Far_Hz1_b = near_to_far(consts.spacing, Ex_f1_b, Ey_f1_b, Hz_f1_b, optim_distance, eps_r_batch, fwd_src1_b, wvl1)
    theta_obs, Far_Ex2_b, Far_Ey2_b, Far_Hz2_b = near_to_far(consts.spacing, Ex_f2_b, Ey_f2_b, Hz_f2_b, optim_distance, eps_r_batch, fwd_src2_b, wvl2)

    wavelength_in_pixel1 = wvl1 * 1e9 / consts.dL
    wavelength_in_pixel2 = wvl2 * 1e9 / consts.dL
    
    # wavelength 1
    phase_by_section1 = 2 * np.pi / wavelength_in_pixel1 * neighbor_section_distance_in_pixel * \
        np.sin(np.pi/2 - theta_obs.cpu())[None,:] * np.arange(num_sections)[:,None]
    Hz_farfield_sum1 = torch.sum(Far_Hz1_b.detach().cpu() * torch.exp(-1j * phase_by_section1.detach().cpu()), axis=0)
    farfield_intensity1 = torch.abs(Hz_farfield_sum1).detach().cpu().numpy() ** 2

    # wavelength 2
    phase_by_section2 = 2 * np.pi / wavelength_in_pixel2 * neighbor_section_distance_in_pixel * \
        np.sin(np.pi/2 - theta_obs.cpu())[None,:] * np.arange(num_sections)[:,None]
    Hz_farfield_sum2 = torch.sum(Far_Hz2_b.detach().cpu() * torch.exp(-1j * phase_by_section2.detach().cpu()), axis=0)
    farfield_intensity2 = torch.abs(Hz_farfield_sum2).detach().cpu().numpy() ** 2

    
    theta_obs = theta_obs.cpu().detach().numpy()
    
    # polar plot of far field |Hz|^2 (proportional to the poynting vector)
    plt.figure(figsize=(3.5, 3.5))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    plt.axvline(x=np.deg2rad(180+incidence_angle_deg), color='k',label="Incident direction")
    
    # Plot the farfield intensity directly
    farfield_intensity1[len(theta_obs)//2:] = 0  # Zero out the bottom half
    plt.plot(theta_obs - np.pi/2, farfield_intensity1 / np.max(farfield_intensity1), label=f"{wvl1 * 1e9} nm wavelength",
            color="blue")
    farfield_intensity2[len(theta_obs)//2:] = 0  # Zero out the bottom half
    plt.plot(theta_obs - np.pi/2, farfield_intensity2 / np.max(farfield_intensity2), label=f"{wvl2 * 1e9} nm wavelength",
            color="red")
    plt.legend(loc="lower right")
    
    ax.yaxis.set_ticklabels([])
    ax.set_title('Far Field Intensity', fontsize=9)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                      ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.savefig(f'{plotDirectory}/farfield_intensity.png', dpi = 600)
    plt.close()

    # cartesian coordiantes plot
    plt.figure(figsize=(4, 3))
    plt.plot(np.rad2deg(theta_obs[theta_obs<np.pi] - np.pi/2), 
             farfield_intensity1[theta_obs<np.pi] / np.max(farfield_intensity1), label=f"{wvl1 * 1e9} nm wavelength", color="blue")
    plt.plot(np.rad2deg(theta_obs[theta_obs<np.pi] - np.pi/2), 
             farfield_intensity2[theta_obs<np.pi] / np.max(farfield_intensity2), label=f"{wvl2 * 1e9} nm wavelength", color="red")
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Angle (degree)")
    plt.ylabel("Relative Intensity (a.u.)")
    plt.title('Far Field Intensity')
    plt.savefig(f'{plotDirectory}/farfield_intensity_cartesian.png', bbox_inches='tight', dpi = 600)
    plt.close()


def plotSingleWavelengthFarField(eps_r_batch, incidence_angle_deg, wvl1, deflection_angle_deg1, optim_distance, 
                 neighbor_section_distance_in_pixel, plotDirectory):
    num_sections = eps_r_batch.shape[0]

    model_checkpoint = torch.load(os.path.join(checkpoint_directory_multisrc, 'best_model.pt'))
    model = model_checkpoint['model'].cuda()
    model.eval()
    
    with open(os.path.join(checkpoint_directory_multisrc, 'scaling_factors.yaml'), 'r') as f:
        scaling_factors = yaml.safe_load(f)
    field_scaling_factor = scaling_factors['field_scaling_factor']
    src_data_scaling_factor = scaling_factors['src_data_scaling_factor']
    min_angle = scaling_factors['min_angle']
    max_angle = scaling_factors['max_angle']
    min_wavelength = scaling_factors['min_wavelength']
    max_wavelength = scaling_factors['max_wavelength']
    
    scaling_factors = {
        'field_scaling_factor': field_scaling_factor,
        'src_data_scaling_factor': src_data_scaling_factor,
        'min_angle': min_angle,
        'max_angle': max_angle,
        'min_wavelength': min_wavelength,
        'max_wavelength': max_wavelength
    }
    
    # Forward simulations
    Hz_f1_b, Ex_f1_b, Ey_f1_b, fwd_src1_b = forward_sim_nn(model, scaling_factors, eps_r_batch, wvl1, incidence_angle_deg)
    
    # Near to far
    theta_obs, Far_Ex1_b, Far_Ey1_b, Far_Hz1_b = near_to_far(consts.spacing, Ex_f1_b, Ey_f1_b, Hz_f1_b, optim_distance, eps_r_batch, fwd_src1_b, wvl1)

    wavelength_in_pixel1 = wvl1 * 1e9 / consts.dL
    
    # wavelength 1
    phase_by_section1 = 2 * np.pi / wavelength_in_pixel1 * neighbor_section_distance_in_pixel * \
        np.sin(np.pi/2 - theta_obs.cpu())[None,:] * np.arange(num_sections)[:,None]
    Hz_farfield_sum1 = torch.sum(Far_Hz1_b.detach().cpu() * torch.exp(-1j * phase_by_section1.detach().cpu()), axis=0)
    farfield_intensity1 = torch.abs(Hz_farfield_sum1).detach().cpu().numpy() ** 2

    theta_obs = theta_obs.cpu().detach().numpy()
    
    # polar plot of far field |Hz|^2 (proportional to the poynting vector)
    plt.figure(figsize=(3.5, 3.5))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    plt.axvline(x=np.deg2rad(180+incidence_angle_deg), color='k',label="Incident direction")
    
    # Plot the farfield intensity directly
    farfield_intensity1[len(theta_obs)//2:] = 0  # Zero out the bottom half
    plt.plot(theta_obs - np.pi/2, farfield_intensity1 / np.max(farfield_intensity1), label=f"{wvl1 * 1e9} nm wavelength",
            color="blue")
    plt.legend(loc="lower right")
    
    ax.yaxis.set_ticklabels([])
    ax.set_title('Far Field Intensity', fontsize=9)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                      ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.savefig(f'{plotDirectory}/farfield_intensity.png', dpi = 600)
    plt.close()

    # cartesian coordiantes plot
    plt.figure(figsize=(4, 3))
    plt.plot(np.rad2deg(theta_obs[theta_obs<np.pi] - np.pi/2), 
             farfield_intensity1[theta_obs<np.pi] / np.max(farfield_intensity1), label=f"{wvl1 * 1e9} nm wavelength", color="blue")
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Angle (degree)")
    plt.ylabel("Relative Intensity (a.u.)")
    plt.title('Far Field Intensity')
    plt.savefig(f'{plotDirectory}/farfield_intensity_cartesian.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    

def angularSpectrumMethod(Hz_surface, grid_z, window_size: float, wavelength_um: float):
    '''
        Default length unit in this function is um
        Hz_surface is 1d array of Hz field
        grid_z is 1d array of z positions (propagation distance), unit: um
        window_size is the physical size of window of the output (x direction length), unit: um
        wavelength_um unit: um
        return  a 2d array of Hz within the window, and an x position array (unit: um)

        This function might run out of memory if the size is too large!
    '''

    factor = 5  # Downsampling factor
    
    # Calculate the maximum length that's divisible by the factor
    trimmed_length = (len(Hz_surface) // factor) * factor
    # Trim the array to ensure it's divisible by the factor
    Hz_surface_trimmed = Hz_surface[:trimmed_length]
    
    # Reshape the array into groups and calculate mean
    Hz_surface_downsampled = Hz_surface_trimmed.reshape(-1, factor).mean(axis=1)

    atom_period = consts.dL * 1e-3 * factor
    num_atom = Hz_surface_downsampled.shape[0]
    lengthx = num_atom * atom_period
    grid_x = (torch.arange(-num_atom / 2, num_atom / 2) + 0.5) * atom_period
    
    field_lens = torch.tensor(Hz_surface_downsampled)
    num_padding_each_side = 15
    field_lens = torch.nn.functional.pad(field_lens, (num_atom * num_padding_each_side, num_atom * num_padding_each_side), 
                                         mode='constant', value=0)

    field_lens_fre = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(field_lens))) * atom_period
    dfx = 1 / atom_period / (2*num_padding_each_side+1) / num_atom
    grid_fx = torch.arange((-1 / atom_period / 2 + dfx / 2), (1 / atom_period / 2 + dfx / 2 -1e-8), dfx)
    
    grid_z = torch.tensor(grid_z)
    transfer_function = torch.exp(2j * np.pi * grid_z[:, None] * torch.real(torch.sqrt(1 / wavelength_um**2 - grid_fx**2 + 0j))) * \
                        (1 / wavelength_um**2 - grid_fx**2 >= 0)
    field_fre = transfer_function * field_lens_fre
    field = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(field_fre), dim=-1)) * dfx * len(field_lens_fre)

    grid_x = (torch.arange(-(2*num_padding_each_side+1) * num_atom / 2, 
                           (2*num_padding_each_side+1) * num_atom / 2 - 1e-8) + 0.5) * atom_period

    window_size_in_pixel = window_size / atom_period
    return field[:, int((field.shape[1]-window_size_in_pixel)/2):int((field.shape[1]+window_size_in_pixel)/2)].numpy(), \
        grid_x[int((field.shape[1]-window_size_in_pixel)/2):int((field.shape[1]+window_size_in_pixel)/2)].numpy()


def optimize_dual_metalens_superpixel(args):
    """Helper function to optimize a single dual wavelength metalens superpixel on specified GPU"""
    i, gpu_id, params = args
    try:
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cleanup_gpu()
        consts.eps_mat = params.get('eps_mat', consts.eps_mat)
        
        superpixel_x = params['neighbor_section_distance_in_pixel'] * (np.arange(params['num_superpixel']) - params['num_superpixel'] * 0.5 + 0.5)
        focal_point_angle_deg1 = - np.rad2deg(np.arctan2(params['focal_x_offset1'] - superpixel_x, params['focal_length1']))
        focal_point_angle_deg2 = - np.rad2deg(np.arctan2(params['focal_x_offset2'] - superpixel_x, params['focal_length2']))
        phase_by_section1 = - 2 * np.pi / params['wavelength_in_pixel1'] * np.sqrt((params['focal_x_offset1'] - superpixel_x)**2 + params['focal_length1']**2)
        phase_by_section2 = np.random.uniform(low=0., high=2*np.pi, size=None) - 2 * np.pi / params['wavelength_in_pixel2'] * np.sqrt((params['focal_x_offset2'] - superpixel_x)**2 + params['focal_length2']**2)
        
        Far_Hz1, Far_Hz2, pattern, FoM, Hz_f1, Hz_f2 = optimizeDualWavelengthsBestFromBatchDeflector(
            incidence_angle_deg=0.,  # Metalens uses normal incidence
            wvl1=params['wvl1'],
            wvl2=params['wvl2'],
            deflection_angle_deg1=focal_point_angle_deg1[i],
            deflection_angle_deg2=focal_point_angle_deg2[i],
            target_phase1=0+phase_by_section1[i],
            target_phase2=0+phase_by_section2[i],
            optim_distance=params['optim_distance'],
            batch_size=params['batch_size'],
            thickness=params['thickness'],
            seed=None,
            debug=params['ifDebug'],
            debug_dir_root=f"{params['plotDirectory']}/sp{i}/device",
            device_idx=i
        )
        
        # Convert tensors to CPU and detach before returning
        result = (
            i,
            Far_Hz1.cpu().detach().numpy() if torch.is_tensor(Far_Hz1) else Far_Hz1,
            Far_Hz2.cpu().detach().numpy() if torch.is_tensor(Far_Hz2) else Far_Hz2,
            pattern.cpu().detach().numpy() if torch.is_tensor(pattern) else pattern,
            float(FoM) if torch.is_tensor(FoM) else FoM,
            Hz_f1.cpu().detach().numpy() if torch.is_tensor(Hz_f1) else Hz_f1,
            Hz_f2.cpu().detach().numpy() if torch.is_tensor(Hz_f2) else Hz_f2
        )
        
        cleanup_gpu()
        return result
        
    except Exception as e:
        print(f"Error processing superpixel {i} on GPU {gpu_id}: {str(e)}")
        cleanup_gpu()
        return None          

def collect_dual_result(result):
    """Callback function to collect results and update progress bar for dual wavelength optimization"""
    global all_results, pbar
    if result is not None:
        idx, Far_Hz1, Far_Hz2, pattern, FoM, Hz_f1, Hz_f2 = result
        all_results[idx] = (Far_Hz1, Far_Hz2, pattern, FoM, Hz_f1, Hz_f2)
    pbar.update(1)

class DualOptimizationCallback:
    def __init__(self, plot_directory):
        self.plot_directory = plot_directory
    
    def __call__(self, result):
        """Callback function to collect results and update progress bar"""
        global all_results, pbar
        if result is not None:
            idx, Far_Hz1, Far_Hz2, pattern, FoM, Hz_f1, Hz_f2 = result
            all_results[idx] = (Far_Hz1, Far_Hz2, pattern, FoM, Hz_f1, Hz_f2)
        pbar.update(1)
        with open(f"{self.plot_directory}/progress.txt", 'w') as f:
            f.write(f"Optimizing superpixels: {pbar.n}/{pbar.total} ({(pbar.n/pbar.total)*100:.1f}%)")

def wavelength_to_rgb(wavelength_nm):
    """Convert wavelength (in nm) to RGB color values that approximate visual spectral colors."""
    # Based on the algorithm by Dan Bruton: http://www.physics.sfasu.edu/astro/color/spectra.html
    wavelength = wavelength_nm
    
    if wavelength >= 380 and wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif wavelength >= 440 and wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif wavelength >= 490 and wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif wavelength >= 510 and wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif wavelength >= 645 and wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    
    # Intensity adjustment
    if wavelength >= 380 and wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength >= 420 and wavelength <= 700:
        factor = 1.0
    elif wavelength > 700 and wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 0.0
        
    R = R * factor
    G = G * factor
    B = B * factor
    
    return (R, G, B)

def dualWavelengthsMetalens(physical_length_meter: float, 
                            wvl1: float, 
                            wvl2: float, 
                            focal_length_wavelength1_meter: float, 
                            focal_length_wavelength2_meter: float, 
                            focal_x_offset_wavelength1_meter: float, 
                            focal_x_offset_wavelength2_meter: float,
                            gpu_ids: list = [0],
                            feature_size_meter: float = 4e-8, 
                            thickness: float = 500e-9,
                            material_index = 2.4,
                            ifPlot=True,
                            ifDebug=False,
                            plotDirectory: str = "./results",
                            max_retries: int = 3,
                            batch_size: int = 60
                           ):
    """
    Parallelized version of dual wavelength metalens optimization
    """
    if not (4e-7 <= wvl1 <= 7e-7):
        raise ValueError("Wavelength 1 must be between 400nm and 700nm")

    if not (4e-7 <= wvl2 <= 7e-7):
        raise ValueError("Wavelength 2 must be between 400nm and 700nm")

    # if not (0 < physical_length_meter <= 1e-4):
    #     raise ValueError("The length of the deflector should be smaller than 100 um")
    if not (0 < physical_length_meter <= 500e-6):
        raise ValueError("The length of the metalens should be smaller than 500 um")
    
    if not (250e-9 <= thickness <= 500e-9):
        raise ValueError("The thickness of the metalens must be between 250 nm and 500 nm")
    
    neighbor_section_distance_in_pixel = 208
    num_superpixel = round((physical_length_meter * 1e9 / consts.dL) // neighbor_section_distance_in_pixel)
    wavelength_in_pixel1 = wvl1 * 1e9 / consts.dL
    wavelength_in_pixel2 = wvl2 * 1e9 / consts.dL
    feature_size_in_pixel = feature_size_meter * 1e9 / consts.dL
    focal_length1 = focal_length_wavelength1_meter * 1e9 / consts.dL
    focal_length2 = focal_length_wavelength2_meter * 1e9 / consts.dL
    focal_x_offset1 = focal_x_offset_wavelength1_meter * 1e9 / consts.dL
    focal_x_offset2 = focal_x_offset_wavelength2_meter * 1e9 / consts.dL
    optim_distance = 5.192301994298867e-05

    material_index_value = np.mean(material_index) if hasattr(material_index, "__iter__") else float(material_index)
    consts.eps_mat = float(material_index_value) ** 2

    setFeatureSize(feature_size_in_pixel)
    if not os.path.exists(plotDirectory):
        os.makedirs(plotDirectory)

    with open(f"{plotDirectory}/progress.txt", 'w') as f:
        f.write(f"Optimizing superpixels: 0/{num_superpixel} (0.0%)")

    # Package parameters for worker processes
    optimization_params = {
        'num_superpixel': num_superpixel,
        'wavelength_in_pixel1': wavelength_in_pixel1,
        'wavelength_in_pixel2': wavelength_in_pixel2,
        'focal_length1': focal_length1,
        'focal_length2': focal_length2,
        'focal_x_offset1': focal_x_offset1,
        'focal_x_offset2': focal_x_offset2,
        'neighbor_section_distance_in_pixel': neighbor_section_distance_in_pixel,
        'optim_distance': optim_distance,
        'wvl1': wvl1,
        'wvl2': wvl2,
        'feature_size_in_pixel': feature_size_in_pixel,
        'thickness': thickness,
        'ifDebug': ifDebug,
        'plotDirectory': plotDirectory,
        'batch_size': batch_size,
        'eps_mat': consts.eps_mat
    }

    # Initialize global variables for callback
    global all_results, pbar
    all_results = [None] * num_superpixel
    
    try:
        with multiprocessing.Pool(len(gpu_ids)) as pool:
            # Create tasks list
            tasks = [(i, gpu_ids[i % len(gpu_ids)], optimization_params) for i in range(num_superpixel)]
            
            # Initialize progress bar and callback
            with tqdm(total=num_superpixel, desc="Optimizing superpixels") as pbar:
                callback = DualOptimizationCallback(plotDirectory)
                # Submit all tasks
                for task in tasks:
                    pool.apply_async(optimize_dual_metalens_superpixel, 
                                   args=(task,),
                                   callback=callback)
                
                # Wait for completion
                pool.close()
                pool.join()

        # Check for failed tasks and retry if needed
        failed_indices = [i for i, result in enumerate(all_results) if result is None]
        retry_count = 0
        
        while failed_indices and retry_count < max_retries:
            retry_count += 1
            print(f"\nRetrying {len(failed_indices)} failed tasks (attempt {retry_count}/{max_retries})...")
            
            with multiprocessing.Pool(len(gpu_ids)) as pool:
                retry_tasks = [(i, gpu_ids[i % len(gpu_ids)], optimization_params) for i in failed_indices]
                
                with tqdm(total=len(failed_indices), desc="Retrying failed tasks") as pbar:
                    for task in retry_tasks:
                        pool.apply_async(optimize_dual_metalens_superpixel,
                                       args=(task,),
                                       callback=callback)
                    pool.close()
                    pool.join()
            
            failed_indices = [i for i, result in enumerate(all_results) if result is None]

        if failed_indices:
            print(f"\nWarning: {len(failed_indices)} tasks still failed after {max_retries} retries")
            raise RuntimeError("Not all superpixels were successfully optimized")

        # Unpack results
        all_Far_Hz1, all_Far_Hz2, all_best_patterns, all_FoMs, all_Hz_1, all_Hz_2 = zip(*all_results)

        full_pattern = stitchSuperPixels(all_best_patterns, neighbor_section_distance_in_pixel)
        full_pattern = postOptimizationFeatureSizeCorrection(full_pattern, round(feature_size_in_pixel), thickness)

        if ifPlot:
            # near field Hz
            Hz_wvl1_stitched = np.zeros((num_superpixel * round(neighbor_section_distance_in_pixel), consts.nn_y_pix - 2), dtype=np.complex64)
            Hz_wvl2_stitched = np.zeros((num_superpixel * round(neighbor_section_distance_in_pixel), consts.nn_y_pix - 2), dtype=np.complex64)
            for i in range(num_superpixel):
                Hz_wvl1_stitched[i * round(neighbor_section_distance_in_pixel) : (i+1) * round(neighbor_section_distance_in_pixel), :] = \
                    all_Hz_1[i][(consts.nn_x_pix-2 - round(neighbor_section_distance_in_pixel)) // 2 : \
                                  (consts.nn_x_pix-2 + round(neighbor_section_distance_in_pixel)) // 2 , :]
                Hz_wvl2_stitched[i * round(neighbor_section_distance_in_pixel) : (i+1) * round(neighbor_section_distance_in_pixel), :] = \
                    all_Hz_2[i][(consts.nn_x_pix-2 - round(neighbor_section_distance_in_pixel)) // 2 : \
                                  (consts.nn_x_pix-2 + round(neighbor_section_distance_in_pixel)) // 2 , :]

            # Create a custom colormap with white at 0
            colors = ['blue', 'white', 'red']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            # wavelength 1
            plt.figure(figsize=(24,4))
            ax1 = plt.subplot(1,1,1)
            Hz_plot1 = np.real(Hz_wvl1_stitched.T)

            vmin, vmax = np.min(Hz_plot1), np.max(Hz_plot1)
            vabs = max(abs(vmin), abs(vmax))*0.8
            
            im1 = ax1.imshow(Hz_plot1, cmap=custom_cmap, origin='lower', vmin=-vabs, vmax=vabs)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"Near Field Hz at the wavelength of {round(wvl1 * 1e9, 2)} nm")
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="1%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.set_ticks([-vabs, 0, vabs])
            cbar1.set_ticklabels([f'{-vabs:.2e}', '0', f'{vabs:.2e}'])
            cbar1.ax.tick_params(labelsize=12)

            plt.savefig(f'{plotDirectory}/nearfield_Hz_wavelength_1.png', bbox_inches='tight', dpi = 600)
            plt.close()

            # wavelength 2
            plt.figure(figsize=(24,4))
            ax2 = plt.subplot(1,1,1)
            Hz_plot2 = np.real(Hz_wvl2_stitched.T)

            vmin, vmax = np.min(Hz_plot2), np.max(Hz_plot2)
            vabs = max(abs(vmin), abs(vmax))*0.8
            
            im2 = ax2.imshow(Hz_plot2, cmap=custom_cmap, origin='lower', vmin=-vabs, vmax=vabs)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(f"Near Field Hz at the wavelength of {round(wvl2 * 1e9, 2)} nm")
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="1%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_ticks([-vabs, 0, vabs])
            cbar2.set_ticklabels([f'{-vabs:.2e}', '0', f'{vabs:.2e}'])
            cbar2.ax.tick_params(labelsize=12)

            plt.savefig(f'{plotDirectory}/nearfield_Hz_wavelength_2.png', bbox_inches='tight', dpi = 600)
            plt.close()

            # focusing plots
            # Wavelength 1
            grid_z = np.arange(0, 1.5 * focal_length_wavelength1_meter * 1e6, 0.1)
            window_size = 1e6 * min(physical_length_meter * 10, 
                                    max(2.6 * abs(focal_x_offset_wavelength1_meter), 
                                        physical_length_meter
                                       )
                                   )
            farfield_wvl1, grid_x_wvl1 = angularSpectrumMethod(Hz_wvl1_stitched[:,-1], grid_z, window_size, wvl1 * 1e6)
            plt.figure(figsize=(5,10))
            plt.pcolor(grid_x_wvl1, grid_z, np.abs(farfield_wvl1)**2)
            plt.xlabel('x (um)')
            plt.ylabel('z (um)')
            plt.title(f'Far Field Intensity at the Wavelength of {round(wvl1 * 1e6, 2)} um')
            plt.savefig(f'{plotDirectory}/farfield_intensity_wavelength_1.png', bbox_inches='tight', dpi = 600)
            plt.close()

            grid_z = np.arange(0, 1.5 * focal_length_wavelength2_meter * 1e6, 0.1)
            window_size = 1e6 * min(physical_length_meter * 10, 
                                    max(2.6 * abs(focal_x_offset_wavelength2_meter),
                                        physical_length_meter
                                       )
                                   )
            farfield_wvl2, grid_x_wvl2 = angularSpectrumMethod(Hz_wvl2_stitched[:,-1], grid_z, window_size, wvl2 * 1e6)
            plt.figure(figsize=(5,10))
            plt.pcolor(grid_x_wvl2, grid_z, np.abs(farfield_wvl2)**2)
            plt.xlabel('x (um)')
            plt.ylabel('z (um)')
            plt.title(f'Far Field Intensity at the Wavelength of {round(wvl2 * 1e6, 2)} um')
            plt.savefig(f'{plotDirectory}/farfield_intensity_wavelength_2.png', bbox_inches='tight', dpi = 600)
            plt.close()
            
            # device structure plot
            colors = [(1, 1, 1), (30/255, 30/255, 30/255)]  # White to dark gray (RGB 30, 30, 30)
            n_bins = 100  # Number of color gradations
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
            
            # Set min and max values for the colormap
            min_val = 1  # Minimum epsilon (air)
            max_val = consts.eps_mat  # Maximum epsilon (material)
            
            plt.figure(figsize=(24,4))
            plt.imshow(full_pattern.T, origin='lower', cmap=custom_cmap, vmin=min_val, vmax=max_val)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{plotDirectory}/full_pattern.png', bbox_inches='tight', dpi = 600)
            plt.close()

        return full_pattern

    except Exception as e:
        print(f"Error during parallel optimization: {str(e)}")
        cleanup_gpu()
        raise
    
    finally:
        cleanup_gpu()

    return full_pattern


def optimize_single_metalens_superpixel(args):
    """Helper function to optimize a single metalens superpixel on specified GPU"""
    i, gpu_id, params = args
    try:
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cleanup_gpu()
        consts.eps_mat = params.get('eps_mat', consts.eps_mat)
        
        superpixel_x = neighbor_section_distance_in_pixel = params['neighbor_section_distance_in_pixel']
        superpixel_x = neighbor_section_distance_in_pixel * (np.arange(params['num_superpixel']) - params['num_superpixel'] * 0.5 + 0.5)
        focal_point_angle_deg1 = - np.rad2deg(np.arctan2(params['focal_x_offset1'] - superpixel_x, params['focal_length1']))
        phase_by_section1 = - 2 * np.pi / params['wavelength_in_pixel1'] * np.sqrt((params['focal_x_offset1'] - superpixel_x)**2 + params['focal_length1']**2)
        
        Far_Hz1, pattern, FoM, Hz_f1 = optimizeSingleWavelengthBestFromBatchDeflector(
            incidence_angle_deg=0.,  # Metalens uses normal incidence
            wvl1=params['wvl1'],
            deflection_angle_deg1=focal_point_angle_deg1[i],
            target_phase1=0+phase_by_section1[i],
            optim_distance=params['optim_distance'],
            batch_size=params['batch_size'],
            thickness=params['thickness'],
            seed=None,
            debug=params['ifDebug'],
            debug_dir_root=f"{params['plotDirectory']}/sp{i}/device",
            device_idx=i
        )
        
        # Convert tensors to CPU and detach before returning
        result = (
            i,
            Far_Hz1.cpu().detach().numpy() if torch.is_tensor(Far_Hz1) else Far_Hz1,
            pattern.cpu().detach().numpy() if torch.is_tensor(pattern) else pattern,
            float(FoM) if torch.is_tensor(FoM) else FoM,
            Hz_f1.cpu().detach().numpy() if torch.is_tensor(Hz_f1) else Hz_f1
        )
        
        cleanup_gpu()
        return result
        
    except Exception as e:
        print(f"Error processing superpixel {i} on GPU {gpu_id}: {str(e)}")
        cleanup_gpu()
        return None


def singleWavelengthMetalens(physical_length_meter: float, 
                            wvl1: float, 
                            focal_length_wavelength1_meter: float, 
                            focal_x_offset_wavelength1_meter: float,
                            gpu_ids: list = [0],
                            feature_size_meter: float = 4e-8, 
                            thickness: float = 500e-9,
                            material_index = 2.4,
                            ifPlot=True,  
                            ifDebug=False,
                            plotDirectory: str = "./results",
                            max_retries: int = 3,
                            batch_size: int = 60
                           ):
    """
    Parallelized version of single wavelength metalens optimization
    """
    if not (4e-7 <= wvl1 <= 7e-7):
        raise ValueError("Wavelength 1 must be between 400nm and 700nm")

    # if not (0 < physical_length_meter <= 1e-4):
    #     raise ValueError("The length of the deflector should be smaller than 100 um")
    if not (0 < physical_length_meter <= 500e-6):
        raise ValueError("The length of the metalens should be smaller than 500 um")
    
    if not (250e-9 <= thickness <= 500e-9):
        raise ValueError("The thickness of the metalens must be between 250 nm and 500 nm")
    
    neighbor_section_distance_in_pixel = 208
    num_superpixel = round((physical_length_meter * 1e9 / consts.dL) // neighbor_section_distance_in_pixel)
    wavelength_in_pixel1 = wvl1 * 1e9 / consts.dL
    feature_size_in_pixel = feature_size_meter * 1e9 / consts.dL
    focal_length1 = focal_length_wavelength1_meter * 1e9 / consts.dL
    focal_x_offset1 = focal_x_offset_wavelength1_meter * 1e9 / consts.dL
    optim_distance = 5.192301994298867e-05

    material_index_value = np.mean(material_index) if hasattr(material_index, "__iter__") else float(material_index)
    consts.eps_mat = float(material_index_value) ** 2

    setFeatureSize(feature_size_in_pixel)
    if not os.path.exists(plotDirectory):
        os.makedirs(plotDirectory)

    with open(f"{plotDirectory}/progress.txt", 'w') as f:
        f.write(f"Optimizing superpixels: 0/{num_superpixel} (0.0%)")

    # Package parameters for worker processes
    optimization_params = {
        'num_superpixel': num_superpixel,
        'wavelength_in_pixel1': wavelength_in_pixel1,
        'focal_length1': focal_length1,
        'focal_x_offset1': focal_x_offset1,
        'neighbor_section_distance_in_pixel': neighbor_section_distance_in_pixel,
        'optim_distance': optim_distance,
        'wvl1': wvl1,
        'feature_size_in_pixel': feature_size_in_pixel,
        'thickness': thickness,
        'ifDebug': ifDebug,
        'plotDirectory': plotDirectory,
        'batch_size': batch_size,
        'eps_mat': consts.eps_mat
    }

    # Initialize global variables for callback
    global all_results, pbar
    all_results = [None] * num_superpixel
    
    try:
        with multiprocessing.Pool(len(gpu_ids)) as pool:
            # Create tasks list
            tasks = [(i, gpu_ids[i % len(gpu_ids)], optimization_params) for i in range(num_superpixel)]
            
            # Initialize progress bar and callback
            with tqdm(total=num_superpixel, desc="Optimizing superpixels") as pbar:
                callback = OptimizationCallback(plotDirectory)
                # Submit all tasks
                for task in tasks:
                    pool.apply_async(optimize_single_metalens_superpixel, 
                                   args=(task,),
                                   callback=callback)
                
                # Wait for completion
                pool.close()
                pool.join()

        # Check for failed tasks and retry if needed
        failed_indices = [i for i, result in enumerate(all_results) if result is None]
        retry_count = 0
        
        while failed_indices and retry_count < max_retries:
            retry_count += 1
            print(f"\nRetrying {len(failed_indices)} failed tasks (attempt {retry_count}/{max_retries})...")
            
            with multiprocessing.Pool(len(gpu_ids)) as pool:
                retry_tasks = [(i, gpu_ids[i % len(gpu_ids)], optimization_params) for i in failed_indices]
                
                with tqdm(total=len(failed_indices), desc="Retrying failed tasks") as pbar:
                    for task in retry_tasks:
                        pool.apply_async(optimize_single_metalens_superpixel,
                                       args=(task,),
                                       callback=callback)
                    pool.close()
                    pool.join()
            
            failed_indices = [i for i, result in enumerate(all_results) if result is None]

        if failed_indices:
            print(f"\nWarning: {len(failed_indices)} tasks still failed after {max_retries} retries")
            raise RuntimeError("Not all superpixels were successfully optimized")

        # Unpack results
        all_Far_Hz1, all_best_patterns, all_FoMs, all_Hz_1 = zip(*all_results)

        full_pattern = stitchSuperPixels(all_best_patterns, neighbor_section_distance_in_pixel)
        full_pattern = postOptimizationFeatureSizeCorrection(full_pattern, round(feature_size_in_pixel), thickness)

        if ifPlot:
            # near field Hz
            Hz_wvl1_stitched = np.zeros((num_superpixel * round(neighbor_section_distance_in_pixel), consts.nn_y_pix - 2), dtype=np.complex64)
            for i in range(num_superpixel):
                Hz_wvl1_stitched[i * round(neighbor_section_distance_in_pixel) : (i+1) * round(neighbor_section_distance_in_pixel), :] = \
                    all_Hz_1[i][(consts.nn_x_pix-2 - round(neighbor_section_distance_in_pixel)) // 2 : \
                                  (consts.nn_x_pix-2 + round(neighbor_section_distance_in_pixel)) // 2 , :]

            # Create a custom colormap with white at 0
            colors = ['blue', 'white', 'red']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            # wavelength 1
            plt.figure(figsize=(24,4))
            ax1 = plt.subplot(1,1,1)
            Hz_plot1 = np.real(Hz_wvl1_stitched.T)

            vmin, vmax = np.min(Hz_plot1), np.max(Hz_plot1)
            vabs = max(abs(vmin), abs(vmax))*0.8
            
            im1 = ax1.imshow(Hz_plot1, cmap=custom_cmap, origin='lower', vmin=-vabs, vmax=vabs)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"Near Field Hz at the wavelength of {round(wvl1 * 1e9, 2)} nm")
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="1%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.set_ticks([-vabs, 0, vabs])
            cbar1.set_ticklabels([f'{-vabs:.2e}', '0', f'{vabs:.2e}'])
            cbar1.ax.tick_params(labelsize=12)

            plt.savefig(f'{plotDirectory}/nearfield_Hz_wavelength_1.png', bbox_inches='tight', dpi = 600)
            plt.close()

            # focusing plot
            grid_z = np.arange(0, 1.5 * focal_length_wavelength1_meter * 1e6, 0.1)
            window_size = 1e6 * min(physical_length_meter * 10, 
                                    max(2.6 * abs(focal_x_offset_wavelength1_meter), 
                                        physical_length_meter
                                       )
                                   )
            farfield_wvl1, grid_x_wvl1 = angularSpectrumMethod(Hz_wvl1_stitched[:,-1], grid_z, window_size, wvl1 * 1e6)
            plt.figure(figsize=(5,10))
            plt.pcolor(grid_x_wvl1, grid_z, np.abs(farfield_wvl1)**2)
            plt.xlabel('x (um)')
            plt.ylabel('z (um)')
            plt.title(f'Far Field Intensity at the Wavelength of {round(wvl1 * 1e6, 2)} um')
            plt.savefig(f'{plotDirectory}/farfield_intensity_wavelength_1.png', bbox_inches='tight', dpi = 600)
            plt.close()

            # device structure plot
            colors = [(1, 1, 1), (30/255, 30/255, 30/255)]  # White to dark gray (RGB 30, 30, 30)
            n_bins = 100  # Number of color gradations
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
            
            # Set min and max values for the colormap
            min_val = 1  # Minimum epsilon (air)
            max_val = consts.eps_mat  # Maximum epsilon (material)
            
            plt.figure(figsize=(24,4))
            plt.imshow(full_pattern.T, origin='lower', cmap=custom_cmap, vmin=min_val, vmax=max_val)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{plotDirectory}/full_pattern.png', bbox_inches='tight', dpi = 600)
            plt.close()

            ## Saved for debugging characterization
            np.save(f'{plotDirectory}/farfield_intensity_wavelength_1.npy', np.abs(farfield_wvl1)**2)
            np.save(f'{plotDirectory}/grid_z.npy', grid_z)           # grid_z is already a NumPy array
            np.save(f'{plotDirectory}/grid_x.npy', grid_x_wvl1)

        return full_pattern

    except Exception as e:
        print(f"Error during parallel optimization: {str(e)}")
        cleanup_gpu()
        raise
    
    finally:
        cleanup_gpu()

    return full_pattern


def cleanup_gpu():
    """Helper function to clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    # Sleep briefly to ensure memory is freed
    time.sleep(1)

def optimize_single_superpixel(args):
    """Helper function to optimize a single superpixel on specified GPU"""
    i, gpu_id, params = args
    try:
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cleanup_gpu()
        consts.eps_mat = params.get('eps_mat', consts.eps_mat)
        
        superpixel_x = neighbor_section_distance_in_pixel = params['neighbor_section_distance_in_pixel']
        superpixel_x = neighbor_section_distance_in_pixel * (np.arange(params['num_superpixel']) - params['num_superpixel'] * 0.5 + 0.5)
        phase_by_section1 = 2 * np.pi / params['wavelength_in_pixel1'] * superpixel_x * np.sin(np.deg2rad(-params['deflection_angle_deg_wavelength1']))
        
        Far_Hz1, pattern, FoM, Hz_f1 = optimizeSingleWavelengthBestFromBatchDeflector(
            incidence_angle_deg=params['incidence_angle_deg'],
            wvl1=params['wvl1'],
            deflection_angle_deg1=-params['deflection_angle_deg_wavelength1'],
            target_phase1=0+phase_by_section1[i],
            optim_distance=params['optim_distance'],
            batch_size=params['batch_size'],
            thickness=params['thickness'],
            seed=None,
            debug=params['ifDebug'],
            debug_dir_root=f"{params['plotDirectory']}/sp{i}/device",
            device_idx=i
        )
        
        # Convert tensors to CPU and detach before returning
        result = (
            i,
            Far_Hz1.cpu().detach().numpy() if torch.is_tensor(Far_Hz1) else Far_Hz1,
            pattern.cpu().detach().numpy() if torch.is_tensor(pattern) else pattern,
            float(FoM) if torch.is_tensor(FoM) else FoM,
            Hz_f1.cpu().detach().numpy() if torch.is_tensor(Hz_f1) else Hz_f1
        )
        
        cleanup_gpu()
        return result
        
    except Exception as e:
        print(f"Error processing superpixel {i} on GPU {gpu_id}: {str(e)}")
        cleanup_gpu()
        return None

def collect_result(result):
    """Callback function to collect results and update progress bar"""
    global all_results
    global pbar
    global plotDirectory
    if result is not None:
        idx, Far_Hz1, pattern, FoM, Hz_f1 = result
        all_results[idx] = (Far_Hz1, pattern, FoM, Hz_f1)
    pbar.update(1)
    with open(f"{plotDirectory}/progress.txt", 'w') as f:
        f.write(f"Optimizing superpixels: {pbar.n}/{pbar.total} ({(pbar.n/pbar.total)*100:.1f}%)")

class OptimizationCallback:
    def __init__(self, plot_directory):
        self.plot_directory = plot_directory
    
    def __call__(self, result):
        """Callback function to collect results and update progress bar"""
        global all_results, pbar
        if result is not None:
            idx, Far_Hz1, pattern, FoM, Hz_f1 = result
            all_results[idx] = (Far_Hz1, pattern, FoM, Hz_f1)
        pbar.update(1)
        with open(f"{self.plot_directory}/progress.txt", 'w') as f:
            f.write(f"Optimizing superpixels: {pbar.n}/{pbar.total} ({(pbar.n/pbar.total)*100:.1f}%)")

def singleWavelengthDeflector(physical_length_meter: float, 
                            wvl1: float, 
                            incidence_angle_deg: float, 
                            deflection_angle_deg_wavelength1: float,
                            gpu_ids: list = [0],
                            feature_size_meter: float = 4e-8, 
                            thickness: float = 500e-9,
                            material_index = 2.4,
                            ifPlot=True,  
                            ifDebug=False,
                            plotDirectory: str = "./results",
                            max_retries: int = 3,
                            batch_size: int = 60
                           ):
    """
    Modified to use apply_async for non-blocking progress updates
    """
    # Input validation
    if not (4e-7 <= wvl1 <= 7e-7):
        raise ValueError("Wavelength 1 must be between 400nm and 700nm")

    # if not (0 < physical_length_meter <= 1e-4):
    #     raise ValueError("The length of the deflector should be smaller than 100 um")
    if not (0 < physical_length_meter <= 500e-6):
        raise ValueError("The length of the deflector must be smaller than 500 um")

    if not (250e-9 <= thickness <= 500e-9):
        raise ValueError("The thickness of the deflector must be between 250 nm and 500 nm")

    if not (-5. < incidence_angle_deg < 25.):
        raise ValueError("Incident angle must be within -5 to 25 degrees.")
    if not (-90. < deflection_angle_deg_wavelength1 < 90.):
        raise ValueError("Deflection angle for wavelength 1 must be within -90 to 90 degrees.")
    
    neighbor_section_distance_in_pixel = 208
    num_superpixel = round((physical_length_meter * 1e9 / consts.dL) // neighbor_section_distance_in_pixel)
    wavelength_in_pixel1 = wvl1 * 1e9 / consts.dL
    feature_size_in_pixel = feature_size_meter * 1e9 / consts.dL
    optim_distance = 5.192301994298867e-05

    material_index_value = np.mean(material_index) if hasattr(material_index, "__iter__") else float(material_index)
    consts.eps_mat = float(material_index_value) ** 2

    setFeatureSize(feature_size_in_pixel)
    if not os.path.exists(plotDirectory):
        os.makedirs(plotDirectory)

    with open(f"{plotDirectory}/progress.txt", 'w') as f:
        f.write(f"Optimizing superpixels: 0/{num_superpixel} (0.0%)")

    # Package parameters for worker processes
    optimization_params = {
        'num_superpixel': num_superpixel,
        'wavelength_in_pixel1': wavelength_in_pixel1,
        'incidence_angle_deg': incidence_angle_deg,
        'deflection_angle_deg_wavelength1': deflection_angle_deg_wavelength1,
        'neighbor_section_distance_in_pixel': neighbor_section_distance_in_pixel,
        'optim_distance': optim_distance,
        'wvl1': wvl1,
        'feature_size_in_pixel': feature_size_in_pixel,
        'thickness': thickness,
        'ifDebug': ifDebug,
        'plotDirectory': plotDirectory,
        'batch_size': batch_size,
        'eps_mat': consts.eps_mat
    }

    # Initialize global variables for callback
    global all_results, pbar
    all_results = [None] * num_superpixel
    
    try:
        with multiprocessing.Pool(len(gpu_ids)) as pool:
            # Create tasks list
            tasks = [(i, gpu_ids[i % len(gpu_ids)], optimization_params) for i in range(num_superpixel)]
            
            # Initialize progress bar and callback
            with tqdm(total=num_superpixel, desc="Optimizing superpixels") as pbar:
                callback = OptimizationCallback(plotDirectory)
                # Submit all tasks
                for task in tasks:
                    pool.apply_async(optimize_single_superpixel, 
                                   args=(task,),
                                   callback=callback)
                
                # Wait for completion
                pool.close()
                pool.join()

        # Check for failed tasks and retry if needed
        failed_indices = [i for i, result in enumerate(all_results) if result is None]
        retry_count = 0
        
        while failed_indices and retry_count < max_retries:
            retry_count += 1
            print(f"\nRetrying {len(failed_indices)} failed tasks (attempt {retry_count}/{max_retries})...")
            
            with multiprocessing.Pool(len(gpu_ids)) as pool:
                retry_tasks = [(i, gpu_ids[i % len(gpu_ids)], optimization_params) for i in failed_indices]
                
                with tqdm(total=len(failed_indices), desc="Retrying failed tasks") as pbar:
                    for task in retry_tasks:
                        pool.apply_async(optimize_single_superpixel,
                                       args=(task,),
                                       callback=callback)
                    pool.close()
                    pool.join()
            
            failed_indices = [i for i, result in enumerate(all_results) if result is None]

        if failed_indices:
            print(f"\nWarning: {len(failed_indices)} tasks still failed after {max_retries} retries")
            raise RuntimeError("Not all superpixels were successfully optimized")

        # Unpack results
        all_Far_Hz1, all_best_patterns, all_FoMs, all_Hz_1 = zip(*all_results)

        full_pattern = stitchSuperPixels(all_best_patterns, neighbor_section_distance_in_pixel)
        full_pattern = postOptimizationFeatureSizeCorrection(full_pattern, round(feature_size_in_pixel), thickness)

        if ifPlot:
            # near field Hz
            Hz_wvl1_stitched = np.zeros((num_superpixel * round(neighbor_section_distance_in_pixel), consts.nn_y_pix - 2), dtype=np.complex64)
            for i in range(num_superpixel):
                Hz_wvl1_stitched[i * round(neighbor_section_distance_in_pixel) : (i+1) * round(neighbor_section_distance_in_pixel), :] = \
                    all_Hz_1[i][(consts.nn_x_pix-2 - round(neighbor_section_distance_in_pixel)) // 2 : \
                                  (consts.nn_x_pix-2 + round(neighbor_section_distance_in_pixel)) // 2 , :]

            # Create a custom colormap with white at 0
            colors = ['blue', 'white', 'red']
            n_bins = 256
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

            # wavelength 1
            plt.figure(figsize=(24,4))
            ax1 = plt.subplot(1,1,1)
            Hz_plot1 = np.real(Hz_wvl1_stitched.T)

            vmin, vmax = np.min(Hz_plot1), np.max(Hz_plot1)
            vabs = max(abs(vmin), abs(vmax))*0.8
            
            im1 = ax1.imshow(Hz_plot1, cmap=custom_cmap, origin='lower', vmin=-vabs, vmax=vabs)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"Near Field Hz at the wavelength of {round(wvl1 * 1e9, 2)} nm")
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="1%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax1)
            cbar1.set_ticks([-vabs, 0, vabs])
            cbar1.set_ticklabels([f'{-vabs:.2e}', '0', f'{vabs:.2e}'])
            cbar1.ax.tick_params(labelsize=12)

            plt.savefig(f'{plotDirectory}/nearfield_Hz_wavelength_1.png', bbox_inches='tight', dpi = 600)
            plt.close()

            
            # farfield plot
            eps_r_batch = torch.zeros((num_superpixel, consts.Nx, consts.Ny))
            for i in range(num_superpixel):
                eps_r_batch[i] = torch.tensor(postOptimizationFeatureSizeCorrection(
                                        buildDielectricDistribution(
                                            torch.tensor(all_best_patterns[i]),
                                            (consts.Nx, consts.Ny),
                                            [consts.x_start, consts.x_end, consts.y_start, consts.y_end]
                                        ).numpy(), round(feature_size_in_pixel), thickness))
            plotSingleWavelengthFarField(eps_r_batch.cuda(), incidence_angle_deg, wvl1, -deflection_angle_deg_wavelength1, 
                         optim_distance, neighbor_section_distance_in_pixel, plotDirectory)


            # device structure plot
            colors = [(1, 1, 1), (30/255, 30/255, 30/255)]  # White to dark gray (RGB 30, 30, 30)
            n_bins = 100  # Number of color gradations
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
            
            # Set min and max values for the colormap
            min_val = 1  # Minimum epsilon (air)
            max_val = consts.eps_mat  # Maximum epsilon (material)
            
            plt.figure(figsize=(24,4))
            plt.imshow(full_pattern.T, origin='lower', cmap=custom_cmap, vmin=min_val, vmax=max_val)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{plotDirectory}/full_pattern.png', bbox_inches='tight', dpi = 600)
            plt.close()


        return full_pattern

    except Exception as e:
        print(f"Error during parallel optimization: {str(e)}")
        cleanup_gpu()
        raise
    
    finally:
        cleanup_gpu()

    return full_pattern